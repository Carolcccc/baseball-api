"""Train a quick XGBoost baseline using per-player and vs aggregates.

This script builds a simple dataset by joining batter/pitcher aggregates with
vs aggregates. Target: whether batter had any hits vs that pitcher (hits_vs>0).

Outputs:
- models/baseline_xgb.joblib
- models/feature_columns.txt

Note: This is a baseline demonstrator; a production model needs time-based
folding to avoid leakage and better feature engineering.
"""
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import xgboost as xgb


def load_data(data_dir: Path):
    # Prefer smoothed per-role files if they exist (produced by ETL smoothing step).
    batter_candidates = [data_dir / 'features_batter_smoothed.parquet', data_dir / 'features_batter_2024_2025.parquet', data_dir / 'features_batter.parquet']
    pitcher_candidates = [data_dir / 'features_pitcher_smoothed.parquet', data_dir / 'features_pitcher_2024_2025.parquet', data_dir / 'features_pitcher.parquet']

    def choose(cands):
        for p in cands:
            if p.exists():
                return p
        raise FileNotFoundError(f'No candidate files found in {cands}')

    batter = pd.read_parquet(choose(batter_candidates))
    pitcher = pd.read_parquet(choose(pitcher_candidates))
    vs = pd.read_parquet(data_dir / 'features_vs_2024_2025.parquet')
    return batter, pitcher, vs


def build_dataset(batter, pitcher, vs):
    # Robustly select available columns from the produced feature files.
    # Batter file may contain ['batter','pa','hits','hit_rate'] or season/rolling names.
    batter = batter.copy()
    pitcher = pitcher.copy()
    vs = vs.copy()

    # normalize batter columns
    if 'player' in batter.columns:
        batter = batter.rename(columns={'player': 'player'})
    if 'batter' in batter.columns:
        batter = batter.rename(columns={'batter': 'player'})
    # map known batter fields to standardized names
    if 'pa_season' in batter.columns:
        batter = batter.rename(columns={'pa_season': 'pa_season', 'hits_season': 'hits_season', 'hit_rate_season': 'hit_rate_season'})
    elif 'pa' in batter.columns:
        batter = batter.rename(columns={'pa': 'pa_season', 'hits': 'hits_season', 'hit_rate': 'hit_rate_season'})

    # normalize pitcher
    if 'player' in pitcher.columns:
        pitcher = pitcher.rename(columns={'player': 'player'})
    if 'pitcher' in pitcher.columns:
        pitcher = pitcher.rename(columns={'pitcher': 'player'})
    if 'pa_season' in pitcher.columns:
        pitcher = pitcher.rename(columns={'pa_season': 'pa_season', 'hits_season': 'hits_season', 'hit_rate_season': 'hit_rate_season'})
    else:
        # try mapping from earlier names
        rename_map = {}
        if 'pa_faced' in pitcher.columns:
            rename_map['pa_faced'] = 'pa_season'
        if 'hits_allowed' in pitcher.columns:
            rename_map['hits_allowed'] = 'hits_season'
        if 'opp_hit_rate' in pitcher.columns:
            rename_map['opp_hit_rate'] = 'hit_rate_season'
        if 'avg_velo' in pitcher.columns:
            rename_map['avg_velo'] = 'avg_velo_7d'
        if rename_map:
            pitcher = pitcher.rename(columns=rename_map)

    # Prepare merge
    df = vs.copy()
    # If vs table is large, sample it to limit memory during local training runs
    max_vs = 20_000
    if len(df) > max_vs:
        df = df.sample(n=max_vs, random_state=42)
        print(f"Sampling vs table down to {len(df)} rows to limit memory usage")
    # ensure vs has player columns named 'batter' and 'pitcher'
    if 'batter' not in df.columns and 'player' in df.columns:
        df = df.rename(columns={'player': 'batter'})

    # merge batter and pitcher features
    df = df.merge(batter.add_prefix('batter_'), left_on='batter', right_on='batter_player', how='left')
    df = df.merge(pitcher.add_prefix('pitcher_'), left_on='pitcher', right_on='pitcher_player', how='left')

    # target: any hits vs
    df['target'] = (df.get('hits_vs', 0) > 0).astype(int)

    # collect numeric feature columns
    feature_cols = [c for c in df.columns if c.startswith('batter_') or c.startswith('pitcher_')]
    # drop identifier duplicates (player columns)
    feature_cols = [c for c in feature_cols if not c.endswith('_player')]

    X = df[feature_cols].fillna(0)
    y = df['target']
    return X, y, feature_cols


def train_and_save(X, y, feature_cols, out_dir: Path):
    # If dataset is large, sample a smaller stratified subset to avoid OOM during local runs
    max_rows = 200_000
    if len(X) > max_rows:
        frac = float(max_rows) / float(len(X))
        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=frac, random_state=42, stratify=y)
        X_used, y_used = X_sample, y_sample
        print(f"Dataset large ({len(X)} rows), sampling down to {len(X_used)} rows for training")
    else:
        X_used, y_used = X, y

    X_train, X_val, y_train, y_val = train_test_split(X_used, y_used, test_size=0.2, random_state=42, stratify=y_used)

    model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', tree_method='hist')
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, preds)
    print(f'Validation AUC: {auc:.4f}')

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / 'baseline_xgb.joblib'
    joblib.dump(model, model_path)
    (out_dir / 'feature_columns.txt').write_text('\n'.join(feature_cols))
    print(f'Saved model to {model_path}')
    return model_path


def main():
    # compute package inner path: .../baseball_api_mvp/baseball_api_mvp
    outer = Path(__file__).resolve().parents[1]
    pkg_root = outer / 'baseball_api_mvp'
    data_dir = pkg_root / 'data'
    out_dir = pkg_root / 'models'

    print('Loading data...')
    batter, pitcher, vs = load_data(data_dir)
    print('Building dataset...')
    X, y, feature_cols = build_dataset(batter, pitcher, vs)
    print(f'Rows: {len(X)}, Features: {len(feature_cols)}')

    print('Training model...')
    train_and_save(X, y, feature_cols, out_dir)


if __name__ == '__main__':
    main()
