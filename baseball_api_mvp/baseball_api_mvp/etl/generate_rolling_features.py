"""Generate rolling (7d, 30d) and season features from per-month Statcast parquet files.

Output: `data/features_statcast.parquet` containing per-player-date features for batters and pitchers.

This is a lightweight, prototype implementation intended to produce training-ready
features. It computes per-plate-appearance aggregates, then daily aggregations,
then rolling windows (7-day, 30-day) and season-to-date aggregates.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def load_all_statcast(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob('statcast_*.parquet'))
    if not files:
        raise FileNotFoundError('No statcast_*.parquet files found in data dir')
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def make_pa_level(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required columns
    df = df.copy()
    df['game_date'] = pd.to_datetime(df['game_date']).dt.normalize()

    # create is_hit/is_k/is_bb indicators from events
    df['is_hit'] = df.get('events', '').isin(['single', 'double', 'triple', 'home_run']).astype(int)
    df['is_k'] = (df.get('events', '') == 'strikeout').astype(int)
    df['is_bb'] = (df.get('events', '') == 'walk').astype(int)

    # PA identifier
    df['pa_id'] = df['game_pk'].astype(str) + '_' + df['at_bat_number'].astype(str)

    # aggregate to plate-appearance level per batter
    pa = (
        df.groupby(['batter', 'game_date', 'game_pk', 'at_bat_number', 'pa_id'])
        .agg(
            pa_is_hit=('is_hit', 'max'),
            pa_is_k=('is_k', 'max'),
            pa_is_bb=('is_bb', 'max'),
            pa_avg_velo=('velo', 'mean')
        )
        .reset_index()
    )
    return pa


def daily_player_stats(pa: pd.DataFrame, player_col: str) -> pd.DataFrame:
    # pa expected to have columns: player_col (batter/pitcher), game_date, pa_id, pa_is_hit, pa_is_k, pa_is_bb, pa_avg_velo
    g = pa.groupby([player_col, 'game_date']).agg(
        pa=('pa_id', 'nunique'),
        hits=('pa_is_hit', 'sum'),
        ks=('pa_is_k', 'sum'),
        walks=('pa_is_bb', 'sum'),
        avg_velo=('pa_avg_velo', 'mean')
    ).reset_index()
    # fill NaN avg_velo
    g['avg_velo'] = g['avg_velo'].fillna(0.0)
    return g


def compute_rolling(df_daily: pd.DataFrame, player_col: str) -> pd.DataFrame:
    df_daily = df_daily.copy()
    df_daily = df_daily.sort_values([player_col, 'game_date'])
    out_rows = []

    for player, grp in df_daily.groupby(player_col):
        grp = grp.set_index('game_date').asfreq('D', fill_value=0).reset_index()
        grp[player_col] = player
        grp['date'] = grp['game_date']

        # 7-day rolling (use DatetimeIndex; call rolling without `on` after set_index)
        grp_indexed = grp.set_index('date')
        roll7 = grp_indexed.rolling('7D', min_periods=1).agg({'pa':'sum','hits':'sum','ks':'sum','walks':'sum','avg_velo':'mean'}).reset_index()
        roll30 = grp_indexed.rolling('30D', min_periods=1).agg({'pa':'sum','hits':'sum','ks':'sum','walks':'sum','avg_velo':'mean'}).reset_index()

        merged = grp[['date', player_col]].copy()
        merged['pa_7d'] = roll7['pa'].values
        merged['hits_7d'] = roll7['hits'].values
        merged['hit_rate_7d'] = merged['hits_7d'] / merged['pa_7d'].replace(0, np.nan)
        merged['pa_30d'] = roll30['pa'].values
        merged['hits_30d'] = roll30['hits'].values
        merged['hit_rate_30d'] = merged['hits_30d'] / merged['pa_30d'].replace(0, np.nan)
        merged['avg_velo_7d'] = roll7['avg_velo'].values
        merged['avg_velo_30d'] = roll30['avg_velo'].values

        out_rows.append(merged)

    out = pd.concat(out_rows, ignore_index=True)
    # fill NaN rates with 0 when pa==0
    out['hit_rate_7d'] = out['hit_rate_7d'].fillna(0.0)
    out['hit_rate_30d'] = out['hit_rate_30d'].fillna(0.0)
    return out


def season_aggregates(pa: pd.DataFrame, player_col: str) -> pd.DataFrame:
    pa = pa.copy()
    pa['year'] = pa['game_date'].dt.year
    season = pa.groupby([player_col, 'year']).agg(pa_season=('pa_id','nunique'), hits_season=('pa_is_hit','sum')).reset_index()
    season['hit_rate_season'] = season['hits_season'] / season['pa_season'].replace(0, np.nan)
    return season


def main():
    root = Path(__file__).resolve().parents[2]
    data_dir = root / 'baseball_api_mvp' / 'data'
    out_path = data_dir / 'features_statcast.parquet'

    print('Loading statcast monthly files...')
    df = load_all_statcast(data_dir)
    print(f'Rows loaded: {len(df)}')

    # Prepare PA-level and daily for batters
    print('Building plate-appearance level table...')
    pa = make_pa_level(df)

    print('Computing daily batter stats...')
    pa_batter = pa.copy()
    daily_batter = daily_player_stats(pa_batter, 'batter')

    print('Computing 7/30-day rolling features for batters...')
    batter_roll = compute_rolling(daily_batter, 'batter')
    batter_season = season_aggregates(pa_batter, 'batter')

    # Merge season into rolling
    batter_roll['year'] = batter_roll['date'].dt.year
    batter_features = batter_roll.merge(batter_season, left_on=['batter','year'], right_on=['batter','year'], how='left')
    batter_features['role'] = 'batter'

    # For pitchers: aggregate by pitcher using the same PA concept but assign player_col
    if 'pitcher' in df.columns:
        print('Computing daily pitcher stats...')
        # we need pa-level by pitcher: use original df to aggregate by pitcher at PA level
        pa_pitcher = (
            df.groupby(['pitcher','game_date','game_pk','at_bat_number']).agg(pa_is_hit=('is_hit','max'), pa_avg_velo=('velo','mean')).reset_index()
        )
        pa_pitcher['pa_id'] = pa_pitcher['game_pk'].astype(str) + '_' + pa_pitcher['at_bat_number'].astype(str)
        pa_pitcher['pa_is_k'] = 0
        pa_pitcher['pa_is_bb'] = 0

        daily_pitcher = daily_player_stats(pa_pitcher.rename(columns={'pitcher':'pitcher','pa_is_hit':'pa_is_hit','pa_avg_velo':'pa_avg_velo'}), 'pitcher')
        pitcher_roll = compute_rolling(daily_pitcher, 'pitcher')
        pitcher_season = season_aggregates(pa_pitcher.rename(columns={'pitcher':'pitcher'}), 'pitcher')
        pitcher_roll['year'] = pitcher_roll['date'].dt.year
        pitcher_features = pitcher_roll.merge(pitcher_season, left_on=['pitcher','year'], right_on=['pitcher','year'], how='left')
        pitcher_features['role'] = 'pitcher'
    else:
        pitcher_features = pd.DataFrame()

    # unify column names and save
    print('Combining and writing features...')
    # standardize column names for merge
    if not pitcher_features.empty:
        pitcher_features = pitcher_features.rename(columns={'pitcher':'player'})
    batter_features = batter_features.rename(columns={'batter':'player'})

    cols_common = ['player','date','role','pa_7d','hits_7d','hit_rate_7d','pa_30d','hits_30d','hit_rate_30d','avg_velo_7d','avg_velo_30d','pa_season','hits_season','hit_rate_season']
    # ensure columns exist
    for dfc in [batter_features, pitcher_features]:
        for c in cols_common:
            if c not in dfc.columns:
                dfc[c] = np.nan

    combined = pd.concat([batter_features[cols_common], pitcher_features[cols_common]], ignore_index=True, sort=False)
    combined.to_parquet(out_path, index=False)
    print(f'Wrote features to {out_path} (rows: {len(combined)})')


if __name__ == '__main__':
    main()
