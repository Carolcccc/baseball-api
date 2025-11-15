"""Quick data audit for features_statcast.parquet

Produces a human-readable summary: rows, columns, missingness, date range,
unique players and sample size distribution. Designed to be safe if some
columns are missing.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def human_pct(x, total):
    return f"{x} / {total} ({100.0 * x / total:.2f}%)"


def main():
    root = Path(__file__).resolve().parents[2]
    data_file = root / 'baseball_api_mvp' / 'data' / 'features_statcast.parquet'
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return

    print(f"Reading {data_file} ...")
    df = pd.read_parquet(data_file)
    n = len(df)
    print(f"Rows: {n}")
    print(f"Columns: {len(df.columns)} -> {list(df.columns)[:20]}{('...' if len(df.columns)>20 else '')}")

    # Basic dtypes
    print('\nColumn types:')
    print(df.dtypes.value_counts())

    # Missingness
    print('\nTop 20 columns by missing %:')
    miss = df.isna().sum().sort_values(ascending=False)
    miss_pct = (miss / max(n, 1) * 100).round(3)
    for col, m, p in zip(miss.index[:20], miss.values[:20], miss_pct.values[:20]):
        print(f"  {col}: {m} missing ({p}%)")

    # Date range if any obvious date columns
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'day' in c.lower()]
    if date_cols:
        for c in date_cols[:3]:
            try:
                s = pd.to_datetime(df[c], errors='coerce')
                print(f"\nDate column `{c}` range: {s.min()} -> {s.max()}")
            except Exception:
                pass

    # Player columns common names
    for role in ['batter', 'pitcher', 'player', 'batter_id', 'pitcher_id']:
        if any(role == c or role in c for c in df.columns):
            break

    # Try common player id columns
    player_cols = [c for c in df.columns if c.lower() in ('batter', 'batter_id', 'batter_id_raw', 'batter_player')]
    if player_cols:
        col = player_cols[0]
        uniq = df[col].nunique(dropna=True)
        print(f"\nFound batter column `{col}`: unique={uniq}")
        counts = df[col].value_counts()
        print("  Top 5 batter sample sizes:")
        print(counts.head(5).to_string())

    # Pitcher
    pitcher_cols = [c for c in df.columns if c.lower() in ('pitcher', 'pitcher_id')]
    if pitcher_cols:
        col = pitcher_cols[0]
        uniq = df[col].nunique(dropna=True)
        print(f"\nFound pitcher column `{col}`: unique={uniq}")
        counts = df[col].value_counts()
        print("  Top 5 pitcher sample sizes:")
        print(counts.head(5).to_string())

    # Distribution of sample sizes (players with low counts)
    # Try to infer a player column; fallback: 'player'
    inferred = None
    for candidate in ['batter', 'batter_id', 'player', 'pitcher']:
        if candidate in df.columns:
            inferred = candidate
            break
    if inferred:
        vc = df[inferred].value_counts()
        print(f"\nSample-size distribution for `{inferred}`: total players={vc.size}")
        print(vc.describe())
        small = (vc <= 30).sum()
        print(f"Players with <=30 rows: {small} ({100.0*small/vc.size:.1f}%)")

    # Monthly coverage if date present
    date_col = None
    for c in date_cols:
        try:
            pd.to_datetime(df[c], errors='raise')
            date_col = c
            break
        except Exception:
            continue
    if date_col is not None:
        s = pd.to_datetime(df[date_col], errors='coerce')
        by_month = s.dt.to_period('M').value_counts().sort_index()
        print(f"\nMonthly coverage (first/last 6):\n  {by_month.head(6).to_string()}\n  ...\n  {by_month.tail(6).to_string()}")

    print('\nQuick suggestions:')
    print(' - If many players have very few rows, consider smoothing/shrinkage.')
    print(' - If key columns have high missingness, find upstream ETL gaps or backfill from raw statcast files.')
    print(' - Enrich with roster metadata (handedness, team, age) and park factors.')


if __name__ == '__main__':
    main()
