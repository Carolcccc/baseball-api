"""Split `features_statcast_smoothed.parquet` into per-role files for training.

Writes:
 - data/features_batter_smoothed.parquet
 - data/features_pitcher_smoothed.parquet

If role column is missing, the script will attempt to guess by presence of 'role'
values or will exit.
"""
from pathlib import Path
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[2]
    data_dir = root / 'baseball_api_mvp' / 'data'
    smoothed = data_dir / 'features_statcast_smoothed.parquet'
    if not smoothed.exists():
        print('Smoothed features file not found:', smoothed)
        return

    df = pd.read_parquet(smoothed)
    if 'role' not in df.columns:
        print('No role column found in smoothed features; cannot split into batter/pitcher')
        return

    batter = df[df['role'].astype(str).str.lower() == 'batter'].copy()
    pitcher = df[df['role'].astype(str).str.lower() == 'pitcher'].copy()

    if not batter.empty:
        out_b = data_dir / 'features_batter_smoothed.parquet'
        batter.to_parquet(out_b, index=False)
        print('Wrote', out_b, 'rows=', len(batter))
    else:
        print('No batter rows found')

    if not pitcher.empty:
        out_p = data_dir / 'features_pitcher_smoothed.parquet'
        pitcher.to_parquet(out_p, index=False)
        print('Wrote', out_p, 'rows=', len(pitcher))
    else:
        print('No pitcher rows found')


if __name__ == '__main__':
    main()
