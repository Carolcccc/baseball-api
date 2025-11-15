"""Download Statcast data for a date range in monthly chunks and produce features.

Usage:
    python download_statcast_range.py 2024-01-01 2025-11-03

This will:
- download monthly Statcast data (skipping months already downloaded)
- write per-month parquet files named statcast_YYYY_MM.parquet
- create a merged features_statcast.parquet with per-player aggregates

Notes:
- This can produce large files and take time depending on range and network.
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


def ensure_data_dir(root: Path) -> Path:
    out = root / "baseball_api_mvp" / "data"
    out.mkdir(parents=True, exist_ok=True)
    return out


def month_ranges(start: str, end: str):
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    # generate month start dates
    idx = pd.date_range(start=start_ts, end=end_ts, freq='MS')
    ranges = []
    for s in idx:
        month_start = s
        # month end is either end of month or global end
        month_end = (s + pd.offsets.MonthEnd(0))
        if month_end > end_ts:
            month_end = end_ts
        ranges.append((month_start.strftime('%Y-%m-%d'), month_end.strftime('%Y-%m-%d')))
    return ranges


def download_month(start_date: str, end_date: str, out_dir: Path):
    from pybaseball import statcast

    fname = out_dir / f"statcast_{start_date[:7].replace('-', '_')}.parquet"
    if fname.exists():
        print(f"Skipping {start_date}..{end_date}, file exists: {fname.name}")
        return fname

    print(f"Downloading {start_date} .. {end_date} ...")
    df = statcast(start_date, end_date)
    if df.empty:
        print("No data for this period.")
        return None

    # light normalization
    keep = [c for c in ['game_date','game_pk','inning','at_bat_number','pitch_number','player_name','batter','pitcher','pitch_type','events','description','release_speed','release_spin_rate'] if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={'release_speed':'velo','release_spin_rate':'spin'})
    df.to_parquet(fname, index=False)
    print(f"Wrote {fname} ({len(df)} rows)")
    return fname


def merge_and_aggregate(out_dir: Path):
    files = sorted(out_dir.glob('statcast_*.parquet'))
    if not files:
        print('No monthly files to merge.')
        return

    print(f'Merging {len(files)} files...')
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            print(f'Warning: failed to read {f}, skipping')
    full = pd.concat(dfs, ignore_index=True)
    print(f'Total rows after merge: {len(full)}')

    # simple feature aggregation per batter/pitcher
    if 'at_bat_number' in full.columns:
        full['is_hit'] = full.get('events', '').isin(['single','double','triple','home_run'])
    else:
        full['is_hit'] = 0

    batter_aggs = full.groupby('batter').agg(pa=('at_bat_number','count'), hits=('is_hit','sum')).reset_index()
    batter_aggs['hit_rate'] = batter_aggs['hits'] / batter_aggs['pa'].replace(0,1)

    pitcher_aggs = full.groupby('pitcher').agg(pa_faced=('at_bat_number','count'), hits_allowed=('is_hit','sum'), avg_velo=('velo','mean')).reset_index()
    pitcher_aggs['opp_hit_rate'] = pitcher_aggs['hits_allowed'] / pitcher_aggs['pa_faced'].replace(0,1)

    vs = full.groupby(['batter','pitcher']).agg(pa_vs=('at_bat_number','count'), hits_vs=('is_hit','sum')).reset_index()

    batter_aggs.to_parquet(out_dir / 'features_batter_2024_2025.parquet', index=False)
    pitcher_aggs.to_parquet(out_dir / 'features_pitcher_2024_2025.parquet', index=False)
    vs.to_parquet(out_dir / 'features_vs_2024_2025.parquet', index=False)
    print('Wrote aggregated feature files')


def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = ensure_data_dir(root)

    if len(sys.argv) < 3:
        start_date = '2024-01-01'
        end_date = datetime.today().strftime('%Y-%m-%d')
        print(f'No dates provided, defaulting to {start_date}..{end_date}')
    else:
        start_date = sys.argv[1]
        end_date = sys.argv[2]

    ranges = month_ranges(start_date, end_date)
    for s,e in ranges:
        try:
            download_month(s,e,out_dir)
        except Exception as exc:
            print(f'Failed for {s}..{e}: {exc}')

    merge_and_aggregate(out_dir)


if __name__ == '__main__':
    main()
