"""Download Statcast data using pybaseball and produce simple feature parquet files.

Usage (example):
    python get_statcast.py 2024-06-01 2024-06-07

Notes:
- This script is a small ETL example. For large date ranges, run in chunks.
"""
import sys
from pathlib import Path
import pandas as pd

def ensure_data_dir(root: Path) -> Path:
    out = root / "baseball_api_mvp" / "data"
    out.mkdir(parents=True, exist_ok=True)
    return out


def download_statcast(start_date: str, end_date: str) -> pd.DataFrame:
    # import inside function to avoid hard dependency unless this script is used
    try:
        from pybaseball import statcast
    except Exception as e:
        raise RuntimeError("pybaseball is required for Statcast download. Install pybaseball.") from e

    print(f"Downloading Statcast from {start_date} to {end_date} (may take a moment)...")
    df = statcast(start_date, end_date)
    print(f"Downloaded {len(df)} rows")
    return df


def basic_process(df: pd.DataFrame) -> pd.DataFrame:
    # keep a subset of useful cols and do light cleaning
    keep = [
        "game_date",
        "game_pk",
        "inning",
        "at_bat_number",
        "pitch_number",
        "player_name",
        "batter",
        "pitcher",
        "pitch_type",
        "events",
        "description",
        "release_speed",
        "release_spin_rate",
    ]
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()
    # normalize column names
    if "release_speed" in out.columns:
        out = out.rename(columns={"release_speed": "velo"})
    if "release_spin_rate" in out.columns:
        out = out.rename(columns={"release_spin_rate": "spin"})

    # simple result label
    if "events" in out.columns:
        out["is_hit"] = out["events"].isin(["single", "double", "triple", "home_run"]).astype(int)
        out["is_k"] = (out["events"] == "strikeout").astype(int)
        out["is_bb"] = out["events"] == "walk"

    return out


def aggregates_and_write(out_dir: Path, processed: pd.DataFrame):
    # per-batter aggregates
    batter_aggs = (
        processed.groupby("batter")
        .agg(pa=("at_bat_number", "count"), hits=("is_hit", "sum"))
        .reset_index()
    )
    batter_aggs["hit_rate"] = batter_aggs["hits"] / batter_aggs["pa"].replace(0, 1)

    pitcher_aggs = (
        processed.groupby("pitcher")
        .agg(pa_faced=("at_bat_number", "count"), hits_allowed=("is_hit", "sum"), avg_velo=("velo", "mean"))
        .reset_index()
    )
    pitcher_aggs["opp_hit_rate"] = pitcher_aggs["hits_allowed"] / pitcher_aggs["pa_faced"].replace(0, 1)

    vs = (
        processed.groupby(["batter", "pitcher"]).agg(pa_vs=("at_bat_number", "count"), hits_vs=("is_hit", "sum")).reset_index()
    )

    statcast_path = out_dir / "statcast_raw.parquet"
    processed.to_parquet(statcast_path, index=False)
    batter_aggs.to_parquet(out_dir / "batter_aggs_statcast.parquet", index=False)
    pitcher_aggs.to_parquet(out_dir / "pitcher_aggs_statcast.parquet", index=False)
    vs.to_parquet(out_dir / "vs_aggs_statcast.parquet", index=False)

    print(f"Wrote {statcast_path}")
    return statcast_path


def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = ensure_data_dir(root)

    if len(sys.argv) < 3:
        # default short range (small) to avoid huge downloads
        start_date = "2024-06-01"
        end_date = "2024-06-03"
        print("No dates provided, defaulting to small sample: 2024-06-01..2024-06-03")
    else:
        start_date = sys.argv[1]
        end_date = sys.argv[2]

    df = download_statcast(start_date, end_date)
    proc = basic_process(df)
    aggregates_and_write(out_dir, proc)


if __name__ == "__main__":
    main()
