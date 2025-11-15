"""Simple ETL script that reads a sample play-by-play CSV and produces
per-player aggregate features (rolling and season-style aggregates).

This is intentionally small and dependency-light (uses pandas) to serve as
an ETL skeleton you can extend later.
"""
import os
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "baseball_api_mvp" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_sample(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def compute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    # Basic per-player aggregates: total PA, hits, Ks, walks, avg velo by pitcher
    # Expect sample columns: game_id, inning, batter_id, pitcher_id, pitch_type, result, velo, spin
    df = df.copy()
    # create simple indicators
    df["is_hit"] = df["result"].isin(["H", "1B", "2B", "3B", "HR"]).astype(int)
    df["is_k"] = (df["result"] == "K").astype(int)
    df["is_bb"] = (df["result"] == "BB").astype(int)

    # batter aggregates
    batter_aggs = (
        df.groupby("batter_id")
        .agg(
            pa=("result", "count"),
            hits=("is_hit", "sum"),
            ks=("is_k", "sum"),
            walks=("is_bb", "sum"),
        )
        .reset_index()
    )
    batter_aggs["hit_rate"] = batter_aggs["hits"] / batter_aggs["pa"]

    # pitcher aggregates
    pitcher_aggs = (
        df.groupby("pitcher_id")
        .agg(
            pa_faced=("result", "count"),
            hits_allowed=("is_hit", "sum"),
            ks=("is_k", "sum"),
            avg_velo=("velo", "mean"),
        )
        .reset_index()
    )
    pitcher_aggs["opp_hit_rate"] = pitcher_aggs["hits_allowed"] / pitcher_aggs["pa_faced"]

    # join example: batter vs pitcher counts
    vs = (
        df.groupby(["batter_id", "pitcher_id"]).agg(pa_vs=("result", "count")).reset_index()
    )

    # write outputs
    out_dir = DATA_DIR
    batter_aggs.to_parquet(out_dir / "batter_aggs.parquet", index=False)
    pitcher_aggs.to_parquet(out_dir / "pitcher_aggs.parquet", index=False)
    vs.to_parquet(out_dir / "vs_aggs.parquet", index=False)

    return batter_aggs, pitcher_aggs, vs


def main():
    sample_csv = DATA_DIR / "sample_plays.csv"
    if not sample_csv.exists():
        raise FileNotFoundError(f"sample csv not found at {sample_csv}. Create one at this path.")

    print("Loading sample plays...")
    df = load_sample(sample_csv)
    print(f"Rows: {len(df)}")

    print("Computing aggregates...")
    batter_aggs, pitcher_aggs, vs = compute_aggregates(df)

    print("Wrote:")
    print(f" - {DATA_DIR / 'batter_aggs.parquet'}")
    print(f" - {DATA_DIR / 'pitcher_aggs.parquet'}")
    print(f" - {DATA_DIR / 'vs_aggs.parquet'}")


if __name__ == "__main__":
    main()
