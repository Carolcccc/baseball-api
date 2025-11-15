"""Compute simple EB-style smoothing for hit rates and optionally enrich with roster metadata.

This script:
 - Reads `data/features_statcast.parquet`
 - Computes smoothed hit rates for 7-day, 30-day and season windows using a
   simple pseudo-count prior: smoothed = (hits + prior_count * p_global) / (pa + prior_count)
 - If `data/roster.csv` exists, joins roster fields (player_name, handedness, team)
 - Writes `features_statcast_smoothed.parquet` alongside the original file.

This is a pragmatic smoothing approach (fast, robust). We use tunable prior counts
that reflect the window size; these can be improved later by empirical Bayes fitting.
"""
from pathlib import Path
import pandas as pd
import numpy as np


PRIOR_COUNTS = {
    '7d': 10,
    '30d': 40,
    'season': 200,
}


def smooth_rate(hits, pa, p_global, prior_count):
    # handle zero pa: return global p
    pa = np.asarray(pa, dtype=float)
    hits = np.asarray(hits, dtype=float)
    out = np.where(pa > 0, (hits + prior_count * p_global) / (pa + prior_count), p_global)
    return out


def main():
    root = Path(__file__).resolve().parents[2]
    data_dir = root / 'baseball_api_mvp' / 'data'
    features_file = data_dir / 'features_statcast.parquet'
    if not features_file.exists():
        print(f"features file not found: {features_file}")
        return

    print(f"Loading {features_file}")
    df = pd.read_parquet(features_file)

    # windows to smooth: mapping of prefix to (pa_col, hits_col, existing_rate_col)
    windows = {
        '7d': ('pa_7d', 'hits_7d', 'hit_rate_7d'),
        '30d': ('pa_30d', 'hits_30d', 'hit_rate_30d'),
        'season': ('pa_season', 'hits_season', 'hit_rate_season'),
    }

    for w, (pa_col, hits_col, rate_col) in windows.items():
        if pa_col not in df.columns or hits_col not in df.columns:
            print(f"Skipping {w} smoothing because columns missing: {pa_col} or {hits_col}")
            continue

        total_hits = df[hits_col].sum()
        total_pa = df[pa_col].sum()
        if total_pa <= 0:
            p_global = 0.0
        else:
            p_global = float(total_hits) / float(total_pa)

        prior = PRIOR_COUNTS.get(w, 20)
        print(f"Window {w}: total_hits={int(total_hits)}, total_pa={int(total_pa)}, global_p={p_global:.4f}, prior_count={prior}")

        smoothed = smooth_rate(df[hits_col].fillna(0).values, df[pa_col].fillna(0).values, p_global, prior)
        out_col = f"hit_rate_{w}_smooth"
        df[out_col] = smoothed

    # Try to enrich with roster if available
    roster_file = data_dir / 'roster.csv'
    if roster_file.exists():
        try:
            roster = pd.read_csv(roster_file)
            # find a matching key: roster may use 'player' or 'player_id'
            candidate_keys = [k for k in ('player', 'player_id', 'id') if k in roster.columns]
            if not candidate_keys:
                print(f"Roster file found but missing id column (expected 'player' or 'player_id'). Columns: {list(roster.columns)}")
            else:
                key = candidate_keys[0]
                print(f"Joining roster on {key}")
                # normalize types
                df['player'] = df['player'].astype(str)
                roster[key] = roster[key].astype(str)
                # prefer common columns: name, handedness, team
                keep = [c for c in ('player_name', 'name', 'full_name', 'handedness', 'throws', 'team') if c in roster.columns]
                merged = df.merge(roster[[key] + keep].rename(columns={key: 'player'}), on='player', how='left')
                df = merged
        except Exception as e:
            print('Error reading or joining roster:', e)
    else:
        # create a template roster to help human-fill
        template = pd.DataFrame(columns=['player', 'player_name', 'handedness', 'team', 'position', 'mlb_debut'])
        template_file = roster_file
        if not template_file.exists():
            template.to_csv(template_file, index=False)
            print(f"Wrote roster template to {template_file} - fill it and re-run to enrich features.")

    out_file = data_dir / 'features_statcast_smoothed.parquet'
    df.to_parquet(out_file, index=False)
    print(f"Wrote smoothed/enriched features to {out_file} (rows={len(df)})")


if __name__ == '__main__':
    main()
