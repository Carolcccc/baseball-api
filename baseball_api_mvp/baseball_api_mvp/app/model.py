"""Predictor module: provides a default model instance. If a trained model
is present in the models/ directory it will be loaded and used. Otherwise a
MockModel is used for demo/testing.
"""
import hashlib
import joblib
from typing import Dict, Any
from pathlib import Path
import pandas as pd


class MockModel:
    def predict(self, batter_id: str, pitcher_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # deterministic pseudo-probabilities based on hashed ids (for demo only)
        key = f"{batter_id}:{pitcher_id}"
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        base = (h % 1000) / 1000.0

        hit_prob = round(0.08 + 0.5 * base, 3)
        k_prob = round(0.1 + 0.35 * (1 - base), 3)
        walk_prob = round(0.02 + 0.1 * ((h >> 7) % 10) / 10.0, 3)

        pitcher_habits = {
            "vs_right": {"FB": 0.6, "SL": 0.28, "CH": 0.12},
            "vs_left": {"FB": 0.55, "SL": 0.25, "CH": 0.2},
        }

        batter_strategy = {"advice": "prefer low-and-away against breaking pitches"}

        explanation = [
            "This is a mock, hash-based heuristic used for demo and testing.",
            f"base={base:.3f}",
        ]

        return {
            "hit_prob": float(min(max(hit_prob, 0.0), 1.0)),
            "k_prob": float(min(max(k_prob, 0.0), 1.0)),
            "walk_prob": float(min(max(walk_prob, 0.0), 1.0)),
            "pitcher_habits": pitcher_habits,
            "batter_strategy": batter_strategy,
            "explanation": explanation,
        }


class XGBModelWrapper:
    def __init__(self, model_path: Path, data_dir: Path):
        self.model = joblib.load(model_path)
        # load feature columns
        feat_file = model_path.parent / 'feature_columns.txt'
        if feat_file.exists():
            self.feature_columns = [l.strip() for l in feat_file.read_text().splitlines() if l.strip()]
        else:
            self.feature_columns = []

        # load per-player aggregates for quick lookup
        self.batter_features = {}
        self.pitcher_features = {}
        try:
            bf = pd.read_parquet(data_dir / 'features_batter_2024_2025.parquet')
            pf = pd.read_parquet(data_dir / 'features_pitcher_2024_2025.parquet')
            # convert to dict by player
            self.batter_features = bf.set_index('player').to_dict(orient='index')
            self.pitcher_features = pf.set_index('player').to_dict(orient='index')
        except Exception:
            # If files missing, keep empty dicts
            pass

    def _make_row(self, batter_id: str, pitcher_id: str):
        # create a single-row DataFrame matching feature_columns
        data = {}
        # batter prefixes
        b = self.batter_features.get(batter_id, {})
        p = self.pitcher_features.get(pitcher_id, {})
        for col in self.feature_columns:
            if col.startswith('batter_'):
                key = col.replace('batter_', '')
                data[col] = b.get(key, 0)
            elif col.startswith('pitcher_'):
                key = col.replace('pitcher_', '')
                data[col] = p.get(key, 0)
            else:
                data[col] = 0
        return pd.DataFrame([data])

    def predict(self, batter_id: str, pitcher_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        X = self._make_row(batter_id, pitcher_id)
        prob = float(self.model.predict_proba(X[self.feature_columns])[:,1][0]) if self.feature_columns else 0.0
        return {
            'hit_prob': prob,
            'k_prob': 0.0,
            'walk_prob': 0.0,
            'pitcher_habits': {},
            'batter_strategy': {},
            'explanation': ['Model-based prediction']
        }


# By default use the lightweight mock model. Call `load_default_model()` from
# application startup to replace with a trained model if available.
DEFAULT_MODEL = MockModel()


def load_default_model():
    """Attempt to load a trained model and feature metadata from the repo's
    models/ and data/ directories. This should be called during app startup
    to avoid blocking the first request.
    """
    global DEFAULT_MODEL
    ROOT = Path(__file__).resolve().parents[2]
    MODEL_PATH = ROOT / 'baseball_api_mvp' / 'models' / 'baseline_xgb.joblib'
    DATA_DIR = ROOT / 'baseball_api_mvp' / 'data'

    if MODEL_PATH.exists():
        try:
            DEFAULT_MODEL = XGBModelWrapper(MODEL_PATH, DATA_DIR)
        except Exception:
            DEFAULT_MODEL = MockModel()
    else:
        DEFAULT_MODEL = MockModel()

