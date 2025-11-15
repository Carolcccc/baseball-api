import os
import sys
from fastapi.testclient import TestClient

# Ensure the inner package path is on sys.path so imports work when pytest
# is run from the repository root.
ROOT = os.path.dirname(__file__)
INNER_PKG = os.path.normpath(os.path.join(ROOT, "..", "baseball_api_mvp"))
if INNER_PKG not in sys.path:
    sys.path.insert(0, INNER_PKG)

from app.main import app


client = TestClient(app)


def test_predict_matchup_basic():
    payload = {
        "game_id": "G20251103A",
        "inning": 5,
        "outs": 1,
        "bases": [1, 0, 0],
        "batter_id": "B123",
        "pitcher_id": "P987",
    }

    r = client.post("/predict/matchup", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "hit_prob" in data
    assert 0.0 <= data["hit_prob"] <= 1.0
