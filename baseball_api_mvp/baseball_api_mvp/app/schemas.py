from pydantic import BaseModel
from typing import List, Optional, Dict, Union


class Pitch(BaseModel):
    type: str
    velo: Optional[float] = None
    spin: Optional[int] = None
    result: Optional[str] = None


class PredictRequest(BaseModel):
    game_id: str
    inning: int
    outs: int
    bases: List[int]
    # Accept either numeric IDs or string IDs from callers.
    batter_id: Union[int, str]
    pitcher_id: Union[int, str]
    recent_pitches: Optional[List[Pitch]] = None


class PredictResponse(BaseModel):
    hit_prob: float
    k_prob: float
    walk_prob: float
    pitcher_habits: Dict[str, Dict[str, float]]
    batter_strategy: Dict[str, str]
    explanation: List[str]
