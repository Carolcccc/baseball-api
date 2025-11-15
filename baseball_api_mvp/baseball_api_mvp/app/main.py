from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from .schemas import PredictRequest, PredictResponse
from . import model

app = FastAPI(title="Baseball MVP API (MVP)")


@app.get("/", tags=["health"])
def health_check():
    return {"status": "ok", "service": "baseball_mvp"}


# Serve a minimal single-page UI from the `ui/` directory. This lets you open
# the API in a browser (or kiosk) and interact with the `/predict/matchup`
# endpoint without running a separate frontend server.
UI_DIR = Path(__file__).resolve().parents[1] / "ui"
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/app{rest:path}", include_in_schema=False)
def app_ui(rest: str = ""):
    # Serve index for /app and any subpath (helps with encoded/extra chars
    # or client-side routing). This prevents 404 when the browser appends
    # extra text or when users open bookmarked variants.
    index = UI_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"error": "UI not found"}


@app.post("/predict/matchup", response_model=PredictResponse)
def predict_matchup(req: PredictRequest):
    """Return mock predictions and simple insights for a batter/pitcher matchup.

    Replace the internals with a real model loader and feature preparation.
    """
    # Coerce numeric-looking IDs to ints for downstream model code that expects numeric IDs.
    try:
        batter_id = int(req.batter_id)
    except Exception:
        batter_id = req.batter_id

    try:
        pitcher_id = int(req.pitcher_id)
    except Exception:
        pitcher_id = req.pitcher_id

    res = model.DEFAULT_MODEL.predict(batter_id, pitcher_id, req.dict())
    return res



@app.on_event("startup")
def _startup_event():
    # load the trained model and feature metadata into memory at startup
    try:
        model.load_default_model()
    except Exception:
        # fallback silently to MockModel; errors will be visible in logs
        pass
