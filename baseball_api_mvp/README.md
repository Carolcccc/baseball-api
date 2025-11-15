Baseball MVP API (scaffold)

This is a minimal scaffold for an interactive baseball prediction API (MVP).

What's included
- FastAPI app with a /predict/matchup endpoint
- A small mock predictor (no heavy ML deps)
- Pydantic request/response models
- A pytest test that exercises the endpoint

Quick start (macOS / zsh)

1) create and activate a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) install dependencies and run tests:

```bash
pip install -r baseball_api_mvp/requirements.txt
pytest -q
```

3) run the API locally:

```bash
uvicorn baseball_api_mvp.app.main:app --reload --port 8000
```

Notes
- This is a starting point: replace the mock predictor with a trained model and
  add DB / cache integration as next steps.
