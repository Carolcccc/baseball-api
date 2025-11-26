# Baseball Prediction API

An interactive baseball matchup prediction API built with FastAPI that provides real-time probability predictions for batter vs. pitcher matchups. Features a modern web UI with interactive field visualization, probability bars, and strategic insights based on historical performance data.

## Overview

This project implements a complete end-to-end machine learning pipeline for baseball prediction, including data extraction, feature engineering, model training, and an interactive API service with a polished web interface.

## Features

- **FastAPI REST API** with `/predict/matchup` endpoint for real-time predictions
- **Interactive Web UI** with:
  - Responsive dark-themed design
  - SVG field visualization with clickable base runners
  - Animated probability bars (hit, strikeout, walk probabilities)
  - Strategic insights and recommendations
- **Machine Learning Pipeline**:
  - XGBoost baseline model with rolling feature aggregates
  - Empirical Bayes smoothing for reliable small-sample estimates
  - Training script with memory-efficient sampling
- **Data Processing**:
  - ETL pipeline for MLB Statcast data (2024-2025 seasons)
  - Rolling features (7-day, 30-day, season aggregates)
  - Per-player and matchup-specific statistics
- **Testing**: Pytest test suite for API validation
- **Flexible Player IDs**: Accepts both string and integer player identifiers

## What's Included

- FastAPI application with startup model preloading
- Pydantic request/response schemas with validation
- Mock predictor fallback (no heavy ML dependencies required for demo)
- Interactive field visualization with runner animations
- Model training scripts with smoothing and feature engineering
- Comprehensive test coverage

## Project Structure

\`\`\`
baseball_api_mvp/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── baseball_api_mvp/           # Main application package
│   ├── app/                    # FastAPI application
│   │   ├── main.py            # API endpoints and startup
│   │   ├── model.py           # Model wrapper and predictor
│   │   └── schemas.py         # Pydantic models
│   ├── ui/                     # Web interface
│   │   ├── index.html         # Frontend UI
│   │   └── app.js             # JavaScript interactions
│   ├── data/                   # Processed datasets (parquet)
│   └── models/                 # Trained model artifacts
├── train/                      # Training scripts
│   └── train_baseline.py      # XGBoost baseline trainer
└── tests/                      # Test suite
    └── test_api.py            # API integration tests
\`\`\`

## Quick Start (macOS / zsh)

### 1) Create and activate a virtual environment:

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
\`\`\`

### 2) Install dependencies and run tests:

\`\`\`bash
pip install -r baseball_api_mvp/requirements.txt
pytest -q
\`\`\`

### 3) Run the API locally:

\`\`\`bash
uvicorn baseball_api_mvp.app.main:app --reload --port 8000
\`\`\`

### 4) Access the interactive UI:

Open your browser and navigate to:
- **Web UI**: http://127.0.0.1:8000/app
- **API Health**: http://127.0.0.1:8000/
- **API Docs**: http://127.0.0.1:8000/docs

## API Usage

### Predict Matchup Endpoint

**POST** \`/predict/matchup\`

Request body:
\`\`\`json
{
  "game_id": "G20251103A",
  "inning": 5,
  "outs": 1,
  "bases": [1, 0, 0],
  "batter_id": "444482",
  "pitcher_id": "445926"
}
\`\`\`

Response:
\`\`\`json
{
  "hit_prob": 0.285,
  "k_prob": 0.187,
  "walk_prob": 0.094,
  "explanation": ["Model-based prediction", "Batter season avg: .289", "Pitcher allows .245"],
  "batter_strategy": {
    "advice": "Look for pitches low and away"
  },
  "pitcher_habits": {
    "primary_pitch": "fastball",
    "tendency": "Works inside on lefties"
  }
}
\`\`\`

## Technical Stack

- **Backend**: FastAPI, Pydantic, Uvicorn
- **ML**: XGBoost, scikit-learn, pandas, pyarrow
- **Frontend**: Vanilla JavaScript, HTML5, CSS3 (SVG animations)
- **Testing**: pytest, httpx
- **Data**: MLB Statcast (via pybaseball)

## Model Training

Train a new baseline model:
\`\`\`bash
python baseball_api_mvp/train/train_baseline.py
\`\`\`

The training script:
- Loads smoothed per-role aggregate features
- Samples data for memory efficiency
- Trains an XGBoost classifier with ROC-AUC optimization
- Saves model artifact to \`models/baseline_xgb.joblib\`

## Future Enhancements

- [ ] Add runner advancement animations on prediction
- [ ] Implement accessibility features (ARIA labels, keyboard navigation)
- [ ] Add probability visualization chart (donut/pie)
- [ ] Retrain full model on high-memory instance (16-32GB RAM)
- [ ] Migrate to FastAPI lifespan events
- [ ] Fix Pydantic deprecation warnings

## Notes

- This project uses a **mock predictor** by default for demo purposes
- Replace with trained model by ensuring \`models/baseline_xgb.joblib\` exists
- Model preloading happens at app startup to avoid first-request latency
- Large data files (>100MB) are excluded via \`.gitignore\` due to GitHub limits
