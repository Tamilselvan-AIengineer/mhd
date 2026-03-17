"""
FastAPI Backend — Mental Health Detection System
Exposes REST endpoints that can be called from any frontend or integration.

Run locally:
    uvicorn backend:app --reload --port 8000

Interactive API docs:
    http://localhost:8000/docs
"""

import os
import sys
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd

from models.ml_models import (
    predict_single, predict_batch,
    load_models, models_exist, get_top_features,
)
from utils.sentiment import get_full_sentiment, detect_emotions
from utils.preprocessor import extract_risk_keywords
from utils.storage import (
    save_analysis, load_history,
    save_batch,    load_batch,
    clear_all,     get_storage_stats,
)

# ── App setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Mental Health Detection API",
    description=(
        "ML-powered mental health risk detection from social media text. "
        "Provides risk classification, sentiment analysis, and emotion detection."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model loading ────────────────────────────────────────────────
_models = None

def get_models():
    global _models
    if _models is None:
        if not models_exist():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Models not trained yet. "
                    "Run: python train.py  before starting the API."
                ),
            )
        _models = load_models()
    return _models


# ── Schemas ───────────────────────────────────────────────────────────

class TextInput(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I have been feeling really hopeless and can't go on anymore."
            }
        }


class BatchInput(BaseModel):
    texts: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "I feel hopeless and want to end it all.",
                    "Feeling anxious but reaching out for help.",
                    "Had a great day with family today!",
                ]
            }
        }


# ── Health ────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Mental Health Detection API is running",
        "status":  "healthy",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":           "healthy",
        "models_loaded":    _models is not None,
        "models_available": models_exist(),
        "storage":          get_storage_stats(),
    }


# ── Single prediction ─────────────────────────────────────────────────

@app.post("/predict", tags=["Prediction"])
def predict(payload: TextInput):
    """
    Analyse a single text for mental health risk.

    Returns:
    - **label** — Normal / Moderate Risk / High Risk
    - **risk_score** — 0–1 probability
    - **alert** — True when risk_score ≥ 0.70
    - **sentiment** — VADER + TextBlob scores
    - **emotions** — list of detected emotions
    - **keywords** — matched risk / protective keywords
    - **model_predictions** — per-model probabilities
    """
    models = get_models()
    text   = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty.")

    pred      = predict_single(text, models)
    sentiment = get_full_sentiment(text)
    emotions  = detect_emotions(text)
    keywords  = extract_risk_keywords(text)

    result = {
        "text":              text,
        "label":             pred["label"],
        "risk_score":        pred["risk_score"],
        "alert":             pred["alert"],
        "sentiment":         sentiment,
        "emotions":          emotions,
        "keywords":          keywords,
        "model_predictions": {
            "xgboost":             pred["xgboost"],
            "svm":                 pred["svm"],
            "logistic_regression": pred["logistic_regression"],
        },
    }

    # Persist to JSON history
    save_analysis({
        "text":       text[:120],
        "label":      pred["label"],
        "risk_score": pred["risk_score"],
        "compound":   sentiment["compound"],
        "emotions":   emotions,
        "source":     "api",
    })

    return result


# ── Batch prediction ──────────────────────────────────────────────────

@app.post("/predict/batch", tags=["Prediction"])
def predict_batch_endpoint(payload: BatchInput):
    """
    Analyse a list of texts and return summary + per-text results.
    """
    models = get_models()
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty.")

    results    = []
    risk_counts = {"Normal": 0, "Moderate Risk": 0, "High Risk": 0}

    for text in payload.texts:
        pred      = predict_single(text, models)
        sentiment = get_full_sentiment(text)
        emotions  = detect_emotions(text)
        risk_counts[pred["label"]] = risk_counts.get(pred["label"], 0) + 1
        results.append({
            "text_preview":   text[:100] + "…" if len(text) > 100 else text,
            "label":          pred["label"],
            "risk_score":     pred["risk_score"],
            "alert":          pred["alert"],
            "sentiment_label":sentiment["label"],
            "compound":       sentiment["compound"],
            "emotions":       emotions,
        })

    save_batch(results)

    return {
        "total":             len(results),
        "risk_distribution": risk_counts,
        "alert_count":       sum(1 for r in results if r["alert"]),
        "avg_risk_score":    round(
            sum(r["risk_score"] for r in results) / len(results), 4
        ),
        "results": results,
    }


# ── CSV upload ────────────────────────────────────────────────────────

@app.post("/predict/csv", tags=["Prediction"])
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and run batch prediction on all rows.
    The CSV must contain a column named 'text', 'post', 'tweet', or 'content'.
    """
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}")

    text_col = None
    for candidate in ("text", "post", "tweet", "content", "message"):
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        raise HTTPException(
            status_code=400,
            detail=f"No text column found. Columns detected: {list(df.columns)}",
        )

    models = get_models()
    texts  = df[text_col].fillna("").tolist()[:500]  # cap at 500 rows
    results, risk_counts = [], {"Normal": 0, "Moderate Risk": 0, "High Risk": 0}

    for text in texts:
        pred = predict_single(text, models)
        sent = get_full_sentiment(text)
        emo  = detect_emotions(text)
        risk_counts[pred["label"]] = risk_counts.get(pred["label"], 0) + 1
        results.append({
            "text_preview":   text[:100] + "…" if len(text) > 100 else text,
            "label":          pred["label"],
            "risk_score":     pred["risk_score"],
            "alert":          pred["alert"],
            "sentiment_label":sent["label"],
            "compound":       sent["compound"],
            "emotions":       emo,
        })

    save_batch(results)

    return {
        "filename":          file.filename,
        "total_rows":        len(results),
        "risk_distribution": risk_counts,
        "alert_count":       sum(1 for r in results if r["alert"]),
        "avg_risk_score":    round(
            sum(r["risk_score"] for r in results) / len(results), 4
        ),
        "results": results,
    }


# ── Storage endpoints ─────────────────────────────────────────────────

@app.get("/storage/history", tags=["Storage"])
def get_history():
    """Return all saved single-analysis records from JSON."""
    history = load_history()
    return {"count": len(history), "records": history}


@app.get("/storage/batch", tags=["Storage"])
def get_batch():
    """Return the most recent batch analysis results from JSON."""
    batch = load_batch()
    return {"count": len(batch), "records": batch}


@app.delete("/storage/clear", tags=["Storage"])
def clear_storage():
    """Delete all stored JSON history and batch results."""
    clear_all()
    return {"message": "All storage cleared."}


# ── Model info ────────────────────────────────────────────────────────

@app.get("/models/info", tags=["Model"])
def model_info():
    return {
        "models": [
            {"name": "XGBoost",             "role": "Primary risk score prediction"},
            {"name": "SVM (LinearSVC)",      "role": "Emotion-weighted classification"},
            {"name": "Logistic Regression",  "role": "Baseline / interpretability"},
        ],
        "feature_extraction": "TF-IDF (unigrams + bigrams, max 10 000 features)",
        "sentiment_tools":    ["VADER", "TextBlob"],
        "risk_classes":       ["Normal", "Moderate Risk", "High Risk"],
        "alert_threshold":    0.70,
    }


@app.get("/models/features", tags=["Model"])
def top_features(n: int = 20):
    """Return top TF-IDF feature names per class using LR coefficients."""
    models = get_models()
    return get_top_features(models, n=n)


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
