"""
Integration tests — ML Models
Trains on a tiny dataset and checks prediction shape / types.
Run: pytest tests/ -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train import generate_demo_dataset
from models.ml_models import (
    train_models, load_models, models_exist,
    predict_single, predict_batch, get_top_features,
    LABEL_MAP,
)

# ── Fixture: train a tiny model once per test session ─────────────────

@pytest.fixture(scope="session")
def trained_models(tmp_path_factory):
    """Train on a small demo set and return loaded models."""
    import models.ml_models as mm

    # Redirect model paths to a temp directory
    tmp = str(tmp_path_factory.mktemp("models"))
    mm.TFIDF_PATH = os.path.join(tmp, "tfidf_vectorizer.pkl")
    mm.LR_PATH    = os.path.join(tmp, "logistic_regression.pkl")
    mm.SVM_PATH   = os.path.join(tmp, "svm_model.pkl")
    mm.XGB_PATH   = os.path.join(tmp, "xgboost_model.pkl")

    df      = generate_demo_dataset(n_samples=300)
    results = train_models(df)
    m       = load_models()
    return m, results


# ── Training result checks ────────────────────────────────────────────

def test_train_returns_three_models(trained_models):
    _, results = trained_models
    assert set(results.keys()) == {"logistic_regression", "svm", "xgboost"}

def test_all_models_have_accuracy(trained_models):
    _, results = trained_models
    for name, metrics in results.items():
        assert "accuracy" in metrics, f"{name} missing accuracy"
        assert 0.0 <= metrics["accuracy"] <= 1.0

def test_all_models_have_report(trained_models):
    _, results = trained_models
    for name, metrics in results.items():
        assert "report" in metrics, f"{name} missing report"


# ── Prediction structure ──────────────────────────────────────────────

HIGH_RISK_TEXT = "I can't take this anymore, I want to end my life."
NORMAL_TEXT    = "Had a great day with friends, feeling really grateful."

def test_predict_single_returns_required_keys(trained_models):
    models, _ = trained_models
    result = predict_single(HIGH_RISK_TEXT, models)
    required = ("label", "risk_score", "alert", "color",
                 "xgboost", "svm", "logistic_regression")
    assert all(k in result for k in required)

def test_predict_single_label_is_valid(trained_models):
    models, _ = trained_models
    result = predict_single(NORMAL_TEXT, models)
    assert result["label"] in LABEL_MAP.values()

def test_predict_single_risk_score_range(trained_models):
    models, _ = trained_models
    for text in [HIGH_RISK_TEXT, NORMAL_TEXT]:
        result = predict_single(text, models)
        assert 0.0 <= result["risk_score"] <= 1.0

def test_predict_single_alert_is_bool(trained_models):
    models, _ = trained_models
    result = predict_single(HIGH_RISK_TEXT, models)
    assert isinstance(result["alert"], bool)

def test_predict_single_probabilities_sum_to_one(trained_models):
    models, _ = trained_models
    result = predict_single(NORMAL_TEXT, models)
    for model_key in ("xgboost", "svm", "logistic_regression"):
        probs = list(result[model_key]["probabilities"].values())
        assert abs(sum(probs) - 1.0) < 0.01, f"{model_key} probs don't sum to 1"

def test_predict_batch_returns_list(trained_models):
    models, _ = trained_models
    texts  = [HIGH_RISK_TEXT, NORMAL_TEXT, "Feeling anxious about everything."]
    result = predict_batch(texts, models)
    assert isinstance(result, list)
    assert len(result) == 3

def test_predict_empty_string(trained_models):
    models, _ = trained_models
    result = predict_single("", models)
    assert result["label"] in LABEL_MAP.values()


# ── Top features ──────────────────────────────────────────────────────

def test_get_top_features_returns_all_classes(trained_models):
    models, _ = trained_models
    feats = get_top_features(models, n=10)
    assert set(feats.keys()) == set(LABEL_MAP.values())

def test_get_top_features_correct_count(trained_models):
    models, _ = trained_models
    feats = get_top_features(models, n=10)
    for cls, kw_list in feats.items():
        assert len(kw_list) <= 10
