"""
ML Models Module
Trains Logistic Regression, SVM, and XGBoost on TF-IDF features.
Handles model persistence via joblib and prediction for single/batch text.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

from utils.preprocessor import preprocess, batch_preprocess

# ── Paths ─────────────────────────────────────────────────────────────
MODEL_DIR    = os.path.join(os.path.dirname(__file__))
TFIDF_PATH   = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LR_PATH      = os.path.join(MODEL_DIR, "logistic_regression.pkl")
SVM_PATH     = os.path.join(MODEL_DIR, "svm_model.pkl")
XGB_PATH     = os.path.join(MODEL_DIR, "xgboost_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Label mapping ─────────────────────────────────────────────────────
LABEL_MAP = {0: "Normal", 1: "Moderate Risk", 2: "High Risk"}
RISK_COLORS = {
    "Normal":        "#27ae60",
    "Moderate Risk": "#e07b1a",
    "High Risk":     "#c0392b",
}


def _map_label(label_str: str) -> int:
    """Map raw dataset label string to 0/1/2 integer class."""
    l = str(label_str).lower().strip()
    if l in ("suicide", "1", "high risk", "high_risk", "crisis"):
        return 2
    elif l in ("moderate", "moderate risk", "moderate_risk", "2"):
        return 1
    else:
        return 0


def build_tfidf(max_features: int = 10000, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )


# ── Training ──────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> dict:
    """
    Full training pipeline.
    Returns a dict of per-model metrics.
    """
    print("[INFO] Starting training pipeline …")

    # Labels
    y = df[label_col].apply(_map_label).values

    # Preprocess
    print("[INFO] Preprocessing texts …")
    texts     = df[text_col].fillna("").tolist()
    processed = batch_preprocess(texts)

    # TF-IDF
    print("[INFO] Fitting TF-IDF …")
    tfidf = build_tfidf()
    X     = tfidf.fit_transform(processed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # ── Logistic Regression ──────────────────────────────────────────
    print("[INFO] Training Logistic Regression …")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, multi_class="multinomial")
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    results["logistic_regression"] = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "report":   classification_report(
            y_test, preds,
            target_names=list(LABEL_MAP.values()),
            output_dict=True, zero_division=0,
        ),
    }
    joblib.dump(lr, LR_PATH)
    print(f"  LR  accuracy: {results['logistic_regression']['accuracy']:.4f}")

    # ── SVM ──────────────────────────────────────────────────────────
    print("[INFO] Training SVM …")
    svm_base = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    svm      = CalibratedClassifierCV(svm_base, cv=3)
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    results["svm"] = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "report":   classification_report(
            y_test, preds,
            target_names=list(LABEL_MAP.values()),
            output_dict=True, zero_division=0,
        ),
    }
    joblib.dump(svm, SVM_PATH)
    print(f"  SVM accuracy: {results['svm']['accuracy']:.4f}")

    # ── XGBoost ──────────────────────────────────────────────────────
    print("[INFO] Training XGBoost …")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = xgb.predict(X_test)
    results["xgboost"] = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "report":   classification_report(
            y_test, preds,
            target_names=list(LABEL_MAP.values()),
            output_dict=True, zero_division=0,
        ),
    }
    joblib.dump(xgb,   XGB_PATH)
    joblib.dump(tfidf, TFIDF_PATH)
    print(f"  XGB accuracy: {results['xgboost']['accuracy']:.4f}")
    print("[INFO] All models saved.")
    return results


# ── Load ──────────────────────────────────────────────────────────────

def load_models() -> dict:
    """Load all trained models from disk. Raises FileNotFoundError if missing."""
    missing = [p for p in [TFIDF_PATH, LR_PATH, SVM_PATH, XGB_PATH]
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Trained models not found: {missing}. Run train.py first."
        )
    return {
        "tfidf": joblib.load(TFIDF_PATH),
        "lr":    joblib.load(LR_PATH),
        "svm":   joblib.load(SVM_PATH),
        "xgb":   joblib.load(XGB_PATH),
    }


def models_exist() -> bool:
    """Return True when all four model files are present."""
    return all(os.path.exists(p) for p in [TFIDF_PATH, LR_PATH, SVM_PATH, XGB_PATH])


# ── Prediction ────────────────────────────────────────────────────────

def predict_single(text: str, models: dict) -> dict:
    """
    Full prediction for one text string.
    Returns label, risk score, alert flag, and per-model probabilities.
    """
    processed = preprocess(text)
    X = models["tfidf"].transform([processed])

    # XGBoost (primary)
    xgb_proba = models["xgb"].predict_proba(X)[0]
    xgb_class = int(np.argmax(xgb_proba))

    # Weighted risk score: full weight on High Risk + half weight on Moderate
    risk_score = float(xgb_proba[2]) + 0.5 * float(xgb_proba[1])
    risk_score = min(round(risk_score, 4), 1.0)

    # LR
    lr_proba  = models["lr"].predict_proba(X)[0]
    lr_class  = int(np.argmax(lr_proba))

    # SVM
    svm_proba = models["svm"].predict_proba(X)[0]
    svm_class = int(np.argmax(svm_proba))

    label = LABEL_MAP[xgb_class]
    alert = risk_score >= 0.70

    def proba_dict(proba_arr):
        return {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba_arr)}

    return {
        "label":      label,
        "risk_score": risk_score,
        "alert":      alert,
        "color":      RISK_COLORS[label],
        "xgboost": {
            "class": LABEL_MAP[xgb_class],
            "probabilities": proba_dict(xgb_proba),
        },
        "logistic_regression": {
            "class": LABEL_MAP[lr_class],
            "probabilities": proba_dict(lr_proba),
        },
        "svm": {
            "class": LABEL_MAP[svm_class],
            "probabilities": proba_dict(svm_proba),
        },
    }


def predict_batch(texts: list, models: dict) -> list:
    """Run predict_single on a list of texts and return results list."""
    return [predict_single(t, models) for t in texts]


def get_top_features(models: dict, n: int = 20) -> dict:
    """
    Return top n TF-IDF feature names per class using
    Logistic Regression coefficients.
    """
    tfidf        = models["tfidf"]
    lr           = models["lr"]
    feature_names = np.array(tfidf.get_feature_names_out())

    top_features = {}
    if hasattr(lr, "coef_"):
        for class_idx, class_name in LABEL_MAP.items():
            coef    = lr.coef_[class_idx]
            top_idx = np.argsort(coef)[-n:][::-1]
            top_features[class_name] = list(feature_names[top_idx])

    return top_features
