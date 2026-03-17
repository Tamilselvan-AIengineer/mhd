"""
Data Loader Utilities
Handles loading, validating, and converting datasets between CSV and JSON.
Supports the Suicide Detection Dataset and any compatible CSV.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Column detection ──────────────────────────────────────────────────

_TEXT_CANDIDATES  = ("text", "post", "tweet", "content", "message", "body")
_LABEL_CANDIDATES = ("label", "class", "target", "suicide", "category", "sentiment")


def detect_columns(df: pd.DataFrame) -> tuple:
    """
    Auto-detect text and label columns from a DataFrame.
    Returns (text_col, label_col). Raises ValueError if not found.
    """
    text_col  = None
    label_col = None

    for c in df.columns:
        cl = c.lower().strip()
        if text_col  is None and any(k in cl for k in _TEXT_CANDIDATES):
            text_col  = c
        if label_col is None and any(k in cl for k in _LABEL_CANDIDATES):
            label_col = c

    if text_col is None:
        raise ValueError(
            f"Cannot find a text column. "
            f"Rename a column to 'text'. Found: {list(df.columns)}"
        )
    if label_col is None:
        raise ValueError(
            f"Cannot find a label column. "
            f"Rename a column to 'label'. Found: {list(df.columns)}"
        )

    return text_col, label_col


# ── CSV loading ───────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV, auto-detect columns, rename to 'text'/'label',
    drop nulls, and return a clean DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    print(f"[loader] Loaded {len(df)} rows from {path}")

    text_col, label_col = detect_columns(df)

    df = (
        df[[text_col, label_col]]
        .rename(columns={text_col: "text", label_col: "label"})
        .dropna(subset=["text", "label"])
        .reset_index(drop=True)
    )

    # Normalise labels to lowercase strings
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    print(f"[loader] After cleaning: {len(df)} rows")
    print(f"[loader] Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


# ── JSON dataset ──────────────────────────────────────────────────────

def csv_to_json(csv_path: str, json_path: str = None) -> str:
    """
    Convert a CSV dataset to JSON format.
    Returns the path of the saved JSON file.
    """
    df = load_csv(csv_path)

    if json_path is None:
        base     = os.path.splitext(os.path.basename(csv_path))[0]
        json_path = os.path.join(DATA_DIR, f"{base}.json")

    records = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[loader] Saved {len(records)} records to {json_path}")
    return json_path


def load_json_dataset(json_path: str) -> pd.DataFrame:
    """Load a JSON dataset file into a DataFrame."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON dataset not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    print(f"[loader] Loaded {len(df)} records from {json_path}")
    return df


# ── Validation ────────────────────────────────────────────────────────

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Run basic validation checks on a training DataFrame.
    Returns a dict of checks with True/False results.
    """
    checks = {}

    checks["has_text_column"]  = "text"  in df.columns
    checks["has_label_column"] = "label" in df.columns
    checks["no_null_text"]     = df["text"].isnull().sum()  == 0 if "text"  in df.columns else False
    checks["no_null_labels"]   = df["label"].isnull().sum() == 0 if "label" in df.columns else False
    checks["min_rows_met"]     = len(df) >= 100
    checks["row_count"]        = len(df)

    if "label" in df.columns:
        vc = df["label"].value_counts().to_dict()
        checks["label_distribution"] = vc
        checks["min_classes"]        = len(vc) >= 2

    if "text" in df.columns:
        avg_len = df["text"].str.len().mean()
        checks["avg_text_length"] = round(avg_len, 1)

    return checks


# ── Sample builder ────────────────────────────────────────────────────

def build_sample_json(output_path: str = None) -> str:
    """
    Write a small sample JSON dataset to disk.
    Useful for testing without the real Kaggle dataset.
    """
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "sample_dataset.json")

    samples = [
        # High risk
        {"text": "I can't take this anymore. I want to end it all.",            "label": "suicide"},
        {"text": "Nobody would miss me if I disappeared forever.",              "label": "suicide"},
        {"text": "I have been planning to kill myself. No one knows.",          "label": "suicide"},
        {"text": "Life has no meaning. I feel completely hopeless.",            "label": "suicide"},
        {"text": "This is my final goodbye. I have made my decision.",          "label": "suicide"},
        {"text": "The pain is unbearable. I just want it to stop forever.",     "label": "suicide"},
        {"text": "I am a burden to everyone. They'd be better without me.",     "label": "suicide"},
        {"text": "I feel trapped with absolutely no way out of this.",          "label": "suicide"},
        {"text": "Thinking about suicide constantly every single day now.",     "label": "suicide"},
        {"text": "I've written goodbye notes. I don't want to be here.",        "label": "suicide"},
        # Moderate risk
        {"text": "I have been really depressed and can't get out of bed.",      "label": "moderate"},
        {"text": "Anxiety is consuming my life. Panic attacks every day.",      "label": "moderate"},
        {"text": "Feeling so alone. Nobody understands what I'm going through.","label": "moderate"},
        {"text": "I've been crying every day for weeks. Exhausted and broken.", "label": "moderate"},
        {"text": "Can't sleep. Dark thoughts won't stop. Overwhelmed always.",  "label": "moderate"},
        {"text": "Everything is overwhelming. I feel so worthless as a person.","label": "moderate"},
        {"text": "Struggling with depression. Hard to find any motivation.",    "label": "moderate"},
        {"text": "Social anxiety is so bad I can barely leave my house.",       "label": "moderate"},
        {"text": "Isolated from everyone. Deep loneliness every single day.",   "label": "moderate"},
        {"text": "Feel like I am falling apart and cannot hold it together.",   "label": "moderate"},
        # Normal
        {"text": "Had a great day at work! Feeling accomplished and happy.",    "label": "non-suicide"},
        {"text": "Talked to my therapist. Feeling more grounded now.",          "label": "non-suicide"},
        {"text": "Rough week but grateful for my supportive friends.",          "label": "non-suicide"},
        {"text": "Went for a run and it really helped with my stress.",         "label": "non-suicide"},
        {"text": "Taking it one day at a time and making progress.",            "label": "non-suicide"},
        {"text": "Not feeling 100% but tomorrow is a new day.",                 "label": "non-suicide"},
        {"text": "Life is good. Counting my blessings and staying positive.",   "label": "non-suicide"},
        {"text": "Celebrated a small win today. Proud of how far I've come.",   "label": "non-suicide"},
        {"text": "Morning meditation really helped set a positive tone today.", "label": "non-suicide"},
        {"text": "Feeling a bit down but reaching out to friends helped a lot.","label": "non-suicide"},
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"[loader] Sample dataset saved: {output_path} ({len(samples)} records)")
    return output_path


# ── Results exporter ──────────────────────────────────────────────────

def export_predictions_to_json(predictions: list, output_path: str = None) -> str:
    """
    Save a list of prediction dicts to a timestamped JSON file.
    """
    if output_path is None:
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(DATA_DIR, f"predictions_{ts}.json")

    export = {
        "exported_at": datetime.now().isoformat(),
        "count":       len(predictions),
        "records":     predictions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    print(f"[loader] Exported {len(predictions)} predictions → {output_path}")
    return output_path
