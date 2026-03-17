"""
JSON Storage Module
Handles reading and writing of analysis history and batch results to JSON files.
Used for persisting data across sessions locally and on Streamlit Cloud.
"""

import json
import os
from datetime import datetime

STORAGE_DIR   = "data"
HISTORY_FILE  = os.path.join(STORAGE_DIR, "analysis_history.json")
BATCH_FILE    = os.path.join(STORAGE_DIR, "batch_results.json")
DATASET_FILE  = os.path.join(STORAGE_DIR, "demo_dataset.json")

os.makedirs(STORAGE_DIR, exist_ok=True)


# ── Internal helpers ─────────────────────────────────────────────────

def _load(path: str) -> list:
    """Load a JSON file and return its contents as a list."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save(path: str, data):
    """Save data (list or dict) to a JSON file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[WARN] Could not save to {path}: {e}")


# ── Analysis History ─────────────────────────────────────────────────

def save_analysis(record: dict):
    """
    Append a single analysis result to the history JSON file.
    Automatically adds a timestamp.
    """
    history = _load(HISTORY_FILE)
    record["timestamp"] = datetime.now().isoformat()
    # Keep only last 500 records to avoid file bloat
    history.append(record)
    if len(history) > 500:
        history = history[-500:]
    _save(HISTORY_FILE, history)


def load_history() -> list:
    """Load all saved analysis history records."""
    return _load(HISTORY_FILE)


def delete_history_record(index: int):
    """Delete a single record from history by index."""
    history = _load(HISTORY_FILE)
    if 0 <= index < len(history):
        history.pop(index)
        _save(HISTORY_FILE, history)


# ── Batch Results ────────────────────────────────────────────────────

def save_batch(results: list):
    """Save a full batch analysis result list to JSON."""
    _save(BATCH_FILE, results)


def load_batch() -> list:
    """Load saved batch analysis results."""
    return _load(BATCH_FILE)


# ── Clear / Reset ────────────────────────────────────────────────────

def clear_all():
    """Clear all stored data (history + batch)."""
    _save(HISTORY_FILE, [])
    _save(BATCH_FILE, [])


def clear_history():
    """Clear only the analysis history."""
    _save(HISTORY_FILE, [])


def clear_batch():
    """Clear only the batch results."""
    _save(BATCH_FILE, [])


# ── Stats ────────────────────────────────────────────────────────────

def get_storage_stats() -> dict:
    """Return summary stats about stored data."""
    history = _load(HISTORY_FILE)
    batch   = _load(BATCH_FILE)
    return {
        "history_count": len(history),
        "batch_count":   len(batch),
        "history_file":  HISTORY_FILE,
        "batch_file":    BATCH_FILE,
        "history_exists": os.path.exists(HISTORY_FILE),
        "batch_exists":   os.path.exists(BATCH_FILE),
    }


# ── Export ───────────────────────────────────────────────────────────

def export_history_json() -> str:
    """Return history as a formatted JSON string for download."""
    history = _load(HISTORY_FILE)
    return json.dumps(history, indent=2, ensure_ascii=False)


def export_batch_json() -> str:
    """Return batch results as a formatted JSON string for download."""
    batch = _load(BATCH_FILE)
    return json.dumps(batch, indent=2, ensure_ascii=False)
