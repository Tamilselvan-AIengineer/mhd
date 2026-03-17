"""
Unit tests — JSON Storage Module
Run: pytest tests/ -v
"""

import sys
import os
import json
import tempfile
import pytest

# Redirect storage to a temp directory so tests don't touch real data
_tmp = tempfile.mkdtemp()
os.environ["_TEST_STORAGE_DIR"] = _tmp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import utils.storage as storage

# Monkey-patch storage paths to temp dir
storage.STORAGE_DIR   = _tmp
storage.HISTORY_FILE  = os.path.join(_tmp, "analysis_history.json")
storage.BATCH_FILE    = os.path.join(_tmp, "batch_results.json")


# ── Helpers ───────────────────────────────────────────────────────────

def _reset():
    """Delete temp JSON files before each test."""
    for f in (storage.HISTORY_FILE, storage.BATCH_FILE):
        if os.path.exists(f):
            os.remove(f)


# ── save / load history ───────────────────────────────────────────────

def test_save_and_load_history():
    _reset()
    storage.save_analysis({"text": "test", "label": "Normal",
                            "risk_score": 0.1, "compound": 0.3, "emotions": ["Hope"]})
    history = storage.load_history()
    assert len(history) == 1
    assert history[0]["label"] == "Normal"

def test_history_adds_timestamp():
    _reset()
    storage.save_analysis({"text": "hello", "label": "Normal",
                            "risk_score": 0.05, "compound": 0.2, "emotions": []})
    history = storage.load_history()
    assert "timestamp" in history[0]

def test_multiple_saves_accumulate():
    _reset()
    for i in range(5):
        storage.save_analysis({"text": f"post {i}", "label": "Normal",
                                "risk_score": 0.1, "compound": 0.0, "emotions": []})
    assert len(storage.load_history()) == 5

def test_load_history_empty_when_no_file():
    _reset()
    assert storage.load_history() == []


# ── save / load batch ─────────────────────────────────────────────────

def test_save_and_load_batch():
    _reset()
    records = [
        {"text_preview": "post 1", "label": "High Risk",     "risk_score": 0.9},
        {"text_preview": "post 2", "label": "Moderate Risk", "risk_score": 0.5},
    ]
    storage.save_batch(records)
    batch = storage.load_batch()
    assert len(batch) == 2
    assert batch[0]["label"] == "High Risk"

def test_batch_overwrites_previous():
    _reset()
    storage.save_batch([{"text_preview": "old", "label": "Normal", "risk_score": 0.1}])
    storage.save_batch([{"text_preview": "new", "label": "High Risk", "risk_score": 0.9}])
    batch = storage.load_batch()
    assert len(batch) == 1
    assert batch[0]["text_preview"] == "new"

def test_load_batch_empty_when_no_file():
    _reset()
    assert storage.load_batch() == []


# ── clear_all ────────────────────────────────────────────────────────

def test_clear_all():
    _reset()
    storage.save_analysis({"text": "x", "label": "Normal",
                            "risk_score": 0.1, "compound": 0.0, "emotions": []})
    storage.save_batch([{"text_preview": "x", "label": "Normal", "risk_score": 0.1}])
    storage.clear_all()
    assert storage.load_history() == []
    assert storage.load_batch()   == []


# ── get_storage_stats ────────────────────────────────────────────────

def test_storage_stats_counts():
    _reset()
    for _ in range(3):
        storage.save_analysis({"text": "x", "label": "Normal",
                                "risk_score": 0.1, "compound": 0.0, "emotions": []})
    storage.save_batch([{"t": "a"}, {"t": "b"}])
    stats = storage.get_storage_stats()
    assert stats["history_count"] == 3
    assert stats["batch_count"]   == 2


# ── export ────────────────────────────────────────────────────────────

def test_export_history_json_is_valid():
    _reset()
    storage.save_analysis({"text": "hello", "label": "Normal",
                            "risk_score": 0.1, "compound": 0.2, "emotions": []})
    exported = storage.export_history_json()
    parsed   = json.loads(exported)
    assert isinstance(parsed, list)
    assert len(parsed) == 1

def test_export_batch_json_is_valid():
    _reset()
    storage.save_batch([{"text_preview": "test", "label": "Normal", "risk_score": 0.1}])
    exported = storage.export_batch_json()
    parsed   = json.loads(exported)
    assert isinstance(parsed, list)
