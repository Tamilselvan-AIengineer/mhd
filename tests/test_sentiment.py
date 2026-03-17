"""
Unit tests — Sentiment Analysis & Emotion Detection
Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.sentiment import (
    get_vader_scores,
    get_textblob_scores,
    get_full_sentiment,
    compound_to_label,
    compound_to_severity,
    detect_emotions,
)


# ── VADER ─────────────────────────────────────────────────────────────

def test_vader_returns_required_keys():
    result = get_vader_scores("I feel great today!")
    assert all(k in result for k in ("compound", "positive", "negative", "neutral"))

def test_vader_positive_text():
    result = get_vader_scores("I am so happy and grateful!")
    assert result["compound"] > 0

def test_vader_negative_text():
    result = get_vader_scores("I feel hopeless and want to die.")
    assert result["compound"] < 0

def test_vader_scores_sum_to_one():
    result = get_vader_scores("Hello world")
    total  = round(result["positive"] + result["negative"] + result["neutral"], 2)
    assert total == 1.0


# ── TextBlob ──────────────────────────────────────────────────────────

def test_textblob_returns_required_keys():
    result = get_textblob_scores("This is amazing!")
    assert "polarity" in result and "subjectivity" in result

def test_textblob_polarity_range():
    result = get_textblob_scores("This is absolutely terrible.")
    assert -1.0 <= result["polarity"] <= 1.0

def test_textblob_subjectivity_range():
    result = get_textblob_scores("I think this is great.")
    assert 0.0 <= result["subjectivity"] <= 1.0


# ── compound_to_label ────────────────────────────────────────────────

def test_label_positive():
    assert compound_to_label(0.5) == "Positive"

def test_label_negative():
    assert compound_to_label(-0.5) == "Negative"

def test_label_neutral():
    assert compound_to_label(0.0) == "Neutral"

def test_label_boundary_positive():
    assert compound_to_label(0.05) == "Positive"

def test_label_boundary_negative():
    assert compound_to_label(-0.05) == "Negative"


# ── compound_to_severity ─────────────────────────────────────────────

def test_severity_strongly_negative():
    assert "Strongly" in compound_to_severity(-0.8)

def test_severity_moderately_negative():
    assert "Moderate" in compound_to_severity(-0.4)

def test_severity_positive():
    assert "Positive" in compound_to_severity(0.7)


# ── get_full_sentiment ───────────────────────────────────────────────

def test_full_sentiment_keys():
    result = get_full_sentiment("I feel hopeless.")
    required = (
        "label", "severity", "compound",
        "vader_positive", "vader_negative", "vader_neutral",
        "polarity", "subjectivity",
    )
    assert all(k in result for k in required)

def test_full_sentiment_empty_text():
    result = get_full_sentiment("")
    assert result["compound"] == 0.0
    assert result["label"]    == "Neutral"

def test_full_sentiment_none_input():
    result = get_full_sentiment(None)
    assert result["compound"] == 0.0


# ── detect_emotions ───────────────────────────────────────────────────

def test_detect_sadness():
    result = detect_emotions("I have been crying all day and feel so sad.")
    assert "Sadness" in result

def test_detect_anxiety():
    result = detect_emotions("I am so anxious and keep having panic attacks.")
    assert "Anxiety" in result

def test_detect_hopelessness():
    result = detect_emotions("Everything feels pointless and worthless.")
    assert "Hopelessness" in result

def test_detect_hope():
    result = detect_emotions("I feel hope and I know things will get better.")
    assert "Hope" in result

def test_detect_no_emotion_returns_neutral():
    result = detect_emotions("The weather is fine today.")
    assert result == ["Neutral"]

def test_detect_multiple_emotions():
    result = detect_emotions("I am both sad and very anxious about everything.")
    assert len(result) >= 2

def test_detect_empty_returns_neutral():
    assert detect_emotions("") == ["Neutral"]
