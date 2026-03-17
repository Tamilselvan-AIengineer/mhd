"""
Unit tests — Text Preprocessor
Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.preprocessor import (
    clean_text,
    tokenize,
    remove_stopwords,
    lemmatize_tokens,
    preprocess,
    extract_risk_keywords,
    highlight_keywords,
)


# ── clean_text ────────────────────────────────────────────────────────

def test_clean_text_lowercase():
    assert clean_text("HELLO WORLD") == "hello world"

def test_clean_text_removes_url():
    result = clean_text("Check this https://example.com out")
    assert "http" not in result
    assert "example" not in result

def test_clean_text_removes_mention():
    result = clean_text("Hello @username how are you")
    assert "@username" not in result

def test_clean_text_strips_hashtag_symbol():
    result = clean_text("Feeling #depressed today")
    assert "#" not in result
    assert "depressed" in result

def test_clean_text_handles_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""

def test_clean_text_collapses_whitespace():
    result = clean_text("too   many    spaces")
    assert "  " not in result


# ── preprocess ────────────────────────────────────────────────────────

def test_preprocess_returns_string():
    result = preprocess("I am feeling very sad today")
    assert isinstance(result, str)

def test_preprocess_returns_tokens():
    result = preprocess("I am feeling very sad today", return_tokens=True)
    assert isinstance(result, list)

def test_preprocess_removes_stopwords_but_keeps_negation():
    result = preprocess("I am not happy", return_tokens=True)
    assert "not" in result
    assert "am" not in result

def test_preprocess_lemmatizes():
    tokens = preprocess("running runs ran", return_tokens=True)
    assert "run" in tokens or "ran" in tokens

def test_preprocess_handles_empty():
    assert preprocess("") == ""


# ── extract_risk_keywords ────────────────────────────────────────────

def test_extract_high_risk():
    kw = extract_risk_keywords("I want to die and end my life")
    assert len(kw["high_risk"]) > 0

def test_extract_moderate_risk():
    kw = extract_risk_keywords("I have been feeling depressed and anxious")
    assert len(kw["moderate_risk"]) > 0

def test_extract_protective():
    kw = extract_risk_keywords("I reached out to my therapist for support")
    assert len(kw["protective"]) > 0

def test_extract_no_keywords():
    kw = extract_risk_keywords("The weather is nice today")
    assert kw["high_risk"]     == []
    assert kw["moderate_risk"] == []

def test_extract_empty_text():
    kw = extract_risk_keywords("")
    assert all(v == [] for v in kw.values())


# ── highlight_keywords ────────────────────────────────────────────────

def test_highlight_contains_mark_tag():
    kw     = extract_risk_keywords("I feel hopeless")
    result = highlight_keywords("I feel hopeless", kw)
    assert "<mark" in result

def test_highlight_no_keywords_returns_original():
    kw     = extract_risk_keywords("Nice sunny day")
    result = highlight_keywords("Nice sunny day", kw)
    assert "Nice sunny day" in result
