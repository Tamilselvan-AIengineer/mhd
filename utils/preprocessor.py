"""
Text Preprocessing Module
Handles tokenization, stopword removal, cleaning, and lemmatization.
"""

import re
import string
import os

import nltk

# ── NLTK resource download (safe for Streamlit Cloud) ────────────────
_NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "omw-1.4",
]

def _download_nltk():
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    for resource in _NLTK_RESOURCES:
        try:
            nltk.download(resource, quiet=True, download_dir=nltk_data_dir)
        except Exception:
            pass

_download_nltk()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

try:
    _STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    _download_nltk()
    _STOP_WORDS = set(stopwords.words("english"))

# Negation words to KEEP even if they appear in stopwords
_KEEP_WORDS = {
    "not", "no", "never", "nothing", "nobody", "none", "cannot",
    "can't", "won't", "don't", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't", "wouldn't",
    "couldn't", "shouldn't", "without", "against",
}

STOP_WORDS = _STOP_WORDS - _KEEP_WORDS

# ── Risk Lexicon ─────────────────────────────────────────────────────
RISK_LEXICON = {
    "high_risk": [
        "suicide", "suicidal", "kill myself", "end my life", "want to die",
        "no reason to live", "take my own life", "self harm", "self-harm",
        "cutting", "overdose", "die", "death", "dead", "hopeless",
        "worthless", "no point", "give up", "can't go on", "disappear",
        "burden", "empty", "numb", "trapped", "no way out", "goodbye forever",
        "farewell", "nothing matters", "ending it", "final goodbye",
    ],
    "moderate_risk": [
        "depressed", "depression", "anxiety", "anxious", "panic attack",
        "lonely", "alone", "isolated", "sad", "crying", "tears",
        "exhausted", "tired all the time", "overwhelmed", "stressed",
        "broken", "hurt", "pain", "suffering", "struggling",
        "can't sleep", "insomnia", "lost", "scared", "afraid",
        "worried", "no motivation", "feel nothing", "empty inside",
        "can't cope", "falling apart",
    ],
    "protective": [
        "help", "support", "friend", "family", "therapy", "therapist",
        "counselor", "better", "hope", "improve", "recover", "healing",
        "grateful", "thankful", "love", "care", "together", "not alone",
        "reached out", "getting help", "taking medication", "making progress",
    ],
}


# ── Core cleaning ─────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove URLs, mentions, hashtags and special characters. Lowercase."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # URLs
    text = re.sub(r"@\w+", " ", text)                  # @mentions
    text = re.sub(r"#(\w+)", r" \1 ", text)            # #hashtags → word
    text = re.sub(r"[^\w\s'\-]", " ", text)            # special chars
    text = re.sub(r"\s+", " ", text).strip()           # extra whitespace
    return text


def tokenize(text: str) -> list:
    """Word-tokenize text using NLTK."""
    try:
        return word_tokenize(text)
    except LookupError:
        _download_nltk()
        return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """Remove stopwords, keeping negation words."""
    return [t for t in tokens if t not in STOP_WORDS or t in _KEEP_WORDS]


def lemmatize_tokens(tokens: list) -> list:
    """Lemmatize a list of tokens."""
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str, return_tokens: bool = False):
    """
    Full preprocessing pipeline:
    clean → tokenize → filter → remove stopwords → lemmatize
    Returns a joined string by default, or a token list if return_tokens=True.
    """
    cleaned = clean_text(text)
    tokens  = tokenize(cleaned)
    tokens  = [t for t in tokens if t not in string.punctuation and len(t) > 1]
    tokens  = remove_stopwords(tokens)
    tokens  = lemmatize_tokens(tokens)
    if return_tokens:
        return tokens
    return " ".join(tokens)


def batch_preprocess(texts: list) -> list:
    """Preprocess a list of texts."""
    return [preprocess(t) for t in texts]


def extract_risk_keywords(text: str) -> dict:
    """
    Match text against the RISK_LEXICON.
    Returns dict with matched keywords per category.
    """
    text_lower = text.lower()
    found = {"high_risk": [], "moderate_risk": [], "protective": []}
    for category, phrases in RISK_LEXICON.items():
        for phrase in phrases:
            if phrase in text_lower:
                found[category].append(phrase)
    return found


def highlight_keywords(text: str, keywords: dict) -> str:
    """
    Return HTML string with risk keywords highlighted in color.
    high_risk → red, moderate_risk → orange, protective → green.
    """
    color_map = {
        "high_risk":     "#c0392b",
        "moderate_risk": "#e07b1a",
        "protective":    "#27ae60",
    }
    highlighted = text
    for category, phrases in keywords.items():
        color = color_map.get(category, "#888")
        for phrase in sorted(phrases, key=len, reverse=True):
            highlighted = re.sub(
                re.escape(phrase),
                f'<mark style="background:rgba({_hex_to_rgb(color)},0.25);'
                f'color:{color};padding:1px 4px;border-radius:3px;font-weight:500">{phrase}</mark>',
                highlighted,
                flags=re.IGNORECASE,
            )
    return highlighted


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #rrggbb to 'r,g,b' string."""
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))
