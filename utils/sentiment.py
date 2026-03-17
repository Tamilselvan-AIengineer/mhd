"""
Sentiment Analysis Module
VADER for social-media polarity scoring.
TextBlob for polarity + subjectivity.
Rule-based emotion detection via keyword lexicon.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

_vader = SentimentIntensityAnalyzer()

# ── Emotion lexicon ───────────────────────────────────────────────────
_EMOTION_LEXICON = {
    "Sadness": [
        "sad", "cry", "crying", "tears", "grief", "sorrow", "depressed",
        "miserable", "unhappy", "heartbroken", "gloomy", "melancholy",
    ],
    "Anxiety": [
        "anxious", "anxiety", "panic", "worried", "worry", "nervous",
        "scared", "fear", "dread", "uneasy", "restless", "tense",
    ],
    "Anger": [
        "angry", "anger", "rage", "furious", "hate", "mad", "frustrated",
        "irritated", "resentment", "bitter", "hostile",
    ],
    "Hopelessness": [
        "hopeless", "no hope", "pointless", "worthless", "useless",
        "give up", "no reason", "meaningless", "futile", "despair",
    ],
    "Loneliness": [
        "lonely", "alone", "isolated", "nobody", "no one", "abandoned",
        "left out", "unwanted", "invisible", "disconnected",
    ],
    "Exhaustion": [
        "tired", "exhausted", "drained", "burnt out", "burnout",
        "fatigued", "no energy", "overwhelmed", "worn out",
    ],
    "Hope": [
        "hope", "better", "improve", "recover", "healing", "forward",
        "grateful", "thankful", "optimistic", "looking forward",
    ],
    "Despair": [
        "despair", "dark", "suffer", "suffering", "anguish", "torment",
        "agony", "devastated", "shattered", "broken",
    ],
}


# ── VADER ─────────────────────────────────────────────────────────────

def get_vader_scores(text: str) -> dict:
    """Return raw VADER polarity scores."""
    s = _vader.polarity_scores(text)
    return {
        "compound": s["compound"],
        "positive": s["pos"],
        "negative": s["neg"],
        "neutral":  s["neu"],
    }


# ── TextBlob ──────────────────────────────────────────────────────────

def get_textblob_scores(text: str) -> dict:
    """Return TextBlob polarity and subjectivity."""
    blob = TextBlob(text)
    return {
        "polarity":     blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
    }


# ── Labels ────────────────────────────────────────────────────────────

def compound_to_label(compound: float) -> str:
    """Map VADER compound score to a simple sentiment label."""
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    return "Neutral"


def compound_to_severity(compound: float) -> str:
    """Map compound score to a descriptive severity label."""
    if compound <= -0.60:
        return "Strongly Negative"
    elif compound <= -0.20:
        return "Moderately Negative"
    elif compound <= 0.05:
        return "Slightly Negative / Neutral"
    elif compound <= 0.40:
        return "Slightly Positive"
    return "Positive"


# ── Combined ─────────────────────────────────────────────────────────

def get_full_sentiment(text: str) -> dict:
    """
    Run both VADER and TextBlob and return a consolidated dict.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "label": "Neutral", "severity": "Neutral",
            "compound": 0.0, "vader_positive": 0.0,
            "vader_negative": 0.0, "vader_neutral": 1.0,
            "polarity": 0.0, "subjectivity": 0.0,
        }

    vader  = get_vader_scores(text)
    blob   = get_textblob_scores(text)
    label    = compound_to_label(vader["compound"])
    severity = compound_to_severity(vader["compound"])

    return {
        "label":          label,
        "severity":       severity,
        "compound":       round(vader["compound"],    4),
        "vader_positive": round(vader["positive"],    4),
        "vader_negative": round(vader["negative"],    4),
        "vader_neutral":  round(vader["neutral"],     4),
        "polarity":       round(blob["polarity"],     4),
        "subjectivity":   round(blob["subjectivity"], 4),
    }


# ── Emotion detection ─────────────────────────────────────────────────

def detect_emotions(text: str) -> list:
    """
    Rule-based emotion detection.
    Returns a list of detected emotion labels.
    Falls back to ['Neutral'] if nothing is detected.
    """
    if not isinstance(text, str) or not text.strip():
        return ["Neutral"]

    text_lower = text.lower()
    detected   = []

    for emotion, keywords in _EMOTION_LEXICON.items():
        for kw in keywords:
            if kw in text_lower:
                detected.append(emotion)
                break   # one match per emotion is enough

    return detected if detected else ["Neutral"]


# ── Batch ─────────────────────────────────────────────────────────────

def batch_sentiment(texts: list) -> list:
    """Run sentiment analysis on a list of texts."""
    return [get_full_sentiment(t) for t in texts]
