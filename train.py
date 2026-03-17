"""
Training Script — Mental Health Detection System
Usage:
    python train.py                               # uses demo dataset
    python train.py --data data/your_file.csv     # uses real dataset
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ml_models import train_models


# ── Demo dataset generator ────────────────────────────────────────────

def generate_demo_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """
    Builds a synthetic dataset for testing when no real CSV is provided.
    In production replace with the Suicide Detection Dataset from Kaggle.
    """
    np.random.seed(42)

    high_risk = [
        "I can't take this anymore, I want to end it all.",
        "Nobody would miss me if I just disappeared.",
        "I've been planning to kill myself for weeks now.",
        "Life has no meaning, I feel completely hopeless.",
        "I'm writing this as my final goodbye to everyone.",
        "The pain is unbearable and I just want it to stop.",
        "I have nothing left to live for, no point going on.",
        "Thinking about suicide constantly every single day.",
        "I've decided there is no point in continuing.",
        "Goodbye everyone. I've made my decision finally.",
        "I want to die. There is no other way out for me.",
        "I feel trapped with no way out of this darkness.",
        "I am a burden to everyone around me, better off gone.",
        "Everything is numb and empty, I feel completely dead inside.",
        "I've written notes for everyone. This is the end.",
    ]

    moderate_risk = [
        "I've been really depressed lately and can't get out of bed.",
        "Anxiety is consuming my entire life, I can't sleep at all.",
        "Feeling so completely alone and like nobody understands me.",
        "I've been crying every single day for the past few weeks.",
        "Struggling with really dark thoughts but trying to stay strong.",
        "Can't stop worrying about everything, constant panic attacks.",
        "I feel broken and completely worthless as a person.",
        "Everything feels overwhelming, I'm exhausted all the time.",
        "Having really bad mental health days recently.",
        "Feeling lost and confused about everything in my life.",
        "I'm so tired of feeling this way, depressed every morning.",
        "Isolated from everyone, severe loneliness and social anxiety.",
        "Can't eat, can't sleep, can't function because of depression.",
        "The anxiety is so bad I can barely leave my house anymore.",
        "Feeling like I'm falling apart and can't hold it together.",
    ]

    normal = [
        "Had a great day at work today! Feeling accomplished.",
        "Feeling a bit stressed but going for a run really helped.",
        "Talked to my therapist today and feeling more grounded now.",
        "Rough week but the weekend is almost here, excited!",
        "Really grateful for my amazing friends and supportive family.",
        "Trying to take things one day at a time, making progress.",
        "Not feeling 100% today but that's okay, tomorrow is new.",
        "Made some really good progress on my personal goals today.",
        "Enjoying a quiet evening at home reading a good book.",
        "Life is good and I'm trying to count my blessings daily.",
        "Had a minor setback but choosing to stay positive overall.",
        "Working on my mental health with therapy and it's helping.",
        "Feeling a bit down but reached out to a friend who helped.",
        "Meditation this morning really helped set a positive tone.",
        "Celebrated a small win today, proud of how far I've come.",
    ]

    texts, labels = [], []
    per_class = n_samples // 3

    for _ in range(per_class):
        texts.append(np.random.choice(high_risk))
        labels.append("suicide")

    for _ in range(per_class):
        texts.append(np.random.choice(moderate_risk))
        labels.append("moderate")

    for _ in range(per_class):
        texts.append(np.random.choice(normal))
        labels.append("non-suicide")

    df = pd.DataFrame({"text": texts, "label": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ── Dataset loader ────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV dataset and auto-detect text / label columns.
    Falls back to the demo dataset if the file is not found.
    """
    if not os.path.exists(path):
        print(f"[WARN] Dataset not found at '{path}'. Using demo dataset.")
        df = generate_demo_dataset(n_samples=3000)
        print(f"[INFO] Demo dataset: {len(df)} rows")
        return df

    df = pd.read_csv(path)
    print(f"[INFO] Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"[INFO] Columns: {list(df.columns)}")

    text_col  = None
    label_col = None

    for c in df.columns:
        cl = c.lower()
        if text_col is None and any(k in cl for k in ("text", "post", "tweet", "content", "message")):
            text_col = c
        if label_col is None and any(k in cl for k in ("label", "class", "target", "suicide", "category")):
            label_col = c

    if text_col is None or label_col is None:
        raise ValueError(
            f"Cannot find text/label columns in {list(df.columns)}. "
            "Rename them to 'text' and 'label'."
        )

    print(f"[INFO] text_col='{text_col}', label_col='{label_col}'")
    print(f"[INFO] Label distribution:\n{df[label_col].value_counts().to_string()}")

    df = (df[[text_col, label_col]]
            .rename(columns={text_col: "text", label_col: "label"})
            .dropna()
            .reset_index(drop=True))
    return df


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Mental Health Detection Models")
    parser.add_argument("--data", type=str, default="data/suicide_detection.csv",
                        help="Path to the training CSV file")
    args = parser.parse_args()

    print("=" * 56)
    print("  Mental Health Detection System — Model Training")
    print("=" * 56)

    df = load_dataset(args.data)
    results = train_models(df)

    print("\n── Results ─────────────────────────────────────────────")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper().replace('_', ' ')}")
        print(f"  Accuracy : {metrics['accuracy'] * 100:.2f}%")
        r = metrics["report"]
        for cls in ["Normal", "Moderate Risk", "High Risk"]:
            if cls in r:
                print(f"  {cls:15s}  F1={r[cls]['f1-score']:.3f}  "
                      f"P={r[cls]['precision']:.3f}  "
                      f"R={r[cls]['recall']:.3f}")

    print("\n[INFO] Done. Run:  streamlit run app.py")


if __name__ == "__main__":
    main()
