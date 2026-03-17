# 🧠 Mental Health Detection & Monitoring System

Early detection of mental health risk signals in social media text using NLP and machine learning.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## Features

| Feature | Description |
|---|---|
| **Risk Classification** | Normal · Moderate Risk · High Risk |
| **Emotion Detection** | Sadness, Anxiety, Anger, Hopelessness, Loneliness, Hope, Despair |
| **Sentiment Analysis** | VADER compound score + TextBlob polarity / subjectivity |
| **Risk Score** | 0–1 probability (XGBoost primary model) |
| **Keyword Highlighting** | TF-IDF + predefined risk lexicon with colour-coded output |
| **Trend Visualisation** | Rolling-average time-series charts across posts |
| **Emotion Timeline** | Stacked area chart tracking emotion shifts |
| **Alert System** | Configurable threshold (default 0.70) |
| **Batch Analysis** | Upload CSV for bulk processing |
| **JSON Storage** | Persistent analysis history in `data/` folder |

---

## Project Structure

```
mental_health_detector/
├── app.py                      ← Streamlit dashboard
├── train.py                    ← CLI model training script
├── requirements.txt
├── packages.txt                ← Streamlit Cloud system deps
├── .gitignore
├── .streamlit/
│   └── config.toml
├── models/
│   ├── __init__.py
│   └── ml_models.py            ← LR · SVM · XGBoost
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py
│   ├── sentiment.py
│   ├── visualizations.py
│   └── storage.py              ← JSON read/write
└── data/
    ├── analysis_history.json   ← auto-created
    └── batch_results.json      ← auto-created
```

---

## Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/mental-health-detector.git
cd mental-health-detector

# 2. Install
pip install -r requirements.txt

# 3. (Optional) Add real dataset
#    Place suicide_detection.csv in data/
#    CSV must have columns: text, label  (label = suicide / non-suicide)

# 4. Train
python train.py                           # uses built-in demo dataset
python train.py --data data/your.csv      # uses your real dataset

# 5. Launch
streamlit run app.py
```

Open **http://localhost:8501**

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Set **Main file path** → `app.py`
5. Click **Deploy**

On first boot the app auto-trains on a built-in demo dataset (~1 min).  
To use real data, commit your CSV and the app will detect and use it automatically.

---

## Dataset

Recommended: [Suicide and Depression Detection — Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

Expected CSV format:
```
text,class
"I feel hopeless and don't want to go on","suicide"
"Had a great day with friends today","non-suicide"
```

---

## ML Pipeline

```
Raw text
  → Clean (URLs · mentions · hashtags removed)
  → Tokenise (NLTK)
  → Remove stopwords (preserve negations)
  → Lemmatise (WordNetLemmatizer)
  → TF-IDF (unigrams + bigrams, 10 000 features)
  → XGBoost  → primary risk score + class
  → SVM      → emotion-weighted class
  → Logistic Regression → baseline class
  → VADER + TextBlob → sentiment
```

---

## JSON Storage

Analysis results are saved automatically:

- `data/analysis_history.json` — every single-text analysis
- `data/batch_results.json` — latest batch run

Both files can be downloaded from the **💾 JSON Storage** page inside the app.

> **Note:** On Streamlit Cloud the filesystem resets on each redeploy.  
> Download your JSON files before pushing updates.

---

## Tech Stack

- **Frontend:** Streamlit
- **ML:** Scikit-learn · XGBoost
- **NLP:** NLTK · TextBlob · VADER
- **Visualisation:** Plotly
- **Storage:** JSON (flat files)
- **Model persistence:** Joblib

---

## Disclaimer

This tool is for **research and educational purposes only**.  
It is not a substitute for professional mental health assessment.  
If you or someone you know is in crisis, please contact a qualified mental health professional or a local crisis helpline.
