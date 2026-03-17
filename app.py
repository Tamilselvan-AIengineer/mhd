"""
Mental Health Detection & Monitoring System
Streamlit Dashboard — Main Application

Run locally : streamlit run app.py
Deploy      : push to GitHub → connect to share.streamlit.io
"""

# ── NLTK bootstrap (must be first) ───────────────────────────────────
import os
import sys
import nltk

_nltk_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(_nltk_dir, exist_ok=True)
for _res in ["punkt", "punkt_tab", "stopwords", "wordnet",
             "averaged_perceptron_tagger", "omw-1.4"]:
    try:
        nltk.download(_res, quiet=True, download_dir=_nltk_dir)
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Standard imports ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import streamlit as st

from models.ml_models import (
    predict_single, predict_batch, load_models,
    models_exist, train_models, get_top_features, LABEL_MAP,
)
from utils.sentiment    import get_full_sentiment, detect_emotions
from utils.preprocessor import extract_risk_keywords, highlight_keywords
from utils.storage      import (
    save_analysis, load_history,
    save_batch,    load_batch,
    clear_all,     get_storage_stats,
    export_history_json, export_batch_json,
)
from utils.visualizations import (
    risk_gauge, sentiment_bar, emotion_radar,
    model_comparison_bar, risk_trend_chart,
    emotion_timeline, risk_distribution_pie,
    keyword_frequency_bar, sentiment_trend_chart,
)
from train import generate_demo_dataset


# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main .block-container { padding-top: 1.5rem; }
  .alert-high {
    background: rgba(192,57,43,0.10); border: 1px solid rgba(192,57,43,0.40);
    border-radius: 10px; padding: 12px 16px; color: #c0392b; font-weight: 600;
    margin-bottom: 1rem;
  }
  .alert-mod {
    background: rgba(224,123,26,0.10); border: 1px solid rgba(224,123,26,0.40);
    border-radius: 10px; padding: 12px 16px; color: #e07b1a; font-weight: 600;
    margin-bottom: 1rem;
  }
  .alert-ok {
    background: rgba(39,174,96,0.10); border: 1px solid rgba(39,174,96,0.40);
    border-radius: 10px; padding: 12px 16px; color: #27ae60; font-weight: 600;
    margin-bottom: 1rem;
  }
  .kw-high  { background:#fde8e8; color:#c0392b; padding:3px 9px;
              border-radius:4px; margin:2px; font-size:13px; display:inline-block; }
  .kw-mod   { background:#fef3e2; color:#e07b1a; padding:3px 9px;
              border-radius:4px; margin:2px; font-size:13px; display:inline-block; }
  .kw-prot  { background:#e8f8ee; color:#27ae60; padding:3px 9px;
              border-radius:4px; margin:2px; font-size:13px; display:inline-block; }
  .sec-head { font-size:1.05rem; font-weight:600; color:#2c3e50;
              margin-bottom:.4rem; }
  div[data-testid="metric-container"] {
    border: 1px solid #e9ecef; border-radius: 10px; padding: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Loading / training models …")
def load_or_train():
    if models_exist():
        return load_models()
    st.toast("No saved models found — training on demo data. Please wait ~1 min …")
    df = generate_demo_dataset(n_samples=3000)
    train_models(df)
    return load_models()


models = load_or_train()


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Mental Health Monitor")
    st.caption("ML-powered risk detection from text")
    st.divider()

    page = st.radio(
        "Navigate to",
        ["🔍 Single Analysis", "📊 Batch Analysis",
         "📈 Trend Dashboard",  "💾 JSON Storage",
         "🛠 Model Info"],
        index=0,
    )

    st.divider()
    st.markdown("### ⚙️ Settings")
    alert_threshold   = st.slider("Alert threshold",    0.40, 0.95, 0.70, 0.05)
    rolling_window    = st.slider("Trend rolling window", 2,   15,    5)
    show_model_cmp    = st.checkbox("Show model comparison", value=True)

    st.divider()
    if st.button("🔄 Retrain on demo data", use_container_width=True):
        with st.spinner("Training …"):
            df = generate_demo_dataset(n_samples=3000)
            train_models(df)
            st.cache_resource.clear()
        st.success("Retrained!")
        st.rerun()

    stats = get_storage_stats()
    st.caption(
        f"📁 History: {stats['history_count']} records  \n"
        f"📁 Batch: {stats['batch_count']} records"
    )


# ════════════════════════════════════════════════════════════════════
#  PAGE 1 — SINGLE ANALYSIS
# ════════════════════════════════════════════════════════════════════
if "Single" in page:
    st.title("🔍 Mental Health Risk Analysis")
    st.markdown("Analyse a single social-media post for mental health risk signals.")

    SAMPLES = {
        "High Risk sample":     "I can't take this anymore. I've been thinking about ending it all. Nobody would miss me if I was gone.",
        "Moderate Risk sample": "I've been feeling really depressed and anxious lately. Can't sleep, can't focus. Everything feels overwhelming.",
        "Normal sample":        "Had a tough week but talked to my therapist and feel better. Taking it one day at a time.",
        "Type your own …":      "",
    }

    choice   = st.selectbox("Quick sample", list(SAMPLES.keys()))
    default  = SAMPLES[choice]
    user_txt = st.text_area("Enter text to analyse", value=default, height=120,
                             placeholder="Paste or type a social media post …")
    col_btn, _ = st.columns([1, 4])
    run = col_btn.button("🔎 Analyse", type="primary")

    if run and user_txt.strip():
        with st.spinner("Analysing …"):
            pred      = predict_single(user_txt, models)
            sentiment = get_full_sentiment(user_txt)
            emotions  = detect_emotions(user_txt)
            keywords  = extract_risk_keywords(user_txt)
            pred["alert"] = pred["risk_score"] >= alert_threshold

            # ── Save to JSON ──────────────────────────────────────────
            save_analysis({
                "text":       user_txt[:120],
                "label":      pred["label"],
                "risk_score": pred["risk_score"],
                "compound":   sentiment["compound"],
                "emotions":   emotions,
            })

        # ── Alert banner ──────────────────────────────────────────────
        if pred["alert"]:
            st.markdown(
                f'<div class="alert-high">⚑ HIGH RISK ALERT — score {pred["risk_score"]:.3f} '
                f'exceeds threshold {alert_threshold:.2f}. Immediate support recommended.</div>',
                unsafe_allow_html=True,
            )
        elif pred["label"] == "Moderate Risk":
            st.markdown(
                f'<div class="alert-mod">⚐ Moderate Risk — score {pred["risk_score"]:.3f}. '
                f'Monitor and consider reaching out.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="alert-ok">✓ Normal — score {pred["risk_score"]:.3f}. '
                f'No immediate concern detected.</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Row 1: Gauge | Sentiment metrics | Emotion radar ─────────
        c1, c2, c3 = st.columns([1.2, 1, 1])

        with c1:
            st.markdown('<div class="sec-head">Risk Score</div>', unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(pred["risk_score"], pred["label"]),
                            use_container_width=True)

        with c2:
            st.markdown('<div class="sec-head">Sentiment</div>', unsafe_allow_html=True)
            st.metric("VADER Compound",  f"{sentiment['compound']:.3f}")
            st.metric("TextBlob Polarity", f"{sentiment['polarity']:.3f}")
            st.metric("Subjectivity",    f"{sentiment['subjectivity']:.3f}")
            st.metric("Label",           sentiment["severity"])
            st.plotly_chart(sentiment_bar(sentiment), use_container_width=True)

        with c3:
            st.markdown('<div class="sec-head">Emotion Radar</div>', unsafe_allow_html=True)
            st.plotly_chart(emotion_radar(emotions), use_container_width=True)

        # ── Row 2: Keywords | Model comparison ───────────────────────
        c4, c5 = st.columns([1, 1.4])

        with c4:
            st.markdown('<div class="sec-head">🔑 Keyword Highlighting</div>',
                        unsafe_allow_html=True)
            if any(keywords.values()):
                if keywords["high_risk"]:
                    st.markdown("**High Risk**")
                    st.markdown(
                        " ".join(f'<span class="kw-high">{k}</span>'
                                 for k in keywords["high_risk"]),
                        unsafe_allow_html=True,
                    )
                if keywords["moderate_risk"]:
                    st.markdown("**Moderate Risk**")
                    st.markdown(
                        " ".join(f'<span class="kw-mod">{k}</span>'
                                 for k in keywords["moderate_risk"]),
                        unsafe_allow_html=True,
                    )
                if keywords["protective"]:
                    st.markdown("**Protective Factors**")
                    st.markdown(
                        " ".join(f'<span class="kw-prot">{k}</span>'
                                 for k in keywords["protective"]),
                        unsafe_allow_html=True,
                    )
                st.plotly_chart(keyword_frequency_bar(keywords),
                                use_container_width=True)
            else:
                st.info("No specific risk keywords detected.")

            st.markdown("**Detected Emotions**")
            for e in emotions:
                st.markdown(f"• {e}")

        with c5:
            if show_model_cmp:
                st.markdown('<div class="sec-head">📊 Model Comparison</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(model_comparison_bar(pred), use_container_width=True)

                rows = []
                for mk, name in [("xgboost",             "XGBoost"),
                                  ("svm",                  "SVM"),
                                  ("logistic_regression",  "Logistic Reg.")]:
                    prob = pred.get(mk, {}).get("probabilities", {})
                    rows.append({
                        "Model":       name,
                        "Predicted":   pred.get(mk, {}).get("class", "—"),
                        "P(Normal)":   f"{prob.get('Normal',0):.3f}",
                        "P(Moderate)": f"{prob.get('Moderate Risk',0):.3f}",
                        "P(High)":     f"{prob.get('High Risk',0):.3f}",
                    })
                st.dataframe(pd.DataFrame(rows),
                             use_container_width=True, hide_index=True)

        # ── Highlighted text ──────────────────────────────────────────
        if any(keywords.values()):
            st.divider()
            st.markdown('<div class="sec-head">📝 Highlighted Text</div>',
                        unsafe_allow_html=True)
            highlighted = highlight_keywords(user_txt, keywords)
            st.markdown(f'<p style="line-height:1.8">{highlighted}</p>',
                        unsafe_allow_html=True)

    elif run:
        st.warning("Please enter some text first.")


# ════════════════════════════════════════════════════════════════════
#  PAGE 2 — BATCH ANALYSIS
# ════════════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.title("📊 Batch Dataset Analysis")
    st.markdown("Upload a CSV or run the built-in demo to analyse multiple posts at once.")

    tab_upload, tab_demo = st.tabs(["⬆ Upload CSV", "🗂 Demo Dataset"])

    # ── Upload tab ────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "CSV must contain a 'text' (or 'post') column", type=["csv"]
        )
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                st.dataframe(df_up.head(), use_container_width=True)
                text_col = st.selectbox("Select text column", df_up.columns.tolist())
                max_rows = st.slider("Max rows", 10,
                                     min(500, len(df_up)),
                                     min(100,  len(df_up)))

                if st.button("▶ Run", type="primary", key="run_upload"):
                    texts = df_up[text_col].fillna("").head(max_rows).tolist()
                    prog  = st.progress(0, text="Analysing …")
                    results = []
                    for i, txt in enumerate(texts):
                        p = predict_single(txt, models)
                        s = get_full_sentiment(txt)
                        e = detect_emotions(txt)
                        p["alert"] = p["risk_score"] >= alert_threshold
                        results.append({
                            "text_preview": txt[:70] + "…" if len(txt) > 70 else txt,
                            "label":        p["label"],
                            "risk_score":   p["risk_score"],
                            "alert":        p["alert"],
                            "sentiment":    s["label"],
                            "compound":     round(s["compound"], 4),
                            "emotions":     ", ".join(e),
                        })
                        prog.progress((i + 1) / len(texts),
                                      text=f"Analysing {i+1}/{len(texts)} …")
                    save_batch(results)
                    prog.empty()
                    st.success(f"✅ Analysed {len(results)} posts — saved to JSON.")
                    st.rerun()
            except Exception as ex:
                st.error(f"Error reading file: {ex}")

    # ── Demo tab ──────────────────────────────────────────────────────
    with tab_demo:
        n_demo = st.slider("Number of demo posts", 20, 300, 60)
        if st.button("▶ Analyse Demo", type="primary", key="run_demo"):
            demo_df = generate_demo_dataset(n_samples=n_demo)
            prog    = st.progress(0, text="Analysing …")
            results = []
            for i, row in demo_df.iterrows():
                p = predict_single(row["text"], models)
                s = get_full_sentiment(row["text"])
                e = detect_emotions(row["text"])
                p["alert"] = p["risk_score"] >= alert_threshold
                results.append({
                    "text_preview": row["text"][:70] + "…" if len(row["text"]) > 70 else row["text"],
                    "label":        p["label"],
                    "risk_score":   round(p["risk_score"], 4),
                    "alert":        p["alert"],
                    "sentiment":    s["label"],
                    "compound":     round(s["compound"], 4),
                    "emotions":     ", ".join(e),
                })
                prog.progress((i + 1) / n_demo, text=f"Analysing {i+1}/{n_demo} …")
            save_batch(results)
            prog.empty()
            st.success(f"✅ Analysed {n_demo} posts — saved to JSON.")
            st.rerun()

    # ── Results panel ─────────────────────────────────────────────────
    batch_data = load_batch()
    if batch_data:
        st.divider()
        st.subheader("Batch Results")
        res_df = pd.DataFrame(batch_data)

        m1, m2, m3, m4 = st.columns(4)
        rc   = res_df["label"].value_counts().to_dict()
        total = len(res_df)
        m1.metric("Total Posts",      total)
        m2.metric("High Risk",        rc.get("High Risk", 0),
                  delta=f"{rc.get('High Risk',0)/total*100:.1f}%")
        m3.metric("Moderate Risk",    rc.get("Moderate Risk", 0),
                  delta=f"{rc.get('Moderate Risk',0)/total*100:.1f}%")
        m4.metric("Alerts Triggered", int(res_df["alert"].sum()))

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(risk_distribution_pie(rc), use_container_width=True)
        with c2:
            st.plotly_chart(risk_trend_chart(res_df, window=rolling_window),
                            use_container_width=True)

        st.subheader("Detailed Table")

        def _colour_label(val):
            m = {"High Risk":     "color:#c0392b;font-weight:bold",
                 "Moderate Risk": "color:#e07b1a;font-weight:bold",
                 "Normal":        "color:#27ae60"}
            return m.get(val, "")

        styled = res_df.style.applymap(_colour_label, subset=["label"])
        st.dataframe(styled, use_container_width=True, height=380)

        col_dl1, col_dl2, _ = st.columns([1, 1, 3])
        col_dl1.download_button(
            "⬇ Download CSV",
            res_df.to_csv(index=False),
            "batch_results.csv",
            "text/csv",
        )
        col_dl2.download_button(
            "⬇ Download JSON",
            export_batch_json(),
            "batch_results.json",
            "application/json",
        )
    else:
        st.info("No batch results yet. Run a batch analysis above.")


# ════════════════════════════════════════════════════════════════════
#  PAGE 3 — TREND DASHBOARD
# ════════════════════════════════════════════════════════════════════
elif "Trend" in page:
    st.title("📈 Mental Health Trend Dashboard")

    history    = load_history()
    batch_data = load_batch()

    if not history and not batch_data:
        st.info("No data yet. Run a Single Analysis or Batch Analysis first.")
        st.stop()

    source = "history"
    if batch_data:
        use_batch = st.checkbox(
            f"Use batch results ({len(batch_data)} posts) instead of history "
            f"({len(history)} posts)",
            value=len(batch_data) > len(history),
        )
        if use_batch:
            source = "batch"

    data = batch_data if source == "batch" else history

    trend_df = pd.DataFrame([{
        "label":      r.get("label", "Normal"),
        "risk_score": r.get("risk_score", 0.0),
        "compound":   r.get("compound", 0.0),
    } for r in data])

    emotion_data = [
        {"post_id": i, "emotions": r.get("emotions", ["Neutral"])
         if isinstance(r.get("emotions"), list)
         else r.get("emotions", "Neutral").split(", ")}
        for i, r in enumerate(data)
    ]

    st.markdown(f"Showing **{len(trend_df)} posts** from **{source}**.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Risk Score",   f"{trend_df['risk_score'].mean():.3f}")
    m2.metric("Peak Risk Score",  f"{trend_df['risk_score'].max():.3f}")
    m3.metric("Avg Sentiment",    f"{trend_df['compound'].mean():.3f}")
    m4.metric("High Risk Posts",  int((trend_df["label"] == "High Risk").sum()))

    st.plotly_chart(risk_trend_chart(trend_df, window=rolling_window),
                    use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(emotion_timeline(emotion_data), use_container_width=True)
    with c2:
        rc = trend_df["label"].value_counts().to_dict()
        st.plotly_chart(risk_distribution_pie(rc), use_container_width=True)

    st.plotly_chart(sentiment_trend_chart(trend_df), use_container_width=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE 4 — JSON STORAGE
# ════════════════════════════════════════════════════════════════════
elif "Storage" in page:
    st.title("💾 JSON Storage Manager")
    st.markdown("View and manage data saved to local JSON files.")

    stats = get_storage_stats()

    c1, c2, c3 = st.columns(3)
    c1.metric("History records",  stats["history_count"])
    c2.metric("Batch records",    stats["batch_count"])
    c3.metric("Storage path",     "data/")

    st.divider()

    tab_h, tab_b = st.tabs(["📋 Analysis History", "📦 Batch Results"])

    with tab_h:
        history = load_history()
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True, height=400)
            col1, col2, _ = st.columns([1, 1, 3])
            col1.download_button(
                "⬇ Download JSON",
                export_history_json(),
                "analysis_history.json",
                "application/json",
            )
            col2.download_button(
                "⬇ Download CSV",
                pd.DataFrame(history).to_csv(index=False),
                "analysis_history.csv",
                "text/csv",
            )
        else:
            st.info("No history records yet.")

    with tab_b:
        batch = load_batch()
        if batch:
            st.dataframe(pd.DataFrame(batch), use_container_width=True, height=400)
            col1, col2, _ = st.columns([1, 1, 3])
            col1.download_button(
                "⬇ Download JSON",
                export_batch_json(),
                "batch_results.json",
                "application/json",
            )
            col2.download_button(
                "⬇ Download CSV",
                pd.DataFrame(batch).to_csv(index=False),
                "batch_results.csv",
                "text/csv",
            )
        else:
            st.info("No batch results yet.")

    st.divider()
    st.subheader("⚠ Danger Zone")
    col_clear, _ = st.columns([1, 4])
    if col_clear.button("🗑 Clear ALL stored data", type="secondary"):
        clear_all()
        st.success("All data cleared.")
        st.rerun()

    st.caption(
        "JSON files are stored at `data/analysis_history.json` and "
        "`data/batch_results.json`. On Streamlit Cloud these reset on each "
        "redeploy — download your data before pushing new changes."
    )


# ════════════════════════════════════════════════════════════════════
#  PAGE 5 — MODEL INFO
# ════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.title("🛠 Model Information")

    st.subheader("Architecture")
    st.markdown("""
| Component | Detail |
|---|---|
| **Frontend** | Streamlit |
| **Primary model** | XGBoost (gradient-boosted ensemble) |
| **Emotion model** | SVM with LinearSVC + probability calibration |
| **Baseline** | Logistic Regression (multinomial) |
| **Features** | TF-IDF — unigrams + bigrams, max 10 000 features |
| **Sentiment** | VADER compound + TextBlob polarity / subjectivity |
| **Preprocessing** | Tokenise → stopword removal → lemmatise |
| **Storage** | JSON flat files (data/) via `utils/storage.py` |
| **Alert threshold** | Configurable in sidebar (default 0.70) |
| **Risk classes** | Normal · Moderate Risk · High Risk |
""")

    st.subheader("Top Predictive Features (Logistic Regression coefficients)")
    with st.spinner("Loading …"):
        try:
            top = get_top_features(models, n=15)
            c1, c2, c3 = st.columns(3)
            for col, (cls, feats) in zip([c1, c2, c3], top.items()):
                with col:
                    st.markdown(f"**{cls}**")
                    for f in feats:
                        st.markdown(f"• `{f}`")
        except Exception as ex:
            st.warning(f"Could not load features: {ex}")

    st.subheader("Preprocessing Pipeline")
    st.code("""
Raw text
  ↓  Lowercase · remove URLs, @mentions, #hashtags
  ↓  Tokenise (NLTK word_tokenize)
  ↓  Drop punctuation tokens < 2 chars
  ↓  Remove stopwords (preserve negations: not, no, never …)
  ↓  Lemmatise (WordNetLemmatizer)
  ↓  TF-IDF vectorise (unigrams + bigrams, max 10 000 features)
  ↓  XGBoost → primary risk score + class
  ↓  SVM     → emotion-weighted class
  ↓  LR      → baseline class
  ↓  VADER + TextBlob → sentiment scores
Output: label · risk_score · sentiment · emotions · keywords
""", language="text")

    st.subheader("Project File Structure")
    st.code("""
mental_health_detector/
├── app.py                      ← Streamlit dashboard (this file)
├── train.py                    ← CLI training script
├── requirements.txt
├── packages.txt                ← Streamlit Cloud system deps
├── .gitignore
├── .streamlit/
│   └── config.toml             ← Theme + server settings
├── models/
│   ├── __init__.py
│   └── ml_models.py            ← LR · SVM · XGBoost
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py         ← Cleaning · tokenising · lemmatising
│   ├── sentiment.py            ← VADER · TextBlob · emotion detection
│   ├── visualizations.py       ← All Plotly charts
│   └── storage.py              ← JSON read/write helpers
└── data/
    ├── analysis_history.json   ← Auto-created on first analysis
    └── batch_results.json      ← Auto-created on first batch run
""", language="text")
