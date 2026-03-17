"""
Visualization Module
All Plotly charts used throughout the Streamlit dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Color constants ───────────────────────────────────────────────────
C = {
    "Normal":        "#27ae60",
    "Moderate Risk": "#e07b1a",
    "High Risk":     "#c0392b",
    "positive":      "#27ae60",
    "neutral":       "#95a5a6",
    "negative":      "#c0392b",
    "blue":          "#4a9eda",
    "purple":        "#9b59b6",
    "teal":          "#16a085",
}

_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#2c3e50"),
    margin=dict(l=40, r=20, t=44, b=40),
)


# ── 1. Risk Gauge ─────────────────────────────────────────────────────

def risk_gauge(score: float, label: str) -> go.Figure:
    """Gauge chart showing the 0–1 risk probability score."""
    color = C.get(label, "#888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 3),
        number={"valueformat": ".3f", "font": {"size": 38, "color": color}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#ccc",
                     "tickvals": [0, 0.35, 0.70, 1.0],
                     "ticktext": ["0", "0.35", "0.70", "1"]},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,    0.35], "color": "rgba(39,174,96,0.12)"},
                {"range": [0.35, 0.70], "color": "rgba(224,123,26,0.12)"},
                {"range": [0.70, 1.0],  "color": "rgba(192,57,43,0.12)"},
            ],
            "threshold": {
                "line":      {"color": "#e74c3c", "width": 3},
                "thickness": 0.75,
                "value":     0.70,
            },
        },
        title={"text": f"<b>{label}</b>", "font": {"size": 15, "color": color}},
    ))
    fig.update_layout(**_THEME, height=260)
    return fig


# ── 2. Sentiment horizontal bar ───────────────────────────────────────

def sentiment_bar(sentiment: dict) -> go.Figure:
    """Stacked horizontal bar showing VADER positive / neutral / negative split."""
    categories = ["Positive", "Neutral", "Negative"]
    values     = [
        sentiment.get("vader_positive", 0),
        sentiment.get("vader_neutral",  0),
        sentiment.get("vader_negative", 0),
    ]
    colors = [C["positive"], C["neutral"], C["negative"]]

    fig = go.Figure()
    for cat, val, col in zip(categories, values, colors):
        fig.add_trace(go.Bar(
            x=[val], y=[""],
            orientation="h",
            name=cat,
            marker_color=col,
            text=f"{val:.0%}" if val > 0.05 else "",
            textposition="inside",
            insidetextanchor="middle",
        ))

    fig.update_layout(
        **_THEME,
        barmode="stack",
        height=90,
        showlegend=True,
        legend=dict(orientation="h", y=2.2, x=0, font=dict(size=11)),
        xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False,
                   showline=False, zeroline=False),
        yaxis=dict(showticklabels=False),
        margin=dict(l=0, r=0, t=28, b=0),
    )
    return fig


# ── 3. Emotion radar ──────────────────────────────────────────────────

def emotion_radar(emotions: list) -> go.Figure:
    """Radar / spider chart of detected emotions."""
    all_emotions = [
        "Sadness", "Anxiety", "Anger", "Hopelessness",
        "Loneliness", "Exhaustion", "Hope", "Despair",
    ]
    vals         = [1 if e in emotions else 0 for e in all_emotions]
    vals_closed  = vals + [vals[0]]
    cats_closed  = all_emotions + [all_emotions[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(74,158,218,0.18)",
        line=dict(color=C["blue"], width=2),
        marker=dict(size=6, color=C["blue"]),
    ))
    fig.update_layout(
        **_THEME,
        height=300,
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False,
    )
    return fig


# ── 4. Model probability comparison bar ──────────────────────────────

def model_comparison_bar(model_preds: dict) -> go.Figure:
    """Grouped bar — probability per class for each model."""
    model_display = {
        "xgboost":              "XGBoost",
        "svm":                  "SVM",
        "logistic_regression":  "Logistic Reg.",
    }
    classes    = ["Normal", "Moderate Risk", "High Risk"]
    fig = go.Figure()

    for cls in classes:
        vals = []
        for mk in ["xgboost", "svm", "logistic_regression"]:
            p = model_preds.get(mk, {}).get("probabilities", {})
            vals.append(round(p.get(cls, 0.0), 3))
        fig.add_trace(go.Bar(
            name=cls,
            x=list(model_display.values()),
            y=vals,
            marker_color=C[cls],
            text=[f"{v:.3f}" for v in vals],
            textposition="auto",
        ))

    fig.update_layout(
        **_THEME,
        barmode="group",
        height=320,
        title="Model Probability Comparison",
        yaxis=dict(range=[0, 1], title="Probability"),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


# ── 5. Risk trend line chart ──────────────────────────────────────────

def risk_trend_chart(df: pd.DataFrame, window: int = 5) -> go.Figure:
    """
    Line + scatter chart of risk scores over time.
    df must contain columns: 'risk_score', 'label'.
    """
    df = df.copy().reset_index(drop=True)
    if "risk_score" not in df.columns:
        return go.Figure()

    df["rolling_avg"] = (
        df["risk_score"].rolling(window=window, min_periods=1).mean()
    )

    fig = go.Figure()

    for label in ["Normal", "Moderate Risk", "High Risk"]:
        mask = df.get("label", pd.Series(dtype=str)) == label
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df[mask].index,
                y=df[mask]["risk_score"],
                mode="markers",
                name=label,
                marker=dict(color=C[label], size=7, opacity=0.75),
            ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["rolling_avg"],
        mode="lines",
        name=f"Rolling Avg (n={window})",
        line=dict(color=C["blue"], width=2.5),
    ))

    fig.add_hline(
        y=0.70,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text="Alert threshold (0.70)",
        annotation_position="top right",
        annotation_font_color="#e74c3c",
    )

    fig.update_layout(
        **_THEME,
        height=360,
        title="Risk Score Trend",
        xaxis=dict(title="Post Index"),
        yaxis=dict(title="Risk Score", range=[0, 1]),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


# ── 6. Emotion timeline (stacked area) ───────────────────────────────

def emotion_timeline(emotion_data: list) -> go.Figure:
    """
    Stacked area chart for emotions across multiple posts.
    emotion_data: list of {'post_id': int, 'emotions': list[str]}
    """
    tracked = ["Sadness", "Anxiety", "Anger", "Hopelessness", "Loneliness", "Hope"]
    color_map = {
        "Sadness":     "#4a9eda",
        "Anxiety":     "#e07b1a",
        "Anger":       "#c0392b",
        "Hopelessness":"#8e44ad",
        "Loneliness":  "#16a085",
        "Hope":        "#27ae60",
    }

    if not emotion_data:
        return go.Figure()

    records = []
    for item in emotion_data:
        row = {"post_id": item["post_id"]}
        for emo in tracked:
            row[emo] = 1 if emo in item.get("emotions", []) else 0
        records.append(row)

    df = pd.DataFrame(records)

    fig = go.Figure()
    for emo in tracked:
        if emo in df.columns:
            smoothed = df[emo].rolling(3, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df["post_id"],
                y=smoothed,
                mode="lines",
                name=emo,
                stackgroup="one",
                line=dict(color=color_map.get(emo, "#888"), width=1),
            ))

    fig.update_layout(
        **_THEME,
        height=340,
        title="Emotion Timeline Across Posts",
        xaxis=dict(title="Post Index"),
        yaxis=dict(title="Emotion Presence"),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


# ── 7. Risk distribution donut ────────────────────────────────────────

def risk_distribution_pie(counts: dict) -> go.Figure:
    """Donut chart showing the breakdown of risk labels."""
    labels = [k for k, v in counts.items() if v > 0]
    values = [counts[k] for k in labels]
    colors = [C.get(l, "#888") for l in labels]

    if not values:
        return go.Figure()

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.56,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hoverinfo="label+value+percent",
    ))
    fig.update_layout(
        **_THEME,
        height=300,
        showlegend=False,
        title="Risk Label Distribution",
    )
    return fig


# ── 8. Keyword frequency bar ──────────────────────────────────────────

def keyword_frequency_bar(keywords: dict) -> go.Figure:
    """Bar chart showing which risk keywords were found and in which category."""
    rows = []
    for category, kw_list in keywords.items():
        for kw in kw_list:
            rows.append({"keyword": kw, "category": category})

    if not rows:
        fig = go.Figure()
        fig.update_layout(**_THEME, height=120, title="No Risk Keywords Found")
        return fig

    df = pd.DataFrame(rows)
    cat_colors = {
        "high_risk":     C["High Risk"],
        "moderate_risk": C["Moderate Risk"],
        "protective":    C["Normal"],
    }

    fig = go.Figure()
    for cat, grp in df.groupby("category"):
        fig.add_trace(go.Bar(
            x=grp["keyword"],
            y=[1] * len(grp),
            name=cat.replace("_", " ").title(),
            marker_color=cat_colors.get(cat, "#888"),
        ))

    fig.update_layout(
        **_THEME,
        height=260,
        title="Detected Risk Keywords",
        barmode="stack",
        yaxis=dict(showticklabels=False, showgrid=False),
        xaxis=dict(tickangle=-30),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


# ── 9. Sentiment compound history line ───────────────────────────────

def sentiment_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart of VADER compound score over post history."""
    if "compound" not in df.columns:
        return go.Figure()

    df = df.copy().reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["compound"],
        mode="lines+markers",
        name="Compound Score",
        line=dict(color=C["blue"], width=2),
        marker=dict(size=5),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#888",
                  annotation_text="Neutral", annotation_position="right")
    fig.add_hline(y=-0.5, line_dash="dot", line_color=C["High Risk"],
                  annotation_text="Strongly Negative", annotation_position="right",
                  annotation_font_color=C["High Risk"])

    fig.update_layout(
        **_THEME,
        height=280,
        title="Sentiment (Compound Score) Over Time",
        xaxis=dict(title="Post Index"),
        yaxis=dict(title="Compound", range=[-1, 1]),
    )
    return fig
