from __future__ import annotations

import streamlit as st

from app.utils.resources import load_metrics, load_split


def render() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>✈️ Airline Sentiment Intelligence</h1>
            <p>Deep learning for real-time customer feedback triage — a bidirectional
            LSTM reads raw customer tweets and classifies them as
            <b>negative</b>, <b>neutral</b>, or <b>positive</b>, so support teams
            can prioritize the complaints that matter most, in seconds instead
            of hours of manual review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    train_df = load_split("train")
    metrics = load_metrics()

    cols = st.columns(4)
    stats = [
        ("Training tweets", f"{len(train_df):,}"),
        ("Model", "Bidirectional LSTM"),
        ("Test accuracy", f"{metrics['lstm']['accuracy']:.1%}"),
        ("Macro F1", f"{metrics['lstm']['macro_f1']:.2f}"),
    ]
    for col, (label, value) in zip(cols, stats):
        col.markdown(
            f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Why this problem needs deep learning, not keyword rules")
    st.markdown(
        """
        Airlines receive thousands of public mentions a day. A naive keyword
        filter ("delay", "cancelled", "rude") misses negation ("not bad at
        all"), sarcasm-adjacent phrasing, and context that spans the whole
        sentence. A sequence model reads the tweet as an ordered sequence of
        words instead of an unordered bag, which is exactly what negation and
        clause structure require.

        This app is built on the **real, publicly released Twitter US Airline
        Sentiment dataset** (Crowdflower / Figure Eight, February 2015,
        14,640 tweets covering 6 major US carriers) — see the **Dataset
        Explorer** page for a full breakdown, and **Model Performance** for a
        transparent, head-to-head comparison against a classical TF-IDF +
        Logistic Regression baseline (the LSTM does *not* unconditionally win
        — see why on that page).
        """
    )

    st.info(
        "👉 Head to **Predict** in the sidebar to try the model on your own text, "
        "or **Explainability** to see exactly which words drove a prediction.",
        icon="💡",
    )
