"""Streamlit entry point for the Airline Sentiment Intelligence app.

Run with:
    streamlit run app/streamlit_app.py

The UI is intentionally a thin layer: every heavy computation (model
loading, inference, metric computation) lives in ``src/`` and is imported
here, never re-implemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `src` importable when Streamlit runs this file directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.pages import (  # noqa: E402
    about,
    dataset_explorer,
    explainability,
    home,
    performance,
    prediction,
)
from app.utils.styling import inject_global_css  # noqa: E402

st.set_page_config(
    page_title="Airline Sentiment Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

PAGES = {
    "Home": home,
    "Predict": prediction,
    "Model Performance": performance,
    "Dataset Explorer": dataset_explorer,
    "Explainability": explainability,
    "About": about,
}

with st.sidebar:
    st.markdown("### ✈️ Airline Sentiment\n**Intelligence Platform**")
    st.caption("Bidirectional LSTM · Deep Learning · Explainable AI")
    choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption(
        "Built on the real, publicly released Twitter US Airline Sentiment "
        "dataset (14,640 tweets, Feb 2015)."
    )

PAGES[choice].render()
