"""Cached, expensive resources shared across pages.

Using st.cache_resource for the model/predictor (loaded once per process,
not per interaction) and st.cache_data for dataframes/JSON (re-read only if
the underlying file changes), per the project's performance requirements.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import METRICS_DIR, SPLITS_DIR
from src.inference.predict import SentimentPredictor


@st.cache_resource(show_spinner="Loading trained LSTM model…")
def get_predictor() -> SentimentPredictor:
    return SentimentPredictor()


@st.cache_data(show_spinner=False)
def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS_DIR / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = METRICS_DIR / "model_comparison.json"
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_training_history() -> dict:
    path = METRICS_DIR / "lstm_training_history.json"
    return json.loads(path.read_text())
