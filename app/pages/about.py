from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("ℹ️ About this project")
    st.markdown(
        """
### Airline Sentiment Intelligence

A production-shaped deep learning project demonstrating an end-to-end NLP
pipeline: real raw data → validation → cleaning → a bidirectional LSTM
sequence model (compared honestly against a classical baseline) →
explainability → a deployable Streamlit application.

**Pipeline**

`data/raw/Tweets.csv` → `src/data/preprocessing.py` → `src/features/vocabulary.py`
→ `src/models/lstm_model.py` + `src/training/train_lstm.py` →
`src/evaluation/evaluate.py` → `src/inference/predict.py` → this app.

**Dataset:** Twitter US Airline Sentiment (Crowdflower / Figure Eight, Feb 2015,
14,640 tweets). See `DATASET.md` for full provenance, license, and known
limitations/biases.

**Stack:** TensorFlow/Keras · scikit-learn · pandas · Streamlit · Plotly

**Repository layout, training commands, and reproduction steps:** see the
project `README.md`.

---
*This tool is a decision-support aid for customer-support triage, not an
automated moderation or decision system. Predictions should be reviewed by a
human before any customer-facing action is taken.*
        """
    )
