"""Shared CSS injection so the app doesn't look like a default Streamlit demo."""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.5rem 2.5rem;
        border-radius: 16px;
        color: #f5f7fa;
        margin-bottom: 1.5rem;
    }
    .hero h1 { font-size: 2.1rem; margin-bottom: 0.4rem; font-weight: 700; }
    .hero p { font-size: 1.05rem; color: #d7e1e8; max-width: 720px; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #eaecef;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-card .label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase;
        letter-spacing: 0.04em; }
    .metric-card .value { font-size: 1.7rem; font-weight: 700; color: #111827; }

    .pill {
        display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; margin-right: 0.4rem;
    }
    .pill-negative { background: #fde2e1; color: #b3261e; }
    .pill-neutral { background: #e8eaed; color: #3c4043; }
    .pill-positive { background: #e0f5e6; color: #1e7d34; }

    div[data-testid="stSidebar"] { background: #0f1620; }
    div[data-testid="stSidebar"] * { color: #e5e9ef !important; }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.35rem 0.5rem; border-radius: 8px;
    }
</style>
"""


def inject_global_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def sentiment_pill(label: str) -> str:
    css_class = {"negative": "pill-negative", "neutral": "pill-neutral", "positive": "pill-positive"}[label]
    return f'<span class="pill {css_class}">{label.upper()}</span>'
