from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.utils.resources import load_split
from src.config import LABEL_COLUMN


def render() -> None:
    st.title("🔍 Dataset Explorer")

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    import pandas as pd
    all_df = pd.concat(
        [train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")],
        ignore_index=True,
    )

    st.markdown(
        "**Twitter US Airline Sentiment** — real tweets scraped in February 2015, "
        "labeled by human annotators (Crowdflower / Figure Eight). "
        "See `DATASET.md` for full provenance and license details."
    )

    cols = st.columns(4)
    cols[0].metric("Total tweets (cleaned)", f"{len(all_df):,}")
    cols[1].metric("Train / Val / Test", f"{len(train_df)} / {len(val_df)} / {len(test_df)}")
    cols[2].metric("Airlines covered", all_df["airline"].nunique())
    cols[3].metric("Avg. tweet length", f"{all_df['text_length'].mean():.1f} words")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class distribution (train split)")
        dist = train_df[LABEL_COLUMN].value_counts().reset_index()
        dist.columns = ["sentiment", "count"]
        fig = px.bar(dist, x="sentiment", y="count", color="sentiment",
                     color_discrete_map={"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"})
        fig.update_layout(showlegend=False, height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Notably imbalanced: negative tweets make up "
            f"{(train_df[LABEL_COLUMN] == 'negative').mean():.0%} of the training data — "
            "handled during training via class weighting (see `src/training/train_lstm.py`)."
        )

    with col2:
        st.subheader("Sentiment by airline")
        by_airline = (
            all_df.groupby(["airline", LABEL_COLUMN]).size().reset_index(name="count")
        )
        fig2 = px.bar(
            by_airline, x="airline", y="count", color=LABEL_COLUMN, barmode="stack",
            color_discrete_map={"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"},
        )
        fig2.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Top negative-feedback reasons")
    reasons = (
        all_df[all_df[LABEL_COLUMN] == "negative"]["negativereason"]
        .dropna()
        .value_counts()
        .head(10)
        .reset_index()
    )
    reasons.columns = ["reason", "count"]
    fig3 = px.bar(reasons, x="count", y="reason", orientation="h")
    fig3.update_layout(height=380, yaxis={"categoryorder": "total ascending"},
                        margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("Browse sample tweets")
    sentiment_filter = st.multiselect(
        "Filter by sentiment", options=["negative", "neutral", "positive"],
        default=["negative", "neutral", "positive"],
    )
    filtered = all_df[all_df[LABEL_COLUMN].isin(sentiment_filter)]
    st.dataframe(
        filtered[["airline", LABEL_COLUMN, "text", "negativereason"]].sample(
            min(15, len(filtered)), random_state=7
        ),
        use_container_width=True,
        hide_index=True,
    )
