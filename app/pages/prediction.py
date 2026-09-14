from __future__ import annotations

import streamlit as st

from app.utils.resources import get_predictor
from app.utils.styling import sentiment_pill

EXAMPLES = [
    "Thank you so much for getting me on an earlier flight, you saved my trip!",
    "My flight has been delayed 6 hours and nobody at the gate will explain why.",
    "Flight 2214 departed on time from gate B12.",
]


def render() -> None:
    st.title("🔮 Predict Sentiment")
    st.caption("Type or paste customer feedback below — the model runs entirely on-device.")

    example = st.selectbox("Try an example, or write your own below:", ["(write my own)"] + EXAMPLES)
    default_text = "" if example == "(write my own)" else example

    text = st.text_area("Customer message", value=default_text, height=110,
                         placeholder="e.g. @united my bag never arrived and no one answers the phone")

    col_a, col_b = st.columns([1, 5])
    run = col_a.button("Analyze", type="primary", use_container_width=True)

    if run and text.strip():
        predictor = get_predictor()
        with st.spinner("Running inference…"):
            result = predictor.predict(text)

        st.markdown("### Result")
        st.markdown(sentiment_pill(result.predicted_class), unsafe_allow_html=True)

        cols = st.columns(3)
        cols[0].metric("Confidence", f"{result.confidence:.1%}")
        cols[1].metric("Inference time", f"{result.inference_time_ms:.1f} ms")
        cols[2].metric("Model", "Bidirectional LSTM")

        st.markdown("#### Class probabilities")
        for cls, prob in sorted(result.class_probabilities.items(), key=lambda kv: -kv[1]):
            st.progress(prob, text=f"{cls.capitalize()} — {prob:.1%}")

        st.markdown("#### Top predictions")
        st.table(
            {
                "Class": [c for c, _ in result.top_predictions],
                "Probability": [f"{p:.1%}" for _, p in result.top_predictions],
            }
        )

        with st.expander("See word-level explanation for this prediction"):
            st.caption(
                "Computed on demand via occlusion analysis — go to the "
                "**Explainability** page for a full walkthrough of the method."
            )
            explained = predictor.predict(text, explain=True)
            if explained.tokens:
                max_abs = max(abs(v) for v in explained.token_importances) or 1.0
                html_tokens = []
                for tok, imp in zip(explained.tokens, explained.token_importances):
                    intensity = abs(imp) / max_abs
                    color = f"rgba(220,38,38,{intensity:.2f})" if imp > 0 else f"rgba(37,99,235,{intensity:.2f})"
                    html_tokens.append(
                        f'<span style="background:{color}; padding:2px 4px; '
                        f'border-radius:4px; margin:1px; display:inline-block;">{tok}</span>'
                    )
                st.markdown(" ".join(html_tokens), unsafe_allow_html=True)
                st.caption("🔴 Red = pushed toward the predicted class · 🔵 Blue = pushed away from it")
    elif run:
        st.warning("Please enter some text first.")
