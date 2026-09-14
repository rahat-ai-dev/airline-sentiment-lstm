from __future__ import annotations

import streamlit as st

from app.utils.resources import get_predictor
from app.utils.styling import sentiment_pill

DEFAULT_EXAMPLE = "The gate agent was incredibly rude and my flight was delayed with zero updates."


def render() -> None:
    st.title("🧠 Explainability")
    st.markdown(
        """
        Grad-CAM (the standard explainability technique for image models)
        doesn't apply to a recurrent text model — there's no spatial feature
        map to highlight. Instead, this page uses **occlusion / leave-one-out
        importance**: each word is removed from the tweet one at a time, and
        the model is re-scored. The resulting drop in predicted-class
        probability is that word's contribution — a model-agnostic technique
        that plays the same "what did the model focus on" role Grad-CAM
        plays for vision models.
        """
    )

    text = st.text_area("Tweet to explain", value=DEFAULT_EXAMPLE, height=90)
    run = st.button("Explain prediction", type="primary")

    if run and text.strip():
        predictor = get_predictor()
        with st.spinner("Scoring the tweet once per word…"):
            result = predictor.predict(text, explain=True)

        st.markdown(
            f"**Predicted class:** {sentiment_pill(result.predicted_class)} "
            f"&nbsp; **Confidence:** {result.confidence:.1%}",
            unsafe_allow_html=True,
        )

        if result.tokens:
            max_abs = max(abs(v) for v in result.token_importances) or 1.0
            html_tokens = []
            for tok, imp in zip(result.tokens, result.token_importances):
                intensity = abs(imp) / max_abs
                color = f"rgba(220,38,38,{intensity:.2f})" if imp > 0 else f"rgba(37,99,235,{intensity:.2f})"
                html_tokens.append(
                    f'<span style="background:{color}; padding:3px 6px; '
                    f'border-radius:5px; margin:2px; display:inline-block; font-size:1.05rem;">{tok}</span>'
                )
            st.markdown(" ".join(html_tokens), unsafe_allow_html=True)
            st.caption(
                "🔴 Darker red = removing this word made the model *more* confident "
                "(the word was suppressing the prediction, or context shifted) — "
                "🔵 darker blue = removing this word made the model *less* confident "
                "in its predicted class (the word was supporting it)."
            )

            st.divider()
            st.subheader("Ranked word importance")
            ranked = sorted(
                zip(result.tokens, result.token_importances), key=lambda kv: -abs(kv[1])
            )
            st.dataframe(
                {
                    "Word": [w for w, _ in ranked],
                    "Importance": [round(v, 4) for _, v in ranked],
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("Couldn't tokenize this input — try a longer sentence.")

    st.divider()
    with st.expander("Model card summary"):
        st.markdown(
            """
            - **Intended use:** triage/prioritization aid for customer-support
              teams monitoring airline mentions; NOT a fully automated
              decision system.
            - **Known limitation:** trained on 2015 US-airline Twitter data —
              slang, airline names, and events from that period; performance
              on other domains (reviews, other industries, other eras of
              internet slang) is not validated.
            - **Fairness note:** the dataset does not include demographic
              attributes, but airline names are removed from `clean_text`
              before modeling to reduce the risk of the model learning
              "which airline is being discussed" as a proxy signal.
            - Full details in `docs/model_card.md`.
            """
        )
