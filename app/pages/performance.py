from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.resources import load_metrics, load_training_history
from src.config import CLASS_NAMES


def _confusion_matrix_fig(cm, title):
    fig = px.imshow(
        cm,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        color_continuous_scale="Blues",
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def render() -> None:
    st.title("📊 Model Performance")
    metrics = load_metrics()
    baseline, lstm = metrics["baseline"], metrics["lstm"]

    st.markdown(
        "Both models are evaluated on the **same held-out test split** "
        "(never seen during training or vocabulary/vectorizer fitting)."
    )

    cols = st.columns(4)
    rows = [
        ("Accuracy", "accuracy", ".1%"),
        ("Macro Precision", "macro_precision", ".1%"),
        ("Macro Recall", "macro_recall", ".1%"),
        ("Macro F1", "macro_f1", ".2f"),
    ]
    for col, (label, key, fmt) in zip(cols, rows):
        delta = lstm[key] - baseline[key]
        col.metric(
            label,
            format(lstm[key], fmt),
            delta=format(delta, "+.1%") if fmt == ".1%" else format(delta, "+.3f"),
            help=f"Baseline: {format(baseline[key], fmt)}",
        )

    st.caption(
        "Delta shown is **LSTM minus baseline** — a negative delta means the "
        "simpler TF-IDF + Logistic Regression baseline currently wins on that metric."
    )

    st.divider()
    st.subheader("Baseline vs. LSTM — head to head")
    comp_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
            "Baseline (TF-IDF + LogReg)": [baseline[k] for k, _ in zip(
                ["accuracy", "macro_precision", "macro_recall", "macro_f1"], rows)],
            "Bidirectional LSTM": [lstm[k] for k, _ in zip(
                ["accuracy", "macro_precision", "macro_recall", "macro_f1"], rows)],
        }
    )
    fig = go.Figure()
    fig.add_bar(name="Baseline", x=comp_df["Metric"], y=comp_df["Baseline (TF-IDF + LogReg)"])
    fig.add_bar(name="LSTM", x=comp_df["Metric"], y=comp_df["Bidirectional LSTM"])
    fig.update_layout(barmode="group", yaxis_range=[0, 1], height=380,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Why doesn't the LSTM clearly beat the baseline here? (honest discussion)"):
        st.markdown(
            f"""
On this dataset (**~10,000 training tweets** after cleaning, short texts,
~15 words each) the classical **TF-IDF + Logistic Regression** baseline is
competitive with — and on raw accuracy slightly ahead of — the LSTM
({baseline['accuracy']:.1%} vs {lstm['accuracy']:.1%}).

This is a real, honest result, not a bug:

- **Deep sequence models need more data** to out-learn strong n-gram
  features. With only ~10K short, informal training examples, a
  bidirectional LSTM's ~1.6M parameters have very little signal per
  parameter, and validation loss stops improving after 1–2 epochs
  (see the training curves below) — a classic small-data regime.
- **TF-IDF bigrams already capture short-range negation** ("not good",
  "no help") which is most of what drives sentiment in a ~15-word tweet;
  the LSTM's advantage (long-range dependency modeling) matters more for
  longer documents.
- The honest engineering conclusion: for *this* dataset size, the
  baseline is the better production choice on cost/accuracy grounds. The
  LSTM becomes worthwhile with more data, pretrained embeddings
  (GloVe/FastText), or by fine-tuning a pretrained transformer instead of
  training a recurrent model from scratch — see **docs/model_card.md**
  and the "Future Improvements" section of the README for the concrete
  next steps this project would take in a real product setting.
            """
        )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            _confusion_matrix_fig(baseline["confusion_matrix"], "Baseline — Confusion Matrix"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            _confusion_matrix_fig(lstm["confusion_matrix"], "LSTM — Confusion Matrix"),
            use_container_width=True,
        )

    st.divider()
    st.subheader("LSTM training curves")
    history = load_training_history()
    epochs = list(range(1, len(history["loss"]) + 1))
    curve_fig = go.Figure()
    curve_fig.add_scatter(x=epochs, y=history["loss"], name="Train loss")
    curve_fig.add_scatter(x=epochs, y=history["val_loss"], name="Val loss")
    curve_fig.update_layout(height=350, xaxis_title="Epoch", yaxis_title="Loss",
                             margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(curve_fig, use_container_width=True)
    st.caption(
        "Training stopped early (restore_best_weights) once validation loss "
        "stopped improving — visible here as validation loss turning upward "
        "while training loss keeps falling: textbook overfitting on a small dataset."
    )

    st.divider()
    st.subheader("Per-class breakdown")
    per_class_rows = []
    for cls in CLASS_NAMES:
        per_class_rows.append({
            "Class": cls,
            "Baseline Precision": baseline["per_class"][cls]["precision"],
            "Baseline Recall": baseline["per_class"][cls]["recall"],
            "LSTM Precision": lstm["per_class"][cls]["precision"],
            "LSTM Recall": lstm["per_class"][cls]["recall"],
            "Support": int(baseline["per_class"][cls]["support"]),
        })
    st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True, hide_index=True)

    st.subheader("Inference speed & footprint")
    speed_cols = st.columns(2)
    speed_cols[0].metric("Baseline latency / tweet", f"{baseline['avg_inference_latency_ms']:.2f} ms")
    speed_cols[1].metric("LSTM latency / tweet", f"{lstm['avg_inference_latency_ms']:.2f} ms",
                          help=f"{lstm.get('num_parameters', 0):,} trainable parameters")
