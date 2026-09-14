"""Shared evaluation logic for both the baseline and the LSTM model.

Produces the same metric set for each model so they can be compared
head-to-head on identical held-out test data: accuracy, precision, recall,
F1 (macro and per-class), a confusion matrix, and inference latency.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    BASELINE_MODEL_PATH,
    CLASS_NAMES,
    LABEL_COLUMN,
    LSTM_MODEL_PATH,
    METRICS_DIR,
    SPLITS_DIR,
    VOCAB_PATH,
)
from src.features.vocabulary import Vocabulary
from src.utils.logger import get_logger

logger = get_logger(__name__)
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the standard classification metric bundle for one model."""
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    return {
        "accuracy": report["accuracy"],
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"],
            }
            for cls in CLASS_NAMES
        },
        "confusion_matrix": cm.tolist(),
    }


def evaluate_baseline(test_df: pd.DataFrame) -> dict:
    pipeline = joblib.load(BASELINE_MODEL_PATH)

    start = time.time()
    y_pred = pipeline.predict(test_df["clean_text"])
    latency_ms = (time.time() - start) / len(test_df) * 1000

    y_true = test_df[LABEL_COLUMN].map(LABEL_TO_ID).to_numpy()
    y_pred_ids = pd.Series(y_pred).map(LABEL_TO_ID).to_numpy()

    metrics = compute_metrics(y_true, y_pred_ids)
    metrics["avg_inference_latency_ms"] = latency_ms
    metrics["model_name"] = "TF-IDF + Logistic Regression (baseline)"
    return metrics


def evaluate_lstm(test_df: pd.DataFrame) -> dict:
    import tensorflow as tf  # local import: heavy dependency

    model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    vocab = Vocabulary.load(VOCAB_PATH)

    X_test = vocab.encode_batch(test_df["clean_text"].tolist())
    y_true = test_df[LABEL_COLUMN].map(LABEL_TO_ID).to_numpy()

    start = time.time()
    probs = model.predict(X_test, verbose=0)
    latency_ms = (time.time() - start) / len(test_df) * 1000
    y_pred = probs.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred)
    metrics["avg_inference_latency_ms"] = latency_ms
    metrics["model_name"] = "Bidirectional LSTM"
    metrics["num_parameters"] = int(model.count_params())
    return metrics


def run_full_evaluation() -> dict:
    """Evaluate both models on the test split and persist a comparison report."""
    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")

    baseline_metrics = evaluate_baseline(test_df)
    lstm_metrics = evaluate_lstm(test_df)

    comparison = {"baseline": baseline_metrics, "lstm": lstm_metrics}
    out_path = METRICS_DIR / "model_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    logger.info(
        "Baseline acc=%.4f macro-F1=%.4f | LSTM acc=%.4f macro-F1=%.4f",
        baseline_metrics["accuracy"],
        baseline_metrics["macro_f1"],
        lstm_metrics["accuracy"],
        lstm_metrics["macro_f1"],
    )
    return comparison


if __name__ == "__main__":
    run_full_evaluation()
