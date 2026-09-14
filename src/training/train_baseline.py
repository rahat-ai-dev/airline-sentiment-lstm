"""Train the TF-IDF + Logistic Regression baseline model.

Usage
-----
    python -m src.training.train_baseline
"""

from __future__ import annotations

import json
import time

import joblib
import pandas as pd

from src.config import BASELINE_MODEL_PATH, LABEL_COLUMN, METRICS_DIR, SPLITS_DIR
from src.models.baseline_model import build_baseline_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_baseline() -> dict:
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")

    pipeline = build_baseline_pipeline()

    start = time.time()
    pipeline.fit(train_df["clean_text"], train_df[LABEL_COLUMN])
    train_seconds = time.time() - start

    test_accuracy = pipeline.score(test_df["clean_text"], test_df[LABEL_COLUMN])
    logger.info("Baseline test accuracy: %.4f (trained in %.1fs)", test_accuracy, train_seconds)

    BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, BASELINE_MODEL_PATH)
    logger.info("Saved baseline model to %s", BASELINE_MODEL_PATH)

    summary = {"test_accuracy": float(test_accuracy), "train_seconds": train_seconds}
    (METRICS_DIR / "baseline_train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train_baseline()
