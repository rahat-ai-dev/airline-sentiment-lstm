"""Lightweight data-validation checks run before any preprocessing.

The goal is to fail loudly and early if the raw CSV does not look like the
dataset the rest of the pipeline was built for, rather than silently
producing garbage features downstream.
"""

from __future__ import annotations

import pandas as pd

from src.config import CLASS_NAMES, LABEL_COLUMN, TEXT_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {
    "tweet_id",
    TEXT_COLUMN,
    LABEL_COLUMN,
    "airline",
    "negativereason",
    "retweet_count",
    "tweet_created",
}


class DataValidationError(ValueError):
    """Raised when the raw dataset fails a structural or content check."""


def validate_raw_dataframe(df: pd.DataFrame) -> None:
    """Run a battery of sanity checks on the freshly-loaded raw dataset.

    Raises
    ------
    DataValidationError
        If any check fails, with a message describing exactly what's wrong.
    """
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise DataValidationError(f"Missing expected columns: {sorted(missing_cols)}")

    if df.empty:
        raise DataValidationError("Raw dataset is empty.")

    unexpected_labels = set(df[LABEL_COLUMN].dropna().unique()) - set(CLASS_NAMES)
    if unexpected_labels:
        raise DataValidationError(
            f"Found labels outside the expected set {CLASS_NAMES}: {unexpected_labels}"
        )

    empty_text_ratio = (df[TEXT_COLUMN].astype(str).str.strip() == "").mean()
    if empty_text_ratio > 0.01:
        raise DataValidationError(
            f"{empty_text_ratio:.1%} of rows have empty text — dataset looks corrupted."
        )

    duplicate_ratio = df.duplicated(subset=["tweet_id"]).mean()
    if duplicate_ratio > 0.05:
        logger.warning("%.1f%% duplicate tweet_ids found in raw data.", duplicate_ratio * 100)

    logger.info(
        "Validation passed: %d rows, %d columns, %d unique labels.",
        len(df),
        len(df.columns),
        df[LABEL_COLUMN].nunique(),
    )
