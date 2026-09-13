"""End-to-end offline preprocessing pipeline.

Raw CSV (data/raw/Tweets.csv)
    -> validate
    -> clean text
    -> drop empty/duplicate rows
    -> stratified train/val/test split
    -> persist to data/processed/

This module is imported by notebooks for interactive exploration and is
also runnable as a script for a fully reproducible, one-command pipeline
run (see scripts/prepare_data.py).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LABEL_COLUMN,
    RANDOM_SEED,
    RAW_DATASET_PATH,
    SPLITS_DIR,
    TEST_FRAC,
    TEXT_COLUMN,
    VAL_FRAC,
)
from src.data.text_cleaning import clean_tweet
from src.data.validation import validate_raw_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_dataset(path=RAW_DATASET_PATH) -> pd.DataFrame:
    """Load the raw Twitter US Airline Sentiment CSV."""
    logger.info("Loading raw dataset from %s", path)
    df = pd.read_csv(path)
    validate_raw_dataframe(df)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning and drop rows that become unusable."""
    df = df.copy()
    df["clean_text"] = df[TEXT_COLUMN].apply(clean_tweet)

    before = len(df)
    df = df[df["clean_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["clean_text", LABEL_COLUMN])
    after = len(df)
    logger.info("Cleaning removed %d rows (%d -> %d).", before - after, before, after)

    df["text_length"] = df["clean_text"].str.split().apply(len)
    return df.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split preserving class balance in every split.

    Splitting happens *after* deduplication and *before* any vocabulary or
    vectorizer is fit, so no information from validation/test tweets ever
    leaks into training-time statistics (a common source of inflated
    accuracy in text-classification portfolio projects).
    """
    train_val, test = train_test_split(
        df,
        test_size=test_frac,
        stratify=df[LABEL_COLUMN],
        random_state=seed,
    )
    relative_val = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[LABEL_COLUMN],
        random_state=seed,
    )
    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d", len(train), len(val), len(test)
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def run_preprocessing_pipeline(save: bool = True) -> dict[str, pd.DataFrame]:
    """Run the full pipeline and optionally persist the splits to disk."""
    raw_df = load_raw_dataset()
    clean_df = clean_dataset(raw_df)
    train_df, val_df, test_df = stratified_split(clean_df)

    splits = {"train": train_df, "val": val_df, "test": test_df}

    if save:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        for name, split_df in splits.items():
            out_path = SPLITS_DIR / f"{name}.parquet"
            split_df.to_parquet(out_path, index=False)
            logger.info("Saved %s split (%d rows) to %s", name, len(split_df), out_path)

    return splits


if __name__ == "__main__":
    run_preprocessing_pipeline()
