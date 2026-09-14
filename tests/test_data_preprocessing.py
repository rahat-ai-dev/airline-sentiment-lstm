import pandas as pd
import pytest

from src.data.preprocessing import clean_dataset, stratified_split
from src.data.validation import DataValidationError, validate_raw_dataframe


def _make_valid_df(n_per_class: int = 20) -> pd.DataFrame:
    rows = []
    tid = 0
    for label in ["negative", "neutral", "positive"]:
        for i in range(n_per_class):
            tid += 1
            rows.append(
                {
                    "tweet_id": tid,
                    "text": f"@united this is example {label} tweet number {i}",
                    "airline_sentiment": label,
                    "airline": "United",
                    "negativereason": "Customer Service Issue" if label == "negative" else None,
                    "retweet_count": 0,
                    "tweet_created": "2015-02-24 11:35:52 -0800",
                }
            )
    return pd.DataFrame(rows)


def test_validate_raw_dataframe_accepts_well_formed_data():
    df = _make_valid_df()
    validate_raw_dataframe(df)  # should not raise


def test_validate_raw_dataframe_rejects_missing_columns():
    df = _make_valid_df().drop(columns=["airline"])
    with pytest.raises(DataValidationError):
        validate_raw_dataframe(df)


def test_validate_raw_dataframe_rejects_unknown_labels():
    df = _make_valid_df()
    df.loc[0, "airline_sentiment"] = "very_negative"
    with pytest.raises(DataValidationError):
        validate_raw_dataframe(df)


def test_validate_raw_dataframe_rejects_mostly_empty_text():
    df = _make_valid_df()
    df["text"] = ""
    with pytest.raises(DataValidationError):
        validate_raw_dataframe(df)


def test_clean_dataset_deduplicates_and_adds_columns():
    df = _make_valid_df(n_per_class=5)
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # inject one dup
    cleaned = clean_dataset(duplicated)
    assert "clean_text" in cleaned.columns
    assert "text_length" in cleaned.columns
    assert len(cleaned) == len(df)  # duplicate removed


def test_stratified_split_preserves_class_balance():
    df = clean_dataset(_make_valid_df(n_per_class=60))
    train, val, test = stratified_split(df, val_frac=0.2, test_frac=0.2, seed=0)

    assert len(train) + len(val) + len(test) == len(df)
    for split in (train, val, test):
        counts = split["airline_sentiment"].value_counts(normalize=True)
        # each class should be roughly a third, since input is balanced
        for frac in counts:
            assert 0.2 < frac < 0.45
