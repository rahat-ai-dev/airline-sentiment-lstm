"""
Centralized configuration for the Airline Sentiment LSTM project.

All paths, hyperparameters, and constants live here so that no other
module hardcodes a path or a magic number. Every script/notebook should
import from this module instead of redefining these values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & directory layout
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

RAW_DATASET_PATH = RAW_DATA_DIR / "Tweets.csv"
CLEANED_DATASET_PATH = PROCESSED_DATA_DIR / "tweets_clean.parquet"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

VOCAB_PATH = MODELS_DIR / "vocab.json"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_sentiment.keras"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"

for _dir in (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    ARTIFACTS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    SPLITS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data / labels
# ---------------------------------------------------------------------------
TEXT_COLUMN = "text"
LABEL_COLUMN = "airline_sentiment"
CLASS_NAMES = ["negative", "neutral", "positive"]
NUM_CLASSES = len(CLASS_NAMES)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


@dataclass(frozen=True)
class TextConfig:
    """Tokenization / sequence hyperparameters."""

    max_vocab_size: int = 12_000
    max_sequence_length: int = 40
    oov_token: str = "<OOV>"
    pad_token: str = "<PAD>"


@dataclass(frozen=True)
class LSTMConfig:
    """Architecture & optimization hyperparameters for the LSTM model."""

    embedding_dim: int = 128
    lstm_units: int = 96
    dense_units: int = 64
    dropout_rate: float = 0.4
    recurrent_dropout: float = 0.0
    bidirectional: bool = True
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 12
    early_stopping_patience: int = 3
    class_weighting: bool = True


@dataclass(frozen=True)
class BaselineConfig:
    """TF-IDF + Logistic Regression baseline hyperparameters."""

    max_features: int = 20_000
    ngram_range: tuple = (1, 2)
    C: float = 2.0
    max_iter: int = 1000


TEXT_CFG = TextConfig()
LSTM_CFG = LSTMConfig()
BASELINE_CFG = BaselineConfig()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
