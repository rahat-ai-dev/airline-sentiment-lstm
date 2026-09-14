"""Train the bidirectional LSTM sentiment classifier.

Usage
-----
    python -m src.training.train_lstm
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    CLASS_NAMES,
    LABEL_COLUMN,
    LSTM_CFG,
    LSTM_MODEL_PATH,
    METRICS_DIR,
    RANDOM_SEED,
    SPLITS_DIR,
    VOCAB_PATH,
)
from src.features.vocabulary import Vocabulary
from src.utils.logger import get_logger

logger = get_logger(__name__)

LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val.parquet")
    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")
    return train_df, val_df, test_df


def train_lstm() -> dict:
    """Fit the vocabulary + LSTM on the training split and persist artifacts."""
    set_global_seed()
    train_df, val_df, test_df = load_splits()

    logger.info("Fitting vocabulary on training split only (%d rows).", len(train_df))
    vocab = Vocabulary().fit(train_df["clean_text"].tolist())
    vocab.save(VOCAB_PATH)
    logger.info("Vocabulary size: %d tokens.", vocab.vocab_size)

    X_train = vocab.encode_batch(train_df["clean_text"].tolist())
    X_val = vocab.encode_batch(val_df["clean_text"].tolist())
    X_test = vocab.encode_batch(test_df["clean_text"].tolist())

    y_train = train_df[LABEL_COLUMN].map(LABEL_TO_ID).to_numpy()
    y_val = val_df[LABEL_COLUMN].map(LABEL_TO_ID).to_numpy()
    y_test = test_df[LABEL_COLUMN].map(LABEL_TO_ID).to_numpy()

    class_weight = None
    if LSTM_CFG.class_weighting:
        weights = compute_class_weight(
            class_weight="balanced", classes=np.arange(len(CLASS_NAMES)), y=y_train
        )
        class_weight = dict(enumerate(weights))
        logger.info("Using class weights to counter imbalance: %s", class_weight)

    from src.models.lstm_model import build_lstm_model  # local import: needs TF

    model = build_lstm_model(vocab_size=vocab.vocab_size)
    model.summary(print_fn=logger.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=LSTM_CFG.early_stopping_patience,
            restore_best_weights=True,
        )
    ]

    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_CFG.epochs,
        batch_size=LSTM_CFG.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.time() - start
    logger.info("Training finished in %.1fs (%d epochs run).", train_seconds, len(history.history["loss"]))

    LSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(LSTM_MODEL_PATH)
    logger.info("Saved LSTM model to %s", LSTM_MODEL_PATH)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test accuracy: %.4f | Test loss: %.4f", test_acc, test_loss)

    history_path = METRICS_DIR / "lstm_training_history.json"
    history_path.write_text(json.dumps(history.history, indent=2))

    summary = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "train_seconds": train_seconds,
        "epochs_run": len(history.history["loss"]),
        "vocab_size": vocab.vocab_size,
        "trainable_params": int(
            sum(tf.size(w).numpy() for w in model.trainable_weights)
        ),
    }
    (METRICS_DIR / "lstm_train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train_lstm()
