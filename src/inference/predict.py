"""Inference service for the airline sentiment LSTM.

This is the single reusable prediction entry point. The Streamlit app calls
into this module rather than touching TensorFlow or the vocabulary directly,
so the UI layer stays a thin presentation layer and the same service can
later be dropped behind a REST endpoint (FastAPI, etc.) with no UI code
changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.config import CLASS_NAMES, LSTM_MODEL_PATH, VOCAB_PATH
from src.data.text_cleaning import clean_tweet
from src.evaluation.explainability import token_importance
from src.features.vocabulary import Vocabulary
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    text: str
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    top_predictions: list[tuple[str, float]]
    inference_time_ms: float
    tokens: list[str] = field(default_factory=list)
    token_importances: list[float] = field(default_factory=list)


class SentimentPredictor:
    """Loads the trained LSTM + vocabulary once and serves predictions."""

    def __init__(self, model_path=LSTM_MODEL_PATH, vocab_path=VOCAB_PATH) -> None:
        import tensorflow as tf  # local import: keep module import light for tests

        logger.info("Loading LSTM model from %s", model_path)
        self.model = tf.keras.models.load_model(model_path)
        self.vocab = Vocabulary.load(vocab_path)
        logger.info("Predictor ready (vocab size=%d).", self.vocab.vocab_size)

    def predict(self, text: str, explain: bool = False) -> PredictionResult:
        """Predict sentiment for a single piece of text.

        Parameters
        ----------
        text: raw, unprocessed user input.
        explain: if True, also compute per-token importance (slower — runs
            one forward pass per token of the tweet).
        """
        start = time.time()
        cleaned = clean_tweet(text)
        seq = self.vocab.encode_batch([cleaned])
        probs = self.model.predict(seq, verbose=0)[0]
        elapsed_ms = (time.time() - start) * 1000

        order = np.argsort(probs)[::-1]
        top_predictions = [(CLASS_NAMES[i], float(probs[i])) for i in order]
        predicted_class = top_predictions[0][0]
        confidence = top_predictions[0][1]

        tokens: list[str] = []
        importances: list[float] = []
        if explain and cleaned.strip():
            explanation = token_importance(cleaned, self.model, self.vocab)
            tokens = explanation["tokens"]
            importances = explanation["importances"]

        return PredictionResult(
            text=text,
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities={cls: float(p) for cls, p in zip(CLASS_NAMES, probs)},
            top_predictions=top_predictions,
            inference_time_ms=elapsed_ms,
            tokens=tokens,
            token_importances=importances,
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        return [self.predict(t) for t in texts]
