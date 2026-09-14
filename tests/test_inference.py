"""Inference-path tests.

Rather than depending on the full ~6MB trained artifact (which may not be
present in a fresh checkout before `make train` has run — see README), these
tests build a tiny model + vocab on the fly, save them to a temp dir, and
verify the SentimentPredictor plumbing (loading, cleaning, batching,
top-k ordering, explanation shape) end-to-end.
"""

import pytest

tf = pytest.importorskip("tensorflow")

from src.evaluation.explainability import token_importance
from src.features.vocabulary import Vocabulary
from src.inference.predict import SentimentPredictor
from src.models.lstm_model import build_lstm_model


@pytest.fixture()
def tiny_predictor(tmp_path):
    texts = [
        "great flight thank you",
        "terrible delay no help",
        "flight left on time",
    ] * 5
    vocab = Vocabulary(max_vocab_size=100, max_sequence_length=10).fit(texts)

    model = build_lstm_model(vocab_size=vocab.vocab_size, num_classes=3, max_sequence_length=10)

    model_path = tmp_path / "tiny_model.keras"
    vocab_path = tmp_path / "vocab.json"
    model.save(model_path)
    vocab.save(vocab_path)

    return SentimentPredictor(model_path=model_path, vocab_path=vocab_path)


def test_predict_returns_valid_probability_distribution(tiny_predictor):
    result = tiny_predictor.predict("thank you for the great flight")
    total_prob = sum(result.class_probabilities.values())
    assert 0.99 <= total_prob <= 1.01
    assert result.predicted_class in {"negative", "neutral", "positive"}


def test_predict_top_predictions_sorted_descending(tiny_predictor):
    result = tiny_predictor.predict("no help at all terrible")
    probs = [p for _, p in result.top_predictions]
    assert probs == sorted(probs, reverse=True)


def test_predict_batch_returns_one_result_per_input(tiny_predictor):
    results = tiny_predictor.predict_batch(["good", "bad", "neutral flight"])
    assert len(results) == 3


def test_explain_flag_populates_tokens_and_importances(tiny_predictor):
    result = tiny_predictor.predict("terrible delay no help", explain=True)
    assert len(result.tokens) == len(result.token_importances)
    assert len(result.tokens) > 0


def test_token_importance_handles_empty_text(tiny_predictor):
    explanation = token_importance("", tiny_predictor.model, tiny_predictor.vocab)
    assert explanation["tokens"] == []
    assert explanation["predicted_class"] is None
