import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.models.baseline_model import build_baseline_pipeline
from src.models.lstm_model import build_lstm_model


def test_build_lstm_model_output_shape():
    model = build_lstm_model(vocab_size=500, num_classes=3, max_sequence_length=20)
    dummy_input = np.random.randint(0, 500, size=(4, 20))
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (4, 3)
    # softmax outputs should sum to ~1 per row
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-4)


def test_build_lstm_model_is_compiled():
    model = build_lstm_model(vocab_size=200, num_classes=3, max_sequence_length=10)
    assert model.optimizer is not None
    assert model.loss == "sparse_categorical_crossentropy"


def test_build_baseline_pipeline_fits_and_predicts():
    pipeline = build_baseline_pipeline()
    X = ["great flight crew", "terrible delay", "flight was okay", "loved the service",
         "worst experience ever", "average trip nothing special"]
    y = ["positive", "negative", "neutral", "positive", "negative", "neutral"]
    pipeline.fit(X, y)
    preds = pipeline.predict(["amazing crew"])
    assert preds[0] in {"positive", "negative", "neutral"}
