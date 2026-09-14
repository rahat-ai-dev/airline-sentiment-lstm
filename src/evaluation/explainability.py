"""Explainability for the LSTM sentiment model.

Grad-CAM is a computer-vision technique (it needs spatial conv feature maps)
and does not apply to a recurrent text classifier, so this module implements
the standard text-model analogue: **occlusion / leave-one-out importance**.

For a given tweet and its predicted class, each token is removed one at a
time and the model is re-scored; the drop in the predicted class's
probability is that token's importance. This is model-agnostic, requires no
architecture changes, and produces a human-readable "what the model focused
on" explanation, mirroring the role Grad-CAM plays for image models.
"""

from __future__ import annotations

import numpy as np

from src.config import CLASS_NAMES
from src.data.text_cleaning import basic_tokenize
from src.features.vocabulary import Vocabulary


def token_importance(
    text: str,
    model,
    vocab: Vocabulary,
) -> dict:
    """Compute per-token importance scores for the model's top prediction.

    Returns
    -------
    dict with keys: ``tokens``, ``importances`` (same length), ``predicted_class``,
    ``confidence``.
    """
    tokens = basic_tokenize(text)
    if not tokens:
        return {"tokens": [], "importances": [], "predicted_class": None, "confidence": 0.0}

    base_seq = vocab.encode_batch([text])
    base_probs = model.predict(base_seq, verbose=0)[0]
    predicted_id = int(np.argmax(base_probs))
    base_score = float(base_probs[predicted_id])

    # Build one variant of the tweet per token, each with that token dropped.
    variants = []
    for i in range(len(tokens)):
        variant_tokens = tokens[:i] + tokens[i + 1 :]
        variants.append(" ".join(variant_tokens) if variant_tokens else vocab.pad_token)

    variant_seqs = vocab.encode_batch(variants)
    variant_probs = model.predict(variant_seqs, verbose=0)[:, predicted_id]

    importances = (base_score - variant_probs).tolist()

    return {
        "tokens": tokens,
        "importances": importances,
        "predicted_class": CLASS_NAMES[predicted_id],
        "confidence": base_score,
    }
