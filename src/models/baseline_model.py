"""TF-IDF + Logistic Regression baseline.

Every deep-learning portfolio project needs a classical baseline: it is the
only way to demonstrate that the LSTM's added complexity is actually earning
its keep. If the LSTM couldn't beat this baseline, the honest conclusion
would be "use the simpler, cheaper, more interpretable model instead" —
see docs/evaluation.md for that exact comparison on this dataset.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import BASELINE_CFG, RANDOM_SEED


def build_baseline_pipeline() -> Pipeline:
    """Build an (unfit) TF-IDF + Logistic Regression sklearn pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=BASELINE_CFG.max_features,
                    ngram_range=BASELINE_CFG.ngram_range,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=BASELINE_CFG.C,
                    max_iter=BASELINE_CFG.max_iter,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
