"""Deterministic text-cleaning utilities.

These functions are shared between the offline preprocessing pipeline
(``src/data/preprocessing.py``) and the online inference path
(``src/inference/predict.py``) so that training-serving skew cannot creep in:
whatever transformation a tweet receives before training is exactly what a
user-submitted sentence receives before prediction.
"""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_SYMBOL_RE = re.compile(r"#")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z0-9'\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")


def clean_tweet(text: str) -> str:
    """Normalize a raw tweet into clean, lowercase text.

    Steps (in order):
    1. Lowercase.
    2. Strip URLs.
    3. Strip @mentions (the airline handle carries no sentiment signal and
       was shown in prior analyses of this dataset to bias models toward
       predicting on the airline name rather than the complaint itself).
    4. Drop the ``#`` symbol but keep the hashtag word, since hashtag text
       ("#fail", "#neveragain") is often sentiment-bearing.
    5. Collapse elongated characters ("soooo good" -> "soo good") so the
       vocabulary doesn't fragment across every spelling variant.
    6. Remove punctuation/emoji/non-alphanumeric characters.
    7. Collapse repeated whitespace and strip.

    Parameters
    ----------
    text: raw tweet text.

    Returns
    -------
    Cleaned text, safe to tokenize.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_SYMBOL_RE.sub("", text)
    text = _REPEATED_CHARS_RE.sub(r"\1\1", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def basic_tokenize(text: str) -> list[str]:
    """Whitespace tokenizer applied after :func:`clean_tweet`."""
    return clean_tweet(text).split()
