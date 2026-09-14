"""A small, dependency-free vocabulary + sequence encoder.

Deliberately hand-rolled instead of relying on ``tf.keras.preprocessing.text``
(deprecated) or an external tokenizer library: for a dataset this size a
transparent word-frequency vocabulary is easy to audit, easy to serialize to
plain JSON (so training and serving can never drift apart), and keeps the
project's dependency surface small.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.config import TEXT_CFG
from src.data.text_cleaning import basic_tokenize


class Vocabulary:
    """Maps tokens <-> integer ids and encodes text into padded sequences."""

    def __init__(
        self,
        max_vocab_size: int = TEXT_CFG.max_vocab_size,
        max_sequence_length: int = TEXT_CFG.max_sequence_length,
        pad_token: str = TEXT_CFG.pad_token,
        oov_token: str = TEXT_CFG.oov_token,
    ) -> None:
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.pad_token = pad_token
        self.oov_token = oov_token
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def oov_id(self) -> int:
        return self.token_to_id[self.oov_token]

    def fit(self, texts: list[str]) -> "Vocabulary":
        """Build the vocabulary from a list of (already-cleaned) texts.

        The vocabulary is fit ONLY on the training split — never on
        validation/test text — to avoid leaking distributional information
        about held-out data into the model's input representation.
        """
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(basic_tokenize(text))

        # Reserve ids 0 and 1 for PAD and OOV.
        self.token_to_id = {self.pad_token: 0, self.oov_token: 1}
        most_common = counter.most_common(self.max_vocab_size - 2)
        for token, _freq in most_common:
            self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        return self

    def encode(self, text: str) -> list[int]:
        """Tokenize + map to ids (unknown tokens map to OOV id)."""
        tokens = basic_tokenize(text)
        return [self.token_to_id.get(tok, self.oov_id) for tok in tokens]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode and pad/truncate a batch of texts to a fixed-length matrix."""
        sequences = [self.encode(t) for t in texts]
        return pad_sequences(
            sequences, max_len=self.max_sequence_length, pad_value=self.pad_id
        )

    def save(self, path: Path) -> None:
        payload = {
            "max_vocab_size": self.max_vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "pad_token": self.pad_token,
            "oov_token": self.oov_token,
            "token_to_id": self.token_to_id,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        payload = json.loads(Path(path).read_text())
        vocab = cls(
            max_vocab_size=payload["max_vocab_size"],
            max_sequence_length=payload["max_sequence_length"],
            pad_token=payload["pad_token"],
            oov_token=payload["oov_token"],
        )
        vocab.token_to_id = payload["token_to_id"]
        vocab.id_to_token = {int(v): k for k, v in vocab.token_to_id.items()}
        return vocab


def pad_sequences(sequences: list[list[int]], max_len: int, pad_value: int = 0) -> np.ndarray:
    """Pad (post) or truncate (post) integer sequences to a fixed length."""
    out = np.full((len(sequences), max_len), pad_value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        trimmed = seq[:max_len]
        out[i, : len(trimmed)] = trimmed
    return out
