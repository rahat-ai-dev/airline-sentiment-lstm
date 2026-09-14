"""Bidirectional LSTM architecture for 3-class tweet sentiment classification.

Why an LSTM for this problem: tweet sentiment is driven by word *order* and
long-range interactions ("not... good" vs "good") that bag-of-words baselines
(TF-IDF + Logistic Regression) cannot represent. A recurrent sequence model
reads the tweet left-to-right (and right-to-left, since we use a
bidirectional layer) and builds a representation that is sensitive to
negation, intensifiers, and clause structure — which is exactly where the
TF-IDF baseline is expected to fail (see notebooks/02_modeling_and_evaluation
for the head-to-head comparison and concrete error examples).
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from src.config import LSTM_CFG, NUM_CLASSES, TEXT_CFG


def build_lstm_model(
    vocab_size: int,
    num_classes: int = NUM_CLASSES,
    max_sequence_length: int = TEXT_CFG.max_sequence_length,
) -> keras.Model:
    """Build and compile the bidirectional LSTM sentiment classifier.

    Architecture
    ------------
    Input (padded token ids)
        -> Embedding (learned, masks padding)
        -> Bidirectional LSTM
        -> Dropout
        -> Dense (ReLU)
        -> Dropout
        -> Dense (softmax over 3 classes)
    """
    inputs = keras.Input(shape=(max_sequence_length,), dtype="int32", name="token_ids")

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=LSTM_CFG.embedding_dim,
        mask_zero=True,
        name="embedding",
    )(inputs)

    lstm_layer = layers.LSTM(
        LSTM_CFG.lstm_units,
        dropout=LSTM_CFG.dropout_rate,
        recurrent_dropout=LSTM_CFG.recurrent_dropout,
        name="lstm",
    )
    if LSTM_CFG.bidirectional:
        x = layers.Bidirectional(lstm_layer, name="bidirectional_lstm")(x)
    else:
        x = lstm_layer(x)

    x = layers.Dropout(LSTM_CFG.dropout_rate, name="dropout_1")(x)
    x = layers.Dense(LSTM_CFG.dense_units, activation="relu", name="dense_1")(x)
    x = layers.Dropout(LSTM_CFG.dropout_rate / 2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="sentiment_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="airline_sentiment_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LSTM_CFG.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
