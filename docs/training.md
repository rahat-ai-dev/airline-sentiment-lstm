# Training

## Reproducibility

- Global random seed: `42` (`src.config.RANDOM_SEED`), applied to `numpy`, `tensorflow`, and every `sklearn` split/estimator that accepts a `random_state`.
- All hyperparameters are declared as frozen dataclasses in `src/config.py` (`TextConfig`, `LSTMConfig`, `BaselineConfig`) — no magic numbers in the training scripts themselves.

## Baseline: TF-IDF + Logistic Regression

```bash
python -m src.training.train_baseline
```

| Hyperparameter | Value |
|---|---|
| `max_features` | 20,000 |
| `ngram_range` | (1, 2) — unigrams + bigrams |
| `sublinear_tf` | True |
| Logistic Regression `C` | 2.0 |
| `class_weight` | balanced |
| `max_iter` | 1000 |

Trains in ~1 second on the full training split (9,958 tweets) — this speed is itself part of the honest comparison against the LSTM's ~85 seconds.

## Bidirectional LSTM

```bash
python -m src.training.train_lstm
```

| Hyperparameter | Value |
|---|---|
| Vocabulary size | up to 12,000 tokens (actual: 10,975 fitted on train split) |
| Max sequence length | 40 tokens |
| Embedding dimension | 128 (learned from scratch) |
| LSTM units | 96 per direction, bidirectional (192 effective) |
| Dense layer | 64 units, ReLU |
| Dropout | 0.4 (post-LSTM), 0.2 (post-dense) |
| Optimizer | Adam, lr = 1e-3 |
| Batch size | 64 |
| Max epochs | 12 |
| Early stopping | on `val_loss`, patience = 3, restores best weights |
| Class weighting | `sklearn.utils.class_weight.compute_class_weight("balanced")` |

### Class imbalance handling

The training split is ~63% negative / 21% neutral / 16% positive. Rather than resampling (which would either throw away majority-class data or duplicate minority-class data), the loss function is **reweighted** per class using `compute_class_weight("balanced")`, computed once per training run and passed to `model.fit(..., class_weight=...)`. This keeps every training example, while preventing the model from trivially maximizing accuracy by always predicting "negative."

### What actually happened during training

Early stopping triggered around epoch 4 in most runs — validation loss stops improving (and starts rising) after epoch 1–2, while training loss keeps falling. See `artifacts/figures/lstm_training_curves.png` for the actual curve from the last training run, and `artifacts/metrics/lstm_training_history.json` for the raw per-epoch numbers. This is textbook overfitting given the training set size (~10K short texts) relative to the model's ~1.6M parameters — discussed candidly in the README and `docs/evaluation.md` rather than hidden.

## Running both + evaluation in one command

```bash
make pipeline
# equivalent to:
python scripts/train_all.py
```
