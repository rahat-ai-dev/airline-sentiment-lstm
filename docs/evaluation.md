# Evaluation

## Methodology

Both models are scored on the **same held-out test split** (2,134 tweets), which neither model's parameters, vocabulary, nor vectorizer ever saw during training. `src/evaluation/evaluate.py` computes an identical metric bundle for each model:

- Accuracy
- Precision / recall / F1, both **macro-averaged** (unweighted mean across classes — chosen deliberately over accuracy alone, since accuracy alone is misleading on this dataset's 63/21/16 class imbalance) and **per-class**
- Confusion matrix
- Average inference latency per tweet
- Parameter count (LSTM only — meaningless for the sklearn pipeline)

## Results (test set, n=2,134)

| Metric | Baseline | LSTM |
|---|---|---|
| Accuracy | 0.7985 | 0.7943 |
| Macro Precision | 0.7339 | 0.7217 |
| Macro Recall | 0.7469 | **0.7524** |
| Macro F1 | 0.7399 | 0.7351 |

Per-class F1:

| Class | Baseline | LSTM |
|---|---|---|
| negative | 0.879 | 0.878 |
| neutral | 0.625 | 0.621 |
| positive | 0.716 | **0.707** |

(Exact numbers regenerate at `artifacts/metrics/model_comparison.json` via `python -m src.evaluation.evaluate`; re-running may shift results by a point or two due to TensorFlow's residual non-determinism on CPU even with a fixed seed.)

## Error analysis

The dominant confusion for **both** models (see `artifacts/figures/confusion_matrix_baseline.png` / `confusion_matrix_lstm.png`) is **neutral tweets predicted as negative**. Manually inspecting a sample of these errors (`notebooks/03_explainability.ipynb`) shows why: many "neutral" tweets are mildly sarcastic or lukewarm complaints that sit right on the human-annotation boundary — even the original Crowdflower annotators plausibly disagreed on some of these, since the dataset provides only a single majority label per tweet.

## No data leakage

- The stratified train/val/test split happens **before** any vocabulary or TF-IDF vectorizer is fit.
- Deduplication happens **before** the split, so no near-identical retweet can appear in both train and test.
- The LSTM's early stopping monitors `val_loss` (not `test_loss`) and the test set is only touched once, at final evaluation.

## Why the LSTM doesn't clearly win here (and what would change that)

See the README's [Experiments & Results](../README.md#experiments--results) section and `docs/training.md` for the full discussion — in short, this is a small-data regime (~10K training tweets, ~15 words each) where TF-IDF bigrams already capture most of the local negation signal that drives short-text sentiment, and a from-scratch-embedding LSTM has too little data to learn a better representation than that. Pretrained embeddings or a fine-tuned transformer (see README → Future Improvements) are the standard fixes.
