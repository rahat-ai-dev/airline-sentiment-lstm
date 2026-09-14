# Architecture

## Design principles

1. **Separation of concerns.** `src/` contains all logic; `app/` is a thin presentation layer; `notebooks/` are for exploration and call into `src/`, never the reverse.
2. **Single source of configuration.** Every path and hyperparameter lives in `src/config.py`. No other module hardcodes a path, a batch size, or a random seed.
3. **No train/serve skew.** `src/data/text_cleaning.py` is imported by both the offline preprocessing pipeline and the online inference path (`src/inference/predict.py`), so a tweet is transformed identically whether it's in training data or typed into the Streamlit app.
4. **Fit only on training data.** The vocabulary (`src/features/vocabulary.py`) and the baseline's TF-IDF vectorizer are both fit exclusively on the training split, after the stratified split — not before — to prevent validation/test information leaking into the model's input representation.

## Component diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                              data/                                    │
│  raw/Tweets.csv → processed/splits/{train,val,test}.parquet          │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 ▼                                 ▼
      src/data/validation.py            src/data/text_cleaning.py
                 │                                 │
                 └───────────────┬─────────────────┘
                                  ▼
                    src/data/preprocessing.py
                                  │
                 ┌────────────────┴─────────────────┐
                 ▼                                   ▼
   src/features/vocabulary.py            sklearn TfidfVectorizer
   src/models/lstm_model.py              src/models/baseline_model.py
   src/training/train_lstm.py            src/training/train_baseline.py
                 │                                   │
                 └────────────────┬──────────────────┘
                                   ▼
                     src/evaluation/evaluate.py
                     src/evaluation/explainability.py
                                   │
                                   ▼
                     src/inference/predict.py
                        (SentimentPredictor)
                                   │
                                   ▼
                        app/streamlit_app.py
                    (pages/: home, prediction,
                     performance, dataset_explorer,
                     explainability, about)
```

## Why a Streamlit-calls-src, not src-calls-Streamlit, boundary

`SentimentPredictor.predict()` returns a plain `PredictionResult` dataclass — no Streamlit types anywhere in `src/`. This means:
- `src/` is fully unit-testable without a running Streamlit process (see `tests/test_inference.py`).
- The same predictor could be wrapped in a FastAPI endpoint, a batch scoring script, or a Slack bot with zero changes to `src/`.

## Data flow guarantees

- **Reproducibility**: a fixed seed (`RANDOM_SEED = 42` in `src/config.py`) is threaded through the split, class-weight computation, and TensorFlow's global seed.
- **No hardcoded paths**: every path is derived from `ROOT_DIR = Path(__file__).resolve().parents[1]` in `src/config.py`, so the project runs identically regardless of clone location.
