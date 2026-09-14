# ✈️ Airline Sentiment Intelligence

**Deep learning (Bidirectional LSTM) for real-time airline customer-feedback triage — with an honest baseline comparison, explainability, and a production Streamlit app.**

🔗 **Live App:** https://airline-sentiment-lstm-mqbuzafkuxasyxcswwxkj9.streamlit.app/
📦 **Repository:** https://github.com/rahat-ai-dev/airline-sentiment-lstm

> Built end-to-end on real, publicly released data: the Twitter US Airline Sentiment dataset (14,640 tweets, February 2015). No synthetic data anywhere in this project.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Why This Project](#why-this-project)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Data Pipeline](#data-pipeline)
- [Models](#models)
- [Experiments & Results](#experiments--results)
- [Explainability](#explainability)
- [Streamlit Demo](#streamlit-demo)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Project Overview

Airlines are mentioned publicly on social media thousands of times a day. This project builds a **3-class sentiment classifier** (negative / neutral / positive) for airline-related tweets, using a **bidirectional LSTM** sequence model, and packages it as a full, reproducible, testable, and explainable system — not a single notebook.

Every stage of a real ML product is represented: data validation → cleaning → a classical baseline → a deep learning model → honest head-to-head evaluation → explainability → a deployable UI.

## Problem Statement

Customer-support and social-media teams at airlines need to **triage** an unbounded stream of public mentions: which tweets are angry customers who need an urgent human response, and which are neutral or positive noise? Manually reading every tweet doesn't scale. A model that classifies sentiment automatically — and *explains* its reasoning — lets a small support team focus attention where it's needed most.

## Why This Project

1. **Real business relevance** — social-listening/triage is an actual product category (Sprinklr, Brandwatch, Hootsuite Insights all sell this).
2. **Genuine sequence-modeling problem** — sentiment in short, informal text depends on word order and negation ("not bad" vs "bad"), which a bag-of-words model structurally cannot represent as well as a sequence model.
3. **Honest, not inflated, results** — this project reports a real, slightly inconvenient finding (see [Experiments & Results](#experiments--results)) instead of cherry-picking a favorable number, and explains *why*, which is the kind of judgment that separates production ML engineering from a tutorial.
4. **Full-stack, not just a model** — validation, testing, explainability, and a deployable app are all present, not just a `.ipynb` with an accuracy score at the bottom.
5. **Room to grow into a product** — extending this into a live Twitter-listening pipeline with a scheduled re-training job is a straightforward next step (see [Future Improvements](#future-improvements)).

## Key Features

- Real raw dataset, validated and versioned pipeline (`data/raw` → `data/processed`)
- Config-driven, no hardcoded paths or magic numbers (`src/config.py`)
- Bidirectional LSTM built and trained from scratch in TensorFlow/Keras
- TF-IDF + Logistic Regression baseline, trained and evaluated identically, for a fair comparison
- Class-imbalance handling via computed class weights
- Full metric suite: accuracy, precision/recall/F1 (macro + per-class), confusion matrix, latency, parameter count
- Occlusion-based explainability (the text-model analogue of Grad-CAM)
- Premium multi-page Streamlit app with live prediction, dataset exploration, and a performance dashboard
- 27 unit tests covering cleaning, vocabulary, data validation, model construction, and inference
- Reproducible one-command pipeline (`make pipeline`)

## Architecture

Raw Data (data/raw/Tweets.csv)
│
▼
Data Validation ───────────────► src/data/validation.py
│
▼
Text Cleaning + Dedup ─────────► src/data/text_cleaning.py, preprocessing.py
│
▼
Stratified Train/Val/Test Split
│
├──────────────┐
▼ ▼
TF-IDF Baseline Vocabulary + Bidirectional LSTM
(sklearn) (TensorFlow/Keras)
│ │
└──────┬───────┘
▼
Head-to-Head Evaluation ───► src/evaluation/evaluate.py
│
▼
Explainability (Occlusion) ─► src/evaluation/explainability.py
│
▼
Inference Service ─────────► src/inference/predict.py
│
▼
Streamlit Application ─────► app/streamlit_app.py


Design principle: the UI (`app/`) never touches TensorFlow, pandas transformations, or the vocabulary directly — it only calls `src.inference.predict.SentimentPredictor`. This means the same inference module could be dropped behind a REST API tomorrow with zero UI changes.

## Dataset

**Twitter US Airline Sentiment** — see [`DATASET.md`](DATASET.md) for full provenance, license, and known biases. In short: 14,640 real tweets about 6 major US airlines, scraped in February 2015 and labeled by human annotators as negative / neutral / positive, with negative tweets additionally tagged with a reason (e.g. "Late Flight", "Customer Service Issue").

## Data Pipeline

1. **Validate** (`src/data/validation.py`) — schema, label set, empty-text ratio, duplicate ratio checks; fails loudly on a malformed CSV.
2. **Clean** (`src/data/text_cleaning.py`) — lowercase, strip URLs and `@mentions` (airline handles are removed so the model can't shortcut sentiment prediction by recognizing "which airline"), collapse elongated characters, strip punctuation.
3. **Deduplicate** — exact duplicate (clean_text, label) pairs removed (mostly retweets) so the same example can't appear in both train and test.
4. **Stratified split** (70/15/15) — class balance preserved in every split; the vocabulary and TF-IDF vectorizer are fit **only on the training split**, so no information from validation/test tweets leaks into the model's input representation.

Run it yourself:
```bash
python scripts/prepare_data.py
```

## Models

### Baseline — TF-IDF + Logistic Regression
Unigrams + bigrams, 20,000 features, class-balanced Logistic Regression. Included specifically to prove (or disprove) that the LSTM's added complexity earns its keep — see `src/models/baseline_model.py`.

### Bidirectional LSTM

Input (40 token ids)
→ Embedding (128-dim, learned, masks padding)
→ Bidirectional LSTM (96 units/direction)
→ Dropout (0.4)
→ Dense (64, ReLU)
→ Dropout (0.2)
→ Dense (3, softmax)

~1.59M trainable parameters. Trained with class weighting to counter the dataset's negative-tweet skew, and early stopping on validation loss (`src/models/lstm_model.py`, `src/training/train_lstm.py`).

## Experiments & Results

Both models evaluated on the **same held-out test split** (2,134 tweets, never used for training, vocabulary fitting, or vectorizer fitting).

| Metric | Baseline (TF-IDF + LogReg) | Bidirectional LSTM |
|---|---|---|
| Accuracy | **79.9%** | 79.4% |
| Macro Precision | **73.4%** | 72.2% |
| Macro Recall | 74.7% | **75.2%** |
| Macro F1 | **74.0%** | 73.5% |
| Avg. inference latency / tweet | **0.02 ms** | 0.61 ms |
| Parameters | ~20K TF-IDF features + LR weights | 1,590,147 |

**Honest conclusion: the LSTM does not clearly outperform the classical baseline on this dataset.** This is a real, reproducible result (see `notebooks/02_modeling_and_evaluation.ipynb` and `artifacts/metrics/model_comparison.json`), not a mistake, and here's why:

- **Small-data regime.** After cleaning, there are only ~9,958 training tweets. A 1.6M-parameter LSTM trained from scratch has very little signal per parameter at that scale — validation loss stops improving after epoch 1–2 (see `artifacts/figures/lstm_training_curves.png`), a textbook small-data overfitting pattern.
- **Short texts favor n-grams.** Tweets average ~15–17 words. TF-IDF bigrams already capture most local negation ("no help", "not bad") that matters at this length; the LSTM's structural advantage (long-range dependencies) matters more for longer documents.
- **Where the LSTM does edge ahead:** macro recall, and specifically recall on the minority `positive` class (76.7% vs 72.4%) — consistent with the class-weighted loss doing its job of not ignoring minority classes.

See [Future Improvements](#future-improvements) for what would actually close this gap in a real deployment (pretrained embeddings, more data, or a fine-tuned transformer).

Regenerate these results yourself:
```bash
make pipeline
```

Figures: `artifacts/figures/class_distribution.png`, `model_comparison.png`, `confusion_matrix_baseline.png`, `confusion_matrix_lstm.png`, `lstm_training_curves.png`.

## Explainability

Grad-CAM is a computer-vision technique — it needs a spatial convolutional feature map, which a recurrent text model doesn't have. This project uses the standard text-model substitute instead: **occlusion / leave-one-out importance** (`src/evaluation/explainability.py`). Each token in a tweet is removed one at a time and the model is re-scored; the drop in the predicted class's probability is that token's importance.

Example (from `notebooks/03_explainability.ipynb`):

"My flight has been delayed 6 hours and nobody will explain why."
-> negative (93.5%)
hours +0.117
no +0.063
delayed +0.089

Available live in the Streamlit app's **Explainability** page, with a color-highlighted rendering of the tweet.

## Streamlit Demo

🔗 **Try it live:** https://airline-sentiment-lstm-mqbuzafkuxasyxcswwxkj9.streamlit.app/

Or run it locally:
```bash
streamlit run app/streamlit_app.py
```

Pages: **Home** (hero + key stats) · **Predict** (live inference + on-demand explanation) · **Model Performance** (full metric dashboard, confusion matrices, training curves, an honest discussion of the baseline-vs-LSTM result) · **Dataset Explorer** (class balance, per-airline breakdown, top complaint reasons, sample browser) · **Explainability** (occlusion analysis walkthrough) · **About**.

The model is loaded once via `st.cache_resource`; dataframes and metrics are cached via `st.cache_data` — the model is never reloaded on user interaction.

## Installation

```bash
git clone https://github.com/rahat-ai-dev/airline-sentiment-lstm.git
cd airline-sentiment-lstm
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Usage

```bash
# 1. Prepare data (idempotent, deterministic)
python scripts/prepare_data.py

# 2. Train both models
python -m src.training.train_baseline
python -m src.training.train_lstm

# 3. Evaluate + generate figures
python -m src.evaluation.evaluate
python -m src.visualization.plots

# 4. Launch the app
streamlit run app/streamlit_app.py
```

Or all at once:
```bash
make pipeline   # data -> train -> evaluate -> figures
make app        # launch Streamlit
```

## Training

See [`docs/training.md`](docs/training.md) for hyperparameters, the class-weighting strategy, and reproducibility details (fixed seed = 42 throughout, defined once in `src/config.py`).

## Evaluation

See [`docs/evaluation.md`](docs/evaluation.md) for the full metric methodology and error analysis.

## Project Structure

airline-sentiment-lstm/
├── app/ # Streamlit application (thin UI layer)
│ ├── streamlit_app.py
│ ├── pages/ # home, prediction, performance, dataset_explorer, explainability, about
│ └── utils/ # styling, cached resource loaders
├── data/
│ ├── raw/ # original, untouched Tweets.csv
│ ├── interim/, external/ # reserved for future intermediate/external data
│ └── processed/splits/ # train/val/test parquet (generated, reproducible)
├── notebooks/ # EDA, modeling & evaluation, explainability (all executed, real outputs)
├── src/
│ ├── config.py # all paths & hyperparameters — single source of truth
│ ├── data/ # validation, text cleaning, preprocessing pipeline
│ ├── features/ # vocabulary / sequence encoder
│ ├── models/ # LSTM architecture, baseline pipeline
│ ├── training/ # train_lstm.py, train_baseline.py
│ ├── evaluation/ # evaluate.py, explainability.py
│ ├── inference/ # predict.py — the one production entry point
│ ├── visualization/ # static figure generation
│ └── utils/ # logging
├── models/ # trained artifacts (lstm_sentiment.keras, vocab.json, baseline .joblib)
├── artifacts/{metrics,figures}/ # generated JSON metrics + PNG figures
├── tests/ # 27 pytest tests
├── scripts/ # prepare_data.py, train_all.py
├── docs/ # architecture, training, evaluation, deployment, model_card
├── .github/workflows/ # CI: lint + test on every push
├── requirements.txt, pyproject.toml, Makefile
├── DATASET.md, LICENSE, .env.example
└── README.md


## Testing

```bash
pytest tests/ -v
```
27 tests covering text cleaning, vocabulary encode/decode round-trips, data validation edge cases, stratified split balance, model architecture shape/compile checks, and the inference service (prediction ordering, batch prediction, explanation shape) using a small model trained on the fly — so tests don't depend on the full trained artifact being present.

## Deployment

Live on Streamlit Community Cloud: https://airline-sentiment-lstm-mqbuzafkuxasyxcswwxkj9.streamlit.app/

See [`docs/deployment.md`](docs/deployment.md) for full Streamlit Community Cloud and Hugging Face Spaces instructions. In short: the trained model (`models/lstm_sentiment.keras`, a few MB) and `models/vocab.json` are the only artifacts the app needs at runtime — no GPU required for inference.

## Limitations

- Trained on 2015 US-airline Twitter data; slang, events, and airline-specific context from that period may not generalize to other domains, platforms, or eras.
- The LSTM does not clearly outperform the classical baseline at this dataset size (see [Experiments & Results](#experiments--results)) — this is disclosed, not hidden.
- 3-class sentiment is coarse; it does not capture emotion type (frustration vs. sarcasm vs. genuine anger) or urgency.
- No demographic attributes are in the dataset, but usernames/handles are present in raw data and are stripped before modeling — not before storage. See `DATASET.md` for data-handling notes.

## Future Improvements

1. **Pretrained word embeddings** (GloVe/FastText) instead of learning embeddings from scratch on ~10K tweets — the single highest-leverage change to close the baseline gap.
2. **Fine-tune a small pretrained transformer** (e.g. DistilBERT) as a stronger deep-learning contender — a natural "Advanced Model" tier beyond the LSTM.
3. **More data** — combine with additional airline-sentiment or general-sentiment corpora to give the LSTM enough examples to out-learn n-gram features.
4. **Attention layer** in the LSTM for built-in (not just post-hoc occlusion) interpretability.
5. **Live ingestion** — replace the static CSV with a scheduled Twitter/X API pull and nightly re-training job, turning this from a portfolio project into an actual monitoring product.
6. **REST API** — expose `src.inference.predict.SentimentPredictor` behind FastAPI so non-Streamlit consumers (e.g. a support-ticketing system) can call it directly.

## Author

**Rahat Mia**
Built as a portfolio project demonstrating an end-to-end, honestly-evaluated NLP deep learning pipeline — from real raw data to a deployable, explainable application.

GitHub: [@rahat-ai-dev](https://github.com/rahat-ai-dev)