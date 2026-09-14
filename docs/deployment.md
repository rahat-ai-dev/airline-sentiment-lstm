# Deployment

## What the app needs at runtime

Only three artifacts, all produced by `make pipeline`:
- `models/lstm_sentiment.keras` (~6 MB)
- `models/vocab.json`
- `data/processed/splits/*.parquet` (for the Dataset Explorer and Home page stats — not required for prediction itself)

No GPU is required for inference — the model is small enough to run comfortably on CPU (~0.6 ms/tweet, see `docs/evaluation.md`).

## Option 1 — Streamlit Community Cloud

1. Push this repository to GitHub (see `.gitignore` — trained model files are excluded by default to keep the repo light; either commit them explicitly with `git add -f models/*.keras models/vocab.json`, or add a startup step that runs `make pipeline` on first boot).
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at this repo, branch, and `app/streamlit_app.py` as the entry point.
3. Set the Python version to match `pyproject.toml` (`>=3.10`) and let Streamlit Cloud install from `requirements.txt`.
4. If the model artifacts aren't committed, add a `pre-run` step (or a one-time setup script) that runs:
   ```bash
   python scripts/prepare_data.py
   python -m src.training.train_baseline
   python -m src.training.train_lstm
   ```

## Option 2 — Hugging Face Spaces (Streamlit SDK)

1. Create a new Space, SDK = Streamlit.
2. Push this repo's contents to the Space's git remote.
3. Set `app_file: app/streamlit_app.py` in the Space's `README.md` front matter.
4. Either commit the trained model artifacts directly (Spaces support Git LFS for files over 10MB, though this model is small enough not to need it) or run the training pipeline once via the Space's build step.

## Environment variables

None are required to run the app as-is. `.env.example` documents the one currently-used variable (`LOG_LEVEL`) and reserves a placeholder for future secrets (e.g. a hosted-inference API token) so the pattern is in place before it's needed — no secrets are hardcoded anywhere in the codebase.

## Large-file note

The raw dataset (`data/raw/Tweets.csv`, ~2.9 MB) and trained model (`models/lstm_sentiment.keras`, ~6 MB) are both small enough to commit directly to a standard Git repository. If this project were extended to a larger dataset or a larger model (e.g. a fine-tuned transformer, which can run into the hundreds of MB), the recommended pattern is:
- Keep the large file out of Git (add to `.gitignore`).
- Provide a `scripts/download_artifacts.py` (or a documented `wget`/`curl` command) pointing at a release asset or cloud storage bucket.
- Never replace a large real dataset/model with a synthetic stand-in just to fit under GitHub's file-size limits.
