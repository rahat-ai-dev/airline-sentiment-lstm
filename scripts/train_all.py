#!/usr/bin/env python3
"""Run the full reproducible pipeline end to end:

    python scripts/train_all.py

data/raw/Tweets.csv -> processed splits -> baseline -> LSTM -> evaluation -> figures
"""
from src.data.preprocessing import run_preprocessing_pipeline
from src.evaluation.evaluate import run_full_evaluation
from src.training.train_baseline import train_baseline
from src.training.train_lstm import train_lstm
from src.utils.logger import get_logger
from src.visualization.plots import generate_all_figures

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Step 1/5 — preparing data")
    run_preprocessing_pipeline(save=True)

    logger.info("Step 2/5 — training baseline")
    train_baseline()

    logger.info("Step 3/5 — training LSTM")
    train_lstm()

    logger.info("Step 4/5 — evaluating both models")
    run_full_evaluation()

    logger.info("Step 5/5 — generating figures")
    generate_all_figures()

    logger.info("Pipeline complete. See artifacts/metrics/ and artifacts/figures/.")
