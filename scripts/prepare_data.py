#!/usr/bin/env python3
"""Reproducible data preparation entry point.

    python scripts/prepare_data.py

Loads data/raw/Tweets.csv, validates it, cleans it, and writes stratified
train/val/test splits to data/processed/splits/.
"""
from src.data.preprocessing import run_preprocessing_pipeline

if __name__ == "__main__":
    run_preprocessing_pipeline(save=True)
