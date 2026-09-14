"""Matplotlib figure generation for reports, docs, and notebooks.

Kept separate from Streamlit's own (Plotly-based) charts: these are the
static PNGs referenced by the README / model card / docs, generated once by
``scripts/make_figures.py`` and committed under artifacts/figures/.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, FIGURES_DIR, LABEL_COLUMN, METRICS_DIR, SPLITS_DIR

plt.rcParams["figure.dpi"] = 120


def plot_class_distribution() -> None:
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    counts = train_df[LABEL_COLUMN].value_counts().reindex(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#d62728", "#7f7f7f", "#2ca02c"]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Class Distribution — Training Split")
    ax.set_ylabel("Number of tweets")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png")
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]], title: str, filename: str) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def plot_training_curves() -> None:
    history_path = METRICS_DIR / "lstm_training_history.json"
    if not history_path.exists():
        return
    history = json.loads(history_path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lstm_training_curves.png")
    plt.close(fig)


def plot_model_comparison() -> None:
    comparison_path = METRICS_DIR / "model_comparison.json"
    if not comparison_path.exists():
        return
    comparison = json.loads(comparison_path.read_text())

    plot_confusion_matrix(
        comparison["baseline"]["confusion_matrix"],
        "Confusion Matrix — TF-IDF + LogReg (baseline)",
        "confusion_matrix_baseline.png",
    )
    plot_confusion_matrix(
        comparison["lstm"]["confusion_matrix"],
        "Confusion Matrix — Bidirectional LSTM",
        "confusion_matrix_lstm.png",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    x = np.arange(len(metrics))
    width = 0.35
    baseline_vals = [comparison["baseline"][m] for m in metrics]
    lstm_vals = [comparison["lstm"][m] for m in metrics]
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline (TF-IDF+LogReg)")
    ax.bar(x + width / 2, lstm_vals, width, label="Bidirectional LSTM")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"])
    ax.set_ylim(0, 1)
    ax.set_title("Baseline vs. LSTM — Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png")
    plt.close(fig)


def generate_all_figures() -> None:
    plot_class_distribution()
    plot_training_curves()
    plot_model_comparison()


if __name__ == "__main__":
    generate_all_figures()
