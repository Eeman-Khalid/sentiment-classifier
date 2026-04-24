"""
evaluate.py
-----------
Shared evaluation utilities used by all three model training scripts.
Produces: accuracy, precision, recall, F1, confusion matrix, and
          a comparison bar chart when called with multiple model results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

LABEL_NAMES = ["negative", "neutral", "positive"]


def compute_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Compute and print all evaluation metrics.
    Returns a dict for use in comparison plots.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model", save_path: str = None):
    """
    Plot a labelled confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved → {save_path}")
    plt.show()


def plot_model_comparison(results: list[dict], save_path: str = None):
    """
    Bar chart comparing accuracy, precision, recall, F1
    across multiple models.

    Args:
        results: list of dicts returned by compute_metrics()
        save_path: optional file path to save the figure
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = [r["model"] for r in results]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, result in enumerate(results):
        values = [result[m] for m in metrics]
        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=result["model"], color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score"], fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Comparison chart saved → {save_path}")
    plt.show()