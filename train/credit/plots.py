"""
Diagnostic plots for the Credit Risk model:
  - Training vs. Validation Learning Curves  (from the actual training run)
  - ROC Curve
  - Precision-Recall Curve

All figures are saved to config.PREDICTIONS_DIR (outputs/).
"""

import os

import config
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

def _apply_style():
    """Apply a consistent, publication-ready style to all plots."""
    plt.rcParams.update({
        "figure.facecolor": "#fafafa",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def _save(fig, filename):
    path = os.path.join(config.PREDICTIONS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] Saved {path}")


# ---------------------------------------------------------------------------
# 1. Learning Curves  (from the actual XGBoost training run)
# ---------------------------------------------------------------------------

def plot_learning_curves(
    evals_result: dict,
    *,
    metric="auc",
    filename="credit_learning_curves.png",
):
    """Plot training vs. validation score per boosting round.

    Uses the ``evals_result`` dict populated during the actual
    ``xgb.train()`` call — **no extra models are trained**.

    Parameters
    ----------
    evals_result : dict
        Typically ``{"train": {"auc": [...]}, "valid": {"auc": [...]}}``.
    metric : str
        The eval-metric key to plot (must match ``config.XGB_PARAMS["eval_metric"]``).
    filename : str
        Output PNG filename.
    """
    _apply_style()

    train_scores = evals_result["train"][metric]
    val_scores = evals_result["valid"][metric]
    rounds = range(1, len(train_scores) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        rounds, train_scores,
        color="#2196F3", linewidth=2, label="Training",
    )
    ax.plot(
        rounds, val_scores,
        color="#FF5722", linewidth=2, label="Validation",
    )

    # Mark best iteration
    best_idx = int(np.argmax(val_scores))
    ax.axvline(
        best_idx + 1, color="#999999", linestyle="--", linewidth=1,
        label=f"Best round ({best_idx + 1}, {val_scores[best_idx]:.4f})",
    )
    ax.scatter(
        [best_idx + 1], [val_scores[best_idx]],
        color="#FF5722", s=80, zorder=5, edgecolors="white", linewidths=1.5,
    )

    ax.set_title(
        "Credit Risk — Training vs. Validation Curves",
        fontsize=15, fontweight="bold",
    )
    ax.set_xlabel("Boosting Round", fontsize=12)
    ax.set_ylabel(f"Score ({metric.upper()})", fontsize=12)
    ax.legend(loc="lower right", fontsize=11, frameon=True)

    _save(fig, filename)


# ---------------------------------------------------------------------------
# 2. ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true,
    y_proba,
    *,
    filename="credit_roc_curve.png",
):
    """Plot the Receiver Operating Characteristic curve."""
    _apply_style()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        fpr, tpr,
        color="#2196F3", linewidth=2.5,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
    )
    ax.plot(
        [0, 1], [0, 1],
        color="#999999", linewidth=1, linestyle="--", label="Random baseline",
    )

    ax.fill_between(fpr, tpr, alpha=0.10, color="#2196F3")

    ax.set_title("Credit Risk — ROC Curve", fontsize=15, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    _save(fig, filename)


# ---------------------------------------------------------------------------
# 3. Precision–Recall Curve
# ---------------------------------------------------------------------------

def plot_precision_recall_curve(
    y_true,
    y_proba,
    *,
    filename="credit_precision_recall_curve.png",
):
    """Plot the Precision–Recall curve."""
    _apply_style()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        recall, precision,
        color="#FF5722", linewidth=2.5,
        label=f"PR curve (AUC = {pr_auc:.4f})",
    )
    ax.axhline(
        baseline, color="#999999", linewidth=1, linestyle="--",
        label=f"Baseline (prevalence = {baseline:.3f})",
    )

    ax.fill_between(recall, precision, alpha=0.10, color="#FF5722")

    ax.set_title(
        "Credit Risk — Precision–Recall Curve",
        fontsize=15, fontweight="bold",
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend(loc="upper right", fontsize=11, frameon=True)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    _save(fig, filename)
