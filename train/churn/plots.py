"""
Diagnostic plots for the Churn Prediction model:
  - Training vs. Validation Learning Curves (from actual training runs)
  - ROC Curves (all classifiers on one plot)
  - Precision-Recall Curves (all classifiers on one plot)

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
# Colour palette for up to 7 classifiers
# ---------------------------------------------------------------------------
_COLORS = [
    "#2196F3",  # Blue
    "#FF5722",  # Deep Orange
    "#4CAF50",  # Green
    "#9C27B0",  # Purple
    "#FF9800",  # Orange
    "#00BCD4",  # Cyan
    "#E91E63",  # Pink
]


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
# 1. Learning Curves  (from actual training runs — no extra training)
# ---------------------------------------------------------------------------

def plot_learning_curves(
    results: dict,
    *,
    filename="churn_learning_curves.png",
):
    """Plot training vs. validation score per iteration for iterative models.

    Only classifiers that have a ``history`` entry (XGBoost, LightGBM)
    are plotted — non-iterative models (LR, DT, RF) are skipped.

    Uses the eval histories captured during the actual ``.fit()`` call,
    so **no extra models are trained**.

    Parameters
    ----------
    results : dict
        {name: {"history": dict or None, ...}} — the training results dict.
    filename : str
        Output PNG filename.
    """
    _apply_style()

    # Filter to only models with training history
    iterative = {
        name: res for name, res in results.items()
        if res.get("history") is not None
    }

    if not iterative:
        print("  [plots] No iterative models with training history — skipping learning curves.")
        return

    n = len(iterative)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, (name, res) in enumerate(iterative.items()):
        ax = axes[idx]
        history = res["history"]

        # XGBoost: {"validation_0": {"logloss": [...]}, "validation_1": {"logloss": [...]}}
        # LightGBM: {"training": {"auc": [...]}, "valid_1": {"auc": [...]}}
        # Find the metric and dataset keys dynamically
        dataset_keys = list(history.keys())
        train_key = dataset_keys[0]  # first eval_set = training data
        val_key = dataset_keys[1] if len(dataset_keys) > 1 else dataset_keys[0]

        # Pick the first metric available
        metric_name = list(history[train_key].keys())[0]
        train_scores = history[train_key][metric_name]
        val_scores = history[val_key][metric_name]
        rounds = range(1, len(train_scores) + 1)

        ax.plot(
            rounds, train_scores,
            color="#2196F3", linewidth=2, label="Training",
        )
        ax.plot(
            rounds, val_scores,
            color="#FF5722", linewidth=2, label="Validation",
        )

        # Mark best iteration (use max for metrics like AUC, min for loss)
        is_loss = "loss" in metric_name.lower() or "error" in metric_name.lower()
        if is_loss:
            best_idx = int(np.argmin(val_scores))
        else:
            best_idx = int(np.argmax(val_scores))

        ax.axvline(
            best_idx + 1, color="#999999", linestyle="--", linewidth=1,
            label=f"Best round ({best_idx + 1})",
        )
        ax.scatter(
            [best_idx + 1], [val_scores[best_idx]],
            color="#FF5722", s=80, zorder=5, edgecolors="white", linewidths=1.5,
        )

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.legend(loc="best", fontsize=9, frameon=True)

    fig.suptitle(
        "Churn Prediction — Training vs. Validation Curves",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# 2. ROC Curves  (all classifiers overlaid)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true,
    results: dict,
    *,
    filename="churn_roc_curves.png",
):
    """Plot ROC curves for all classifiers on a single figure.

    Parameters
    ----------
    y_true : array-like — ground-truth labels.
    results : dict — {name: {"probs": array, ...}}.
    filename : output PNG filename.
    """
    _apply_style()

    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, (name, res) in enumerate(results.items()):
        y_proba = res["probs"]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc_val = auc(fpr, tpr)
        color = _COLORS[idx % len(_COLORS)]

        ax.plot(
            fpr, tpr,
            color=color, linewidth=2,
            label=f"{name} (AUC = {roc_auc_val:.4f})",
        )

    ax.plot(
        [0, 1], [0, 1],
        color="#999999", linewidth=1, linestyle="--", label="Random baseline",
    )

    ax.set_title("Churn Prediction — ROC Curves", fontsize=15, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    _save(fig, filename)


# ---------------------------------------------------------------------------
# 3. Precision–Recall Curves  (all classifiers overlaid)
# ---------------------------------------------------------------------------

def plot_precision_recall_curves(
    y_true,
    results: dict,
    *,
    filename="churn_precision_recall_curves.png",
):
    """Plot PR curves for all classifiers on a single figure.

    Parameters
    ----------
    y_true : array-like — ground-truth labels.
    results : dict — {name: {"probs": array, ...}}.
    filename : output PNG filename.
    """
    _apply_style()

    baseline = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, (name, res) in enumerate(results.items()):
        y_proba = res["probs"]
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc_val = auc(recall, precision)
        color = _COLORS[idx % len(_COLORS)]

        ax.plot(
            recall, precision,
            color=color, linewidth=2,
            label=f"{name} (AUC = {pr_auc_val:.4f})",
        )

    ax.axhline(
        baseline, color="#999999", linewidth=1, linestyle="--",
        label=f"Baseline (prevalence = {baseline:.3f})",
    )

    ax.set_title(
        "Churn Prediction — Precision–Recall Curves",
        fontsize=15, fontweight="bold",
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    _save(fig, filename)
