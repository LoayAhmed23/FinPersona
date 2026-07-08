"""
Diagnostic plots for the Recommendation XGBoost models:
  - Micro-averaged ROC Curve  (across all product labels)
  - Micro-averaged Precision-Recall Curve
  - Per-product ROC curves (top-K by AUC)

Since the recommendation module trains one XGBoost per product, we compute
micro-averaged curves that aggregate across all product labels.

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
# Colour palette
# ---------------------------------------------------------------------------
_COLORS = [
    "#2196F3",  # Blue
    "#FF5722",  # Deep Orange
    "#4CAF50",  # Green
    "#9C27B0",  # Purple
    "#FF9800",  # Orange
    "#00BCD4",  # Cyan
    "#E91E63",  # Pink
    "#795548",  # Brown
    "#607D8B",  # Blue Grey
    "#CDDC39",  # Lime
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
# 1. Micro-averaged ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve_micro(
    Y_true,
    proba_dict: dict,
    valid_targets: list,
    *,
    top_k: int = 10,
    filename="recommendation_roc_curves.png",
):
    """Plot a micro-averaged ROC curve across all product labels,
    plus individual per-product curves for the top-K products by AUC.

    Parameters
    ----------
    Y_true : pd.DataFrame — ground-truth binary labels (test set).
    proba_dict : dict — {product_col: array of probabilities on test set}.
    valid_targets : list — product column names.
    top_k : int — how many individual product curves to overlay.
    filename : output PNG filename.
    """
    _apply_style()

    # Micro-average: flatten all products
    y_true_all = np.array([Y_true[p].values for p in valid_targets]).ravel()
    y_proba_all = np.array([proba_dict[p] for p in valid_targets]).ravel()

    fpr_micro, tpr_micro, _ = roc_curve(y_true_all, y_proba_all)
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Per-product curves (sorted by AUC, take top-K)
    product_aucs = []
    for p in valid_targets:
        try:
            fpr_p, tpr_p, _ = roc_curve(Y_true[p].values, proba_dict[p])
            auc_p = auc(fpr_p, tpr_p)
            product_aucs.append((p, fpr_p, tpr_p, auc_p))
        except Exception:
            continue

    product_aucs.sort(key=lambda x: x[3], reverse=True)
    for idx, (p, fpr_p, tpr_p, auc_p) in enumerate(product_aucs[:top_k]):
        label = p.replace("HAS_PROD_", "")
        color = _COLORS[idx % len(_COLORS)]
        ax.plot(
            fpr_p, tpr_p,
            color=color, linewidth=1.2, alpha=0.6,
            label=f"{label} (AUC={auc_p:.3f})",
        )

    # Micro-average on top
    ax.plot(
        fpr_micro, tpr_micro,
        color="#212121", linewidth=2.5,
        label=f"Micro-average (AUC = {roc_auc_micro:.4f})",
    )

    ax.plot(
        [0, 1], [0, 1],
        color="#999999", linewidth=1, linestyle="--", label="Random baseline",
    )

    ax.set_title(
        "Recommendation — ROC Curves (XGBoost per product)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc="lower right", fontsize=8, frameon=True, ncol=1)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    _save(fig, filename)


# ---------------------------------------------------------------------------
# 2. Micro-averaged Precision–Recall Curve
# ---------------------------------------------------------------------------

def plot_precision_recall_curve_micro(
    Y_true,
    proba_dict: dict,
    valid_targets: list,
    *,
    top_k: int = 10,
    filename="recommendation_precision_recall_curves.png",
):
    """Plot a micro-averaged PR curve across all product labels,
    plus individual per-product curves for the top-K products by PR-AUC.

    Parameters
    ----------
    Y_true : pd.DataFrame — ground-truth binary labels (test set).
    proba_dict : dict — {product_col: array of probabilities on test set}.
    valid_targets : list — product column names.
    top_k : int — how many individual product curves to overlay.
    filename : output PNG filename.
    """
    _apply_style()

    # Micro-average: flatten all products
    y_true_all = np.array([Y_true[p].values for p in valid_targets]).ravel()
    y_proba_all = np.array([proba_dict[p] for p in valid_targets]).ravel()

    prec_micro, rec_micro, _ = precision_recall_curve(y_true_all, y_proba_all)
    pr_auc_micro = auc(rec_micro, prec_micro)
    baseline = y_true_all.mean()

    fig, ax = plt.subplots(figsize=(9, 7))

    # Per-product curves
    product_aucs = []
    for p in valid_targets:
        try:
            prec_p, rec_p, _ = precision_recall_curve(
                Y_true[p].values, proba_dict[p],
            )
            auc_p = auc(rec_p, prec_p)
            product_aucs.append((p, rec_p, prec_p, auc_p))
        except Exception:
            continue

    product_aucs.sort(key=lambda x: x[3], reverse=True)
    for idx, (p, rec_p, prec_p, auc_p) in enumerate(product_aucs[:top_k]):
        label = p.replace("HAS_PROD_", "")
        color = _COLORS[idx % len(_COLORS)]
        ax.plot(
            rec_p, prec_p,
            color=color, linewidth=1.2, alpha=0.6,
            label=f"{label} (AUC={auc_p:.3f})",
        )

    # Micro-average on top
    ax.plot(
        rec_micro, prec_micro,
        color="#212121", linewidth=2.5,
        label=f"Micro-average (AUC = {pr_auc_micro:.4f})",
    )

    ax.axhline(
        baseline, color="#999999", linewidth=1, linestyle="--",
        label=f"Baseline (prevalence = {baseline:.3f})",
    )

    ax.set_title(
        "Recommendation — Precision–Recall Curves (XGBoost per product)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=1)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])

    _save(fig, filename)
