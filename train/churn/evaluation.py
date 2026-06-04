"""
Evaluation metrics and report generation for the Churn Prediction System.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

import config


def evaluate(y_true, y_pred, y_proba) -> dict:
    """Compute a full set of binary-classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
    }


def evaluate_all(y_true, results: dict) -> pd.DataFrame:
    """Evaluate all trained models and return a comparison DataFrame.

    Parameters
    ----------
    y_true : array-like — ground truth labels
    results : dict — {name: {"preds": array, "probs": array, ...}}

    Returns
    -------
    pd.DataFrame with one row per model, columns = metric names.
    """
    rows = {}
    for name, res in results.items():
        rows[name] = evaluate(y_true, res["preds"], res["probs"])

    df = pd.DataFrame(rows).T
    df = df.sort_values("ROC_AUC", ascending=False)
    return df


def generate_report(
    metrics_df: pd.DataFrame,
    y_true,
    best_name: str,
    best_preds,
    output_path: str = None,
) -> str:
    """Build a plain-text evaluation report and optionally write to disk."""
    lines = []
    lines.append("=" * 60)
    lines.append("CHURN PREDICTION — EVALUATION REPORT")
    lines.append("=" * 60)

    # All models comparison
    lines.append("\nMODEL COMPARISON")
    lines.append("-" * 60)
    lines.append(metrics_df.to_string())

    # Best model details
    lines.append(f"\n\nBEST MODEL: {best_name}")
    lines.append("-" * 40)

    # Class distribution
    total = len(y_true)
    n_churn = int(np.sum(y_true == 1))
    n_retained = total - n_churn
    lines.append(f"\nCLASS DISTRIBUTION (test set)")
    lines.append(f"  Retained (0): {n_retained:>7,}  ({n_retained / total * 100:.1f}%)")
    lines.append(f"  Churned  (1): {n_churn:>7,}  ({n_churn / total * 100:.1f}%)")

    # Confusion matrix
    cm = confusion_matrix(y_true, best_preds)
    lines.append(f"\nCONFUSION MATRIX")
    lines.append(f"  TN={cm[0, 0]:>7,}   FP={cm[0, 1]:>7,}")
    lines.append(f"  FN={cm[1, 0]:>7,}   TP={cm[1, 1]:>7,}")

    # Classification report
    lines.append(f"\nCLASSIFICATION REPORT")
    lines.append("-" * 40)
    lines.append(classification_report(
        y_true, best_preds, target_names=["Retained", "Churned"],
    ))

    lines.append("=" * 60)
    report = "\n".join(lines)

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"[evaluation] Report saved to {output_path}")

    return report
