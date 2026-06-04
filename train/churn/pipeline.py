"""
Pipeline orchestrator — ties data loading, feature engineering,
preprocessing, training, and evaluation together.
"""

import os

import numpy as np
from sklearn.model_selection import train_test_split

import config
from data_loader import create_churn_labels, load_transaction_data, load_prime_data
from feature_engineering import (
    engineer_transaction_features,
    engineer_prime_features,
    merge_all,
)
from preprocessing import preprocess
from model import train_classifiers, tune_models, save_model
from evaluation import evaluate_all, generate_report


def _ensure_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def _banner(step, total, title):
    print()
    print("=" * 60)
    print(f"  STEP {step}/{total}: {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(tune: bool = False):
    """Full training pipeline: label → load → engineer → preprocess → train → evaluate.

    Parameters
    ----------
    tune : bool
        If True, run GridSearchCV for RF and XGBoost after the initial
        multi-classifier comparison.
    """
    TOTAL = 8 if tune else 7
    _ensure_output_dir()

    # ------------------------------------------------------------------
    _banner(1, TOTAL, "CHURN LABELING")
    # ------------------------------------------------------------------
    churn_labels = create_churn_labels()

    # ------------------------------------------------------------------
    _banner(2, TOTAL, "LOADING DATA")
    # ------------------------------------------------------------------
    txn_df = load_transaction_data()
    prime_df = load_prime_data()

    # ------------------------------------------------------------------
    _banner(3, TOTAL, "FEATURE ENGINEERING")
    # ------------------------------------------------------------------
    txn_features = engineer_transaction_features(txn_df)
    prime_features = engineer_prime_features(prime_df)

    # ------------------------------------------------------------------
    _banner(4, TOTAL, "MERGING DATASETS")
    # ------------------------------------------------------------------
    final_df = merge_all(txn_features, prime_features, churn_labels)

    # ------------------------------------------------------------------
    _banner(5, TOTAL, "PREPROCESSING")
    # ------------------------------------------------------------------
    X, y, artifacts = preprocess(final_df, fit=True)

    print(f"  Features:     {X.shape[1]}")
    print(f"  Samples:      {X.shape[0]:,}")
    print(f"  Churn rate:   {y.mean() * 100:.2f}%")

    # ------------------------------------------------------------------
    _banner(6, TOTAL, "TRAIN / TEST SPLIT")
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")

    # ------------------------------------------------------------------
    step = 7
    _banner(step, TOTAL, "MODEL TRAINING (all classifiers)")
    # ------------------------------------------------------------------
    results = train_classifiers(X_train, y_train, X_test, y_test)

    # Evaluate all models
    metrics_df = evaluate_all(y_test, results)
    print("\n  Model Comparison:")
    print(metrics_df.to_string())

    # Pick best model by ROC AUC
    best_name = metrics_df.index[0]
    best_result = results[best_name]
    best_model = best_result["model"]

    # ------------------------------------------------------------------
    # Optional: hyperparameter tuning
    # ------------------------------------------------------------------
    if tune:
        step += 1
        _banner(step, TOTAL, "HYPERPARAMETER TUNING (GridSearchCV)")
        tuned = tune_models(X_train, y_train)

        # Evaluate tuned models on test set
        for name, tuned_info in tuned.items():
            model = tuned_info["model"]
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            results[f"{name} (tuned)"] = {
                "model": model, "preds": preds, "probs": probs,
            }

        metrics_df = evaluate_all(y_test, results)
        print("\n  Updated Model Comparison (with tuned):")
        print(metrics_df.to_string())

        best_name = metrics_df.index[0]
        best_result = results[best_name]
        best_model = best_result["model"]

    # ------------------------------------------------------------------
    _banner(step + 1 if tune else step + 1, TOTAL + 1 if tune else TOTAL + 1,
            "EVALUATION & OUTPUT")
    # (adjust numbering — always the final step)
    # ------------------------------------------------------------------
    report = generate_report(
        metrics_df, y_test, best_name, best_result["preds"],
        output_path=config.REPORT_PATH,
    )
    print(report)

    # Save best model
    save_model(best_model, artifacts)

    # Save churn scores for all customers
    all_probs = best_model.predict_proba(X)[:, 1]
    all_preds = (all_probs >= config.THRESHOLD).astype(int)

    scores_df = __import__("pandas").DataFrame({
        config.CUSTOMER_ID: final_df[config.CUSTOMER_ID].values,
        "churn_probability": all_probs,
        "predicted_churn": all_preds,
    })
    scores_df.to_csv(config.SCORES_PATH, index=False)
    print(f"  Churn scores saved to {config.SCORES_PATH}")

    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    return metrics_df
