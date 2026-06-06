import numpy as np
import pandas as pd
import pytest
from conftest import load_project_module


def test_churn_evaluate_all_sorts_models_by_roc_auc():
    evaluation = load_project_module(
        "train/churn/evaluation.py", "churn_evaluation_test"
    )

    y_true = np.array([0, 0, 1, 1])
    results = {
        "weak": {
            "preds": np.array([0, 1, 0, 1]),
            "probs": np.array([0.45, 0.55, 0.40, 0.60]),
        },
        "strong": {
            "preds": np.array([0, 0, 1, 1]),
            "probs": np.array([0.05, 0.20, 0.80, 0.95]),
        },
    }

    metrics = evaluation.evaluate_all(y_true, results)

    assert metrics.index[0] == "strong"
    assert metrics.loc["strong", "ROC_AUC"] == pytest.approx(1.0)


def test_churn_generate_report_includes_best_model_and_confusion_matrix():
    evaluation = load_project_module("train/churn/evaluation.py", "churn_report_test")

    metrics_df = pd.DataFrame(
        {
            "Accuracy": [1.0],
            "Precision": [1.0],
            "Recall": [1.0],
            "F1": [1.0],
            "ROC_AUC": [1.0],
        },
        index=["Logistic Regression"],
    )

    report = evaluation.generate_report(
        metrics_df,
        y_true=np.array([0, 0, 1, 1]),
        best_name="Logistic Regression",
        best_preds=np.array([0, 0, 1, 1]),
    )

    assert "CHURN PREDICTION" in report
    assert "BEST MODEL: Logistic Regression" in report
    assert "TN=" in report
    assert "TP=" in report


def test_churn_label_file_loader_maps_raw_customer_id(tmp_path):
    data_loader = load_project_module(
        "train/churn/data_loader.py", "churn_data_loader_test"
    )

    csv_path = tmp_path / "FEB2026.csv"
    csv_path.write_text(
        "RIM_NO,Card account status ,OTHER\n" " 1001 ,NORM,x\n" " 1002 ,WROF,y\n",
        encoding="latin",
    )

    loaded = data_loader._load_label_file(
        str(csv_path),
        "202602",
        ["CUSTOMER_ID", "Card account status "],
    )

    assert loaded.columns.tolist() == ["CUSTOMER_ID", "Card account status "]
    assert loaded["CUSTOMER_ID"].tolist() == ["1001", "1002"]
    assert loaded["Card account status "].tolist() == ["NORM", "WROF"]


def test_churn_transaction_feature_engineering_aggregates_customer_behavior():
    feature_engineering = load_project_module(
        "train/churn/feature_engineering.py",
        "churn_feature_engineering_test",
    )

    txn_df = pd.DataFrame(
        {
            "CUSTOMER_ID": ["1", "1", "2"],
            "BILLING AMT": [100.0, 50.0, 25.0],
            "TRXN DATE": pd.to_datetime(["2026-05-01", "2026-05-10", "2026-05-10"]),
            "REVERSAL FLAG": [1, 0, 0],
        }
    )

    features = feature_engineering.engineer_transaction_features(txn_df)
    by_customer = features.set_index("CUSTOMER_ID")

    assert by_customer.loc["1", "total_spend"] == 150.0
    assert by_customer.loc["1", "avg_spend"] == 75.0
    assert by_customer.loc["1", "transaction_count"] == 2
    assert by_customer.loc["1", "recency_days"] == 0
    assert by_customer.loc["1", "reversal_ratio"] == 0.5
    assert by_customer.loc["2", "total_spend"] == 25.0


def test_churn_merge_all_defaults_missing_labels_to_non_churned():
    feature_engineering = load_project_module(
        "train/churn/feature_engineering.py",
        "churn_merge_all_test",
    )

    txn_features = pd.DataFrame(
        {
            "CUSTOMER_ID": ["1", "2"],
            "total_spend": [100.0, 50.0],
        }
    )
    prime_features = pd.DataFrame(
        {
            "CUSTOMER_ID": ["1", "2", "3"],
            "CREDIT_LIMIT": [1000.0, 2000.0, 3000.0],
        }
    )
    churn_labels = pd.DataFrame(
        {
            "CUSTOMER_ID": ["1"],
            "CHURN": [1],
        }
    )

    merged = feature_engineering.merge_all(txn_features, prime_features, churn_labels)

    assert merged["CUSTOMER_ID"].tolist() == ["1", "2"]
    assert merged["CHURN"].tolist() == [1, 0]
