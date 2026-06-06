import numpy as np
import pandas as pd
import pytest
from conftest import load_project_module


def test_recommendation_preprocessing_fills_and_encodes_features():
    preprocessing = load_project_module(
        "train/recommendation/preprocessing.py",
        "recommendation_preprocessing_test",
    )

    profile = pd.DataFrame(
        {
            "CUSTOMER_ID": [1, 2],
            "TOTAL_SPEND_AMT": [None, 150.0],
            "DAYS_SINCE_LAST_TRXN": [None, 4],
            "AGE_GROUP": ["25-34", "35-44"],
            "GENDER": ["F", "M"],
            "BRANCH_ID": [10, 20],
        }
    )

    filled = preprocessing.fill_missing_values(profile.copy())
    encoded = preprocessing.one_hot_encode_categoricals(filled)

    assert filled["TOTAL_SPEND_AMT"].tolist() == [0.0, 150.0]
    assert filled["DAYS_SINCE_LAST_TRXN"].tolist() == [9999.0, 4.0]
    assert "AGE_GROUP_25_34" in encoded.columns
    assert "GENDER_F" in encoded.columns
    assert "BRANCH_ID_10" in encoded.columns
    assert "AGE_GROUP" not in encoded.columns


def test_recommendation_feature_engineering_builds_user_item_matrix():
    feature_engineering = load_project_module(
        "train/recommendation/feature_engineering.py",
        "recommendation_feature_engineering_test",
    )

    prime_df = pd.DataFrame(
        {
            "CUSTOMER_ID": [1, 1, 2],
            "PRODUCT_NAME": ["Credit Card", "Loan", "Credit Card"],
            "GENDER": ["F", "F", "M"],
        }
    )
    logs = []

    profile, user_item = feature_engineering.build_user_item_matrix(prime_df, logs)

    assert profile["CUSTOMER_ID"].tolist() == [1, 2]
    assert profile["HAS_PROD_Credit_Card"].tolist() == [1, 1]
    assert profile["HAS_PROD_Loan"].tolist() == [1, 0]
    assert user_item.loc[1, "HAS_PROD_Loan"] == 1
    assert any("Built user-item matrix" in line for line in logs)


def test_recommendation_feature_engineering_aggregates_transaction_features():
    feature_engineering = load_project_module(
        "train/recommendation/feature_engineering.py",
        "recommendation_transaction_features_test",
    )

    txn_df = pd.DataFrame(
        {
            "CUSTOMER_ID": [1, 1, 2],
            "BILLING AMT": [100.0, 50.0, 25.0],
            "TRXN DATE": pd.to_datetime(["2026-05-01", "2026-05-10", "2026-05-10"]),
            "MCC": [5411, 5812, 5411],
            "TRXN COUNTRY": ["EGYPT", "UAE", "EGYPT"],
        }
    )

    rfm = feature_engineering.build_rfm_features(txn_df).set_index("CUSTOMER_ID")
    mcc = feature_engineering.build_mcc_spend(txn_df).set_index("CUSTOMER_ID")
    foreign = feature_engineering.build_foreign_trxn_features(txn_df).set_index(
        "CUSTOMER_ID"
    )

    assert rfm.loc[1, "TOTAL_SPEND_AMT"] == 150.0
    assert rfm.loc[1, "AVG_TRXN_AMT"] == 75.0
    assert rfm.loc[1, "TRXN_COUNT"] == 2
    assert mcc.loc[1, "MCC_5411_SPEND"] == 100.0
    assert mcc.loc[1, "MCC_5812_SPEND"] == 50.0
    assert foreign.loc[1, "FOREIGN_TRXN_COUNT"] == 1


def test_recommendation_threshold_optimization_and_metrics():
    evaluation = load_project_module(
        "train/recommendation/evaluation.py",
        "recommendation_evaluation_test",
    )

    valid_targets = ["HAS_PROD_CARD", "HAS_PROD_LOAN"]
    y_val = pd.DataFrame(
        {
            "HAS_PROD_CARD": [0, 1, 1, 0],
            "HAS_PROD_LOAN": [1, 0, 1, 0],
        }
    )
    proba = {
        "HAS_PROD_CARD": np.array([0.1, 0.8, 0.7, 0.2]),
        "HAS_PROD_LOAN": np.array([0.9, 0.3, 0.75, 0.1]),
    }

    thresholds = evaluation.optimize_xgboost_thresholds(valid_targets, proba, y_val)
    y_pred = pd.DataFrame(
        {
            product: (proba[product] >= thresholds[product]).astype(int)
            for product in valid_targets
        }
    )
    metrics, exact_acc, micro_f1, macro_f1 = evaluation.evaluate_xgboost_metrics(
        y_val,
        y_pred,
        valid_targets,
        y_val,
        thresholds,
    )

    assert thresholds == {"HAS_PROD_CARD": 0.25, "HAS_PROD_LOAN": 0.35}
    assert metrics["exact_match_accuracy"] == 100.0
    assert exact_acc == pytest.approx(1.0)
    assert micro_f1 == pytest.approx(1.0)
    assert macro_f1 == pytest.approx(1.0)
    assert metrics["threshold_table"][0]["product"] == "CARD"


def test_recommendation_filter_uncorrelated_features_keeps_predictive_columns():
    preprocessing = load_project_module(
        "train/recommendation/preprocessing.py",
        "recommendation_filter_test",
    )
    preprocessing.config.CORR_THRESHOLD = 0.5

    profile = pd.DataFrame(
        {
            "CUSTOMER_ID": [1, 2, 3, 4],
            "strong_feature": [0, 0, 1, 1],
            "weak_feature": [0, 1, 1, 0],
            "HAS_PROD_CARD": [0, 0, 1, 1],
        }
    )
    logs = []

    filtered, feature_cols, product_cols = preprocessing.filter_uncorrelated_features(
        profile,
        logs,
    )

    assert "strong_feature" in feature_cols
    assert "weak_feature" not in filtered.columns
    assert product_cols == ["HAS_PROD_CARD"]
    assert any("Dropping" in line for line in logs)


def test_recommendation_cbf_metrics_use_product_thresholds():
    evaluation = load_project_module(
        "train/recommendation/evaluation.py",
        "recommendation_cbf_metrics_test",
    )

    scores = np.array(
        [
            [0.8, 0.1],
            [0.2, 0.9],
            [0.7, 0.6],
        ]
    )
    truth = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    test_user_mask = np.array([True, True, True])
    thresholds = {"HAS_PROD_CARD": 0.5, "HAS_PROD_LOAN": 0.5}

    metrics, exact_acc, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1 = (
        evaluation.evaluate_cbf_metrics(
            scores,
            truth,
            test_user_mask,
            thresholds,
            ["HAS_PROD_CARD", "HAS_PROD_LOAN"],
        )
    )

    assert metrics["exact_match_accuracy"] == 100.0
    assert exact_acc == pytest.approx(1.0)
    assert micro_p == pytest.approx(1.0)
    assert micro_r == pytest.approx(1.0)
    assert micro_f1 == pytest.approx(1.0)
    assert macro_p == pytest.approx(1.0)
    assert macro_r == pytest.approx(1.0)
    assert macro_f1 == pytest.approx(1.0)
