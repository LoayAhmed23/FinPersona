import numpy as np
import pandas as pd
import pytest
from conftest import load_project_module


def test_credit_create_target_supports_required_labeling_modes():
    feature_engineering = load_project_module(
        "train/credit/feature_engineering.py",
        "credit_feature_engineering_test",
    )

    df = pd.DataFrame(
        {
            "Card account status ": ["NORM", "30DD", "60DA", "90DA"],
        }
    )

    weighted = feature_engineering.create_target(df, mode="weighted_binary")
    assert weighted["target"].tolist() == [0, 1, 1, 1]
    assert weighted["sample_weight"].tolist() == [0.0, 0.4, 1.0, 1.0]

    hard_only = feature_engineering.create_target(df, mode="hard_only")
    assert hard_only["target"].tolist() == [0, 0, 1, 1]

    tiered = feature_engineering.create_target(df, mode="tiered")
    assert tiered["target"].tolist() == [0, 1, 2, 2]


def test_credit_temporal_features_keep_latest_snapshot_and_trends():
    feature_engineering = load_project_module(
        "train/credit/feature_engineering.py",
        "credit_feature_engineering_temporal_test",
    )

    df = pd.DataFrame(
        {
            "CUSTOMER_ID": [1, 1, 2, 2],
            "snapshot_month": pd.to_datetime(
                ["2026-02-01", "2026-03-01", "2026-02-01", "2026-03-01"]
            ),
            "utilization_ratio": [0.2, 0.5, 0.7, 0.6],
            "overdue_ratio": [0.0, 0.1, 0.2, 0.0],
            "txn_count": [2, 5, 4, 4],
        }
    )

    result = feature_engineering.engineer_temporal_features(df)

    assert result["CUSTOMER_ID"].tolist() == [1, 2]
    assert result["snapshot_month"].tolist() == [pd.Timestamp("2026-03-01")] * 2
    assert result["months_observed"].tolist() == [2, 2]
    assert result.loc[
        result["CUSTOMER_ID"] == 1, "utilization_ratio_delta"
    ].item() == pytest.approx(0.3)
    assert result.loc[
        result["CUSTOMER_ID"] == 2, "txn_count_trend"
    ].item() == pytest.approx(0.0)


def test_credit_evaluation_reports_core_binary_metrics():
    evaluation = load_project_module(
        "train/credit/evaluation.py",
        "credit_evaluation_test",
    )

    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.05, 0.2, 0.8, 0.95])
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = evaluation.evaluate(y_true, y_pred, y_proba, threshold=0.5)
    report = evaluation.generate_report(metrics, y_true, y_pred)

    assert metrics["AUC"] == pytest.approx(1.0)
    assert metrics["Recall"] == pytest.approx(1.0)
    assert metrics["Precision"] == pytest.approx(1.0)
    assert "CREDIT RISK MODEL" in report
    assert "CONFUSION MATRIX" in report


def test_credit_data_loader_extracts_months_from_file_names():
    data_loader = load_project_module(
        "train/credit/data_loader.py", "credit_data_loader_test"
    )

    assert data_loader._get_month_year_prime("APR2026_active.csv") == pd.Timestamp(
        "2026-04-01"
    )
    assert data_loader._get_month_year_prime("cleaned_MAY_2026.csv") == pd.Timestamp(
        "2026-05-01"
    )
    assert data_loader._get_month_year_txn("202603.csv") == pd.Timestamp("2026-03-01")


def test_credit_merge_data_is_customer_and_month_aware():
    data_loader = load_project_module(
        "train/credit/data_loader.py", "credit_merge_test"
    )

    prime_df = pd.DataFrame(
        {
            "CUSTOMER_ID": [1, 1, 2],
            "snapshot_month": pd.to_datetime(
                ["2026-02-01", "2026-03-01", "2026-03-01"]
            ),
            "CREDIT_LIMIT": [1000, 1100, 2000],
        }
    )
    txn_features = pd.DataFrame(
        {
            "txn_count": [2, 5],
            "txn_total_amount": [50.0, 70.0],
        },
        index=pd.MultiIndex.from_tuples(
            [(1, pd.Timestamp("2026-02-01")), (2, pd.Timestamp("2026-03-01"))],
            names=["CUSTOMER_ID", "snapshot_month"],
        ),
    )

    merged = data_loader.merge_data(prime_df, txn_features)

    assert merged["txn_count"].tolist() == [2.0, 0.0, 5.0]
    assert merged["txn_total_amount"].tolist() == [50.0, 0.0, 70.0]


def test_credit_preprocess_reuses_artifacts_and_handles_unseen_categories():
    preprocessing = load_project_module(
        "train/credit/preprocessing.py", "credit_preprocess_test"
    )

    train_df = pd.DataFrame(
        {
            "CUSTOMER_ID": [1, 2, 3],
            "segment": ["mass", "premium", None],
            "balance": [100.0, 200.0, None],
        }
    )
    y = pd.Series([0, 1, 0], index=train_df.index)

    X_train, y_train, artifacts = preprocessing.preprocess(train_df, y, fit=True)
    score_df = pd.DataFrame(
        {
            "CUSTOMER_ID": [4],
            "segment": ["new_segment"],
            "balance": [300.0],
        }
    )
    X_score, y_score, _ = preprocessing.preprocess(
        score_df, fit=False, artifacts=artifacts
    )

    assert "CUSTOMER_ID" not in X_train.columns
    assert y_train.tolist() == [0, 1, 0]
    assert y_score is None
    assert X_score.columns.tolist() == artifacts["feature_order"]
    assert X_score["segment"].tolist() == [-1]
    assert X_score["balance"].tolist() == [300.0]


def test_credit_drop_uncorrelated_features_drops_low_signal_columns():
    preprocessing = load_project_module(
        "train/credit/preprocessing.py",
        "credit_corr_filter_test",
    )

    X = pd.DataFrame(
        {
            "strong": [0, 0, 1, 1],
            "weak": [0, 1, 1, 0],
            "constant": [5, 5, 5, 5],
        }
    )
    y = pd.Series([0, 0, 1, 1])

    filtered, dropped = preprocessing.drop_uncorrelated_features(X, y, threshold=0.5)

    assert filtered.columns.tolist() == ["strong"]
    assert dropped == ["weak", "constant"]


def test_credit_leakage_check_flags_single_feature_proxy():
    pytest.importorskip("imblearn")
    pytest.importorskip("xgboost")
    pipeline = load_project_module("train/credit/pipeline.py", "credit_pipeline_test")

    X = pd.DataFrame(
        {
            "leaky_feature": [0, 0, 1, 1],
            "ordinary_feature": [0.1, 0.4, 0.2, 0.3],
        }
    )
    y = pd.Series([0, 0, 1, 1])

    suspects = pipeline.check_for_leakage(
        X,
        y,
        threshold=0.95,
        abort_on_leak=False,
    )

    assert suspects == [("leaky_feature", 1.0)]
