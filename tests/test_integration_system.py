from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import PROJECT_ROOT, load_project_module


def test_unified_launcher_declares_all_expected_services():
    launcher = load_project_module("main.py", "finpersona_launcher_test")

    services = {service["name"]: service for service in launcher.SERVERS}

    assert set(services) == {"Recommendation", "Credit Risk", "Churn", "Gateway (UI)"}
    assert {service["port"] for service in services.values()} == {
        5000,
        5005,
        5006,
        5050,
    }
    for service in services.values():
        script = Path(service["cwd"]) / service["script"]
        assert script.exists(), f"Missing configured server script: {script}"


def test_gateway_proxy_returns_502_when_backend_is_unreachable(monkeypatch):
    pytest.importorskip("flask")
    pytest.importorskip("requests")
    gateway = load_project_module("UI/app.py", "gateway_app_test")

    def raise_connection_error(**_kwargs):
        raise gateway.http_requests.ConnectionError("backend down")

    monkeypatch.setattr(gateway.http_requests, "request", raise_connection_error)

    with gateway.app.test_request_context("/credit/api/status", method="GET"):
        response, status_code = gateway._proxy("credit", "api/status")

    assert status_code == 502
    assert "Cannot reach credit server" in response.get_json()["error"]


def test_recommendation_batch_prediction_integration_writes_expected_csv(
    tmp_path, monkeypatch
):
    pytest.importorskip("xgboost")
    pipeline = load_project_module(
        "train/recommendation/pipeline.py",
        "recommendation_pipeline_integration_test",
    )

    output_path = tmp_path / "recommendations.csv"
    monkeypatch.setattr(pipeline.config, "BATCH_OUTPUT_PATH", str(output_path))

    prime_df = pd.DataFrame(
        {
            "CUSTOMER_ID": [1, 2],
            "RIMNO": ["R001", "R002"],
            "GENDER": ["F", "M"],
        }
    )

    monkeypatch.setattr(
        pipeline, "load_prime_data", lambda _path, _logs: prime_df.copy()
    )
    monkeypatch.setattr(
        pipeline, "load_transaction_data", lambda _path, _logs: pd.DataFrame()
    )
    monkeypatch.setattr(
        pipeline,
        "build_user_item_matrix",
        lambda df, _logs: (
            df,
            pd.DataFrame({"HAS_PROD_CARD": [0, 0]}, index=[1, 2]),
        ),
    )
    monkeypatch.setattr(pipeline, "build_rfm_features", lambda _df: pd.DataFrame())
    monkeypatch.setattr(pipeline, "build_mcc_spend", lambda _df: pd.DataFrame())
    monkeypatch.setattr(
        pipeline, "build_foreign_trxn_features", lambda _df: pd.DataFrame()
    )
    monkeypatch.setattr(pipeline, "build_demographics_features", lambda df: df)
    monkeypatch.setattr(
        pipeline,
        "merge_all_features",
        lambda *_args: pd.DataFrame(
            {
                "CUSTOMER_ID": [1, 2],
                "RIMNO": ["R001", "R002"],
                "GENDER": ["F", "M"],
                "feature_a": [10.0, 20.0],
            }
        ),
    )

    class PredictsCardForFirstCustomer:
        def predict_proba(self, X):
            assert list(X.columns) == ["feature_a", "missing_training_feature"]
            return np.array([[0.1, 0.8], [0.9, 0.1]])

    results_df, saved_path, logs = pipeline.predict_new_data(
        prime_dir=str(PROJECT_ROOT / "data" / "prime_cleaned"),
        transaction_dir=str(PROJECT_ROOT / "data" / "transaction_cleaned"),
        models_dict={"HAS_PROD_CARD": PredictsCardForFirstCustomer()},
        optimal_thresholds={"HAS_PROD_CARD": 0.5},
        trained_feature_cols=["feature_a", "missing_training_feature"],
        valid_targets=["HAS_PROD_CARD"],
        logs=[],
    )

    assert saved_path == str(output_path)
    assert output_path.exists()
    assert results_df.to_dict("records") == [
        {
            "RIMNO": "R001",
            "CUSTOMER_ID": 1,
            "PRODUCT_NAME": "CARD",
            "PROBABILITY": 80.0,
            "THRESHOLD": 50.0,
            "MODEL": "XGBoost",
        }
    ]
    assert "BATCH PREDICTION SUMMARY" in "\n".join(logs)
