"""Comprehensive automated test suite for pipeline, API contracts, edge cases, batch CLI, and drift."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from serving.app import app, registry
from serving.predict_batch import run_batch_cli
from src.data_pipeline import load_prepared_data
from src.drift_monitor import run_drift_simulation
from src.evaluate import run_evaluation
from src.train import run_training_pipeline


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train models into an isolated temporary artifact directory once per test module."""
    art_dir = tmp_path_factory.mktemp("artifacts")
    run_training_pipeline(output_dir=art_dir)
    registry.artifacts_dir = art_dir
    registry.force_primary_failure = False
    registry.load(auto_train_if_missing=False)
    return art_dir


@pytest.fixture()
def client(trained_artifacts: Path) -> TestClient:  # noqa: ARG001
    registry.force_primary_failure = False
    with TestClient(app) as test_client:
        yield test_client


def _valid_payload(**overrides: object) -> dict:
    base = {
        "customer_id": "CUST-1001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 6.0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 88.50,
        "TotalCharges": 531.00,
    }
    base.update(overrides)
    return base


def test_leak_free_stratified_split_and_audit() -> None:
    splits = load_prepared_data()
    total_rows = len(splits.X_train) + len(splits.X_val) + len(splits.X_test)
    assert abs(len(splits.X_train) / total_rows - 0.70) < 0.02
    assert abs(len(splits.X_val) / total_rows - 0.15) < 0.02
    assert abs(len(splits.X_test) / total_rows - 0.15) < 0.02
    assert splits.leakage_audit["audit_passed"] is True


def test_health_and_valid_prediction(client: TestClient) -> None:
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data["status"] == "ok"
    assert health_data["primary_model_loaded"] is True
    assert health_data["fallback_model_loaded"] is True

    pred_resp = client.post("/predict", json=_valid_payload())
    assert pred_resp.status_code == 200
    body = pred_resp.json()
    assert body["customer_id"] == "CUST-1001"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_tier"] in ("Low", "Medium", "High")
    assert body["degraded_mode"] is False


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"customer_id": "BAD-1", "tenure": -5.0, "MonthlyCharges": 50.0},  # negative tenure
        {"customer_id": "BAD-2", "tenure": 10.0, "MonthlyCharges": -10.0},  # negative charge
        {"customer_id": "BAD-3", "tenure": 10.0, "MonthlyCharges": 50.0, "SeniorCitizen": 5},  # invalid flag
        {"customer_id": "BAD-4", "tenure": 10.0, "MonthlyCharges": 50.0, "unknown_extra_field": 123},  # extra field
        {"tenure": 10.0, "MonthlyCharges": 50.0},  # missing required customer_id
    ],
)
def test_malformed_payloads_return_422(client: TestClient, bad_payload: dict) -> None:
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_missing_values_unseen_categories_and_outliers(client: TestClient) -> None:
    missing_and_unseen_payload = _valid_payload(
        customer_id="EDGE-MISSING-UNSEEN",
        TotalCharges=None,
        InternetService="Satellite-5G-Unseen",
        PaymentMethod="Crypto-Wallet-Unseen",
    )
    resp_missing = client.post("/predict", json=missing_and_unseen_payload)
    assert resp_missing.status_code == 200
    body_missing = resp_missing.json()
    assert 0.0 <= body_missing["churn_probability"] <= 1.0
    assert body_missing["degraded_mode"] is False

    # 2. Extreme numeric outliers within valid schema bounds
    outlier_payload = _valid_payload(
        customer_id="EDGE-OUTLIER",
        tenure=118.0,
        MonthlyCharges=950.0,
        TotalCharges=98000.0,
    )
    resp_outlier = client.post("/predict", json=outlier_payload)
    assert resp_outlier.status_code == 200
    body_outlier = resp_outlier.json()
    assert 0.0 <= body_outlier["churn_probability"] <= 1.0
    assert body_outlier["degraded_mode"] is False


def test_graceful_fallback_on_primary_failure(client: TestClient) -> None:
    registry.force_primary_failure = True
    try:
        resp = client.post("/predict", json=_valid_payload(customer_id="FALLBACK-1"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["degraded_mode"] is True
        assert body["model_source"].startswith("fallback_")
        assert 0.0 <= body["churn_probability"] <= 1.0
    finally:
        registry.force_primary_failure = False


def test_batch_cli_scoring(trained_artifacts: Path, tmp_path: Path) -> None:
    splits = load_prepared_data()
    sample_df = splits.X_test.head(25).copy()
    sample_df.insert(0, "customerID", splits.ids_test.head(25).to_numpy())

    in_csv = tmp_path / "batch_in.csv"
    out_csv = tmp_path / "batch_out.csv"
    sample_df.to_csv(in_csv, index=False)

    stats = run_batch_cli(in_csv, out_csv, artifacts_dir=trained_artifacts)
    assert out_csv.exists()
    assert stats["rows_scored"] == 25
    assert stats["throughput_records_per_sec"] > 0

    scored_df = pd.read_csv(out_csv)
    assert list(scored_df["customerID"]) == list(sample_df["customerID"])
    assert {"churn_probability", "churn_prediction", "risk_tier", "degraded_mode"}.issubset(scored_df.columns)


def test_evaluation_and_drift_monitor(trained_artifacts: Path) -> None:
    splits = load_prepared_data()
    report, plots = run_evaluation(splits=splits, artifacts_dir=trained_artifacts)
    assert "lightgbm_tuned" in report["test_models"]
    assert Path(plots["pr_curve"]).exists()
    assert Path(plots["confusion_matrix"]).exists()
    assert Path(plots["threshold_cost_curve"]).exists()

    drift = run_drift_simulation(output_dir=trained_artifacts)
    assert drift["synthetic_covariate_shift"]["retraining_recommended"] is True
    assert drift["synthetic_covariate_shift"]["alert_count"] >= 1