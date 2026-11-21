"""FastAPI inference service with artifact loading, graceful fallback, and health checks."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from serving.schemas import ChurnPredictionRequest, ChurnPredictionResponse, HealthResponse
from src.data_pipeline import FEATURE_COLUMNS
from src.train import ARTIFACTS_DIR, MODEL_VERSION, run_training_pipeline

logger = logging.getLogger(__name__)


def classify_risk_tier(probability: float, threshold: float) -> str:
    """Assign actionable retention risk tier relative to the cost-tuned threshold."""
    low_cutoff = threshold * 0.65
    if probability >= threshold:
        return "High"
    if probability >= low_cutoff:
        return "Medium"
    return "Low"


class ModelRegistry:
    """Holds primary and fallback pipelines in memory with resilient fallback scoring."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = artifacts_dir
        self.primary_model: Any = None
        self.fallback_model: Any = None
        self.threshold: float = 0.50
        self.model_version: str = MODEL_VERSION
        self.force_primary_failure: bool = False

    def load(self, auto_train_if_missing: bool = True) -> None:
        """Load serialized model artifacts and threshold configuration."""
        primary_path = self.artifacts_dir / "churn_pipeline.joblib"
        fallback_path = self.artifacts_dir / "fallback_pipeline.joblib"
        config_path = self.artifacts_dir / "threshold_config.json"

        if (not primary_path.exists() or not fallback_path.exists()) and auto_train_if_missing:
            logger.warning("Artifacts missing at %s; running training pipeline to generate artifacts.", self.artifacts_dir)
            run_training_pipeline(self.artifacts_dir)

        if primary_path.exists():
            self.primary_model = joblib.load(primary_path)
        if fallback_path.exists():
            self.fallback_model = joblib.load(fallback_path)

        if config_path.exists():
            cfg: Dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
            self.threshold = float(cfg.get("optimal_threshold", 0.50))
            self.model_version = str(cfg.get("model_version", MODEL_VERSION))

    @staticmethod
    def _request_to_dataframe(payload: ChurnPredictionRequest) -> pd.DataFrame:
        data = payload.model_dump(exclude={"customer_id"})
        if data.get("TotalCharges") is None:
            data["TotalCharges"] = np.nan
        row = {col: data.get(col, np.nan) for col in FEATURE_COLUMNS}
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def predict_single(self, payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
        """Score a subscriber with primary model; gracefully fall back on any failure."""
        df = self._request_to_dataframe(payload)

        try:
            if self.force_primary_failure:
                raise RuntimeError("Simulated primary model scoring failure.")
            if self.primary_model is None:
                raise RuntimeError("Primary model pipeline is not loaded.")

            prob = float(self.primary_model.predict_proba(df)[0, 1])
            prob = float(np.clip(prob, 0.0, 1.0))
            pred = int(prob >= self.threshold)
            return ChurnPredictionResponse(
                customer_id=payload.customer_id,
                churn_probability=round(prob, 6),
                churn_prediction=pred,
                risk_tier=classify_risk_tier(prob, self.threshold),  # type: ignore[arg-type]
                threshold_used=self.threshold,
                model_source="primary_lightgbm",
                model_version=self.model_version,
                degraded_mode=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Primary scoring failed for customer_id=%s (%s); engaging fallback.", payload.customer_id, exc)
            return self._fallback_predict(payload, df)

    def _fallback_predict(self, payload: ChurnPredictionRequest, df: pd.DataFrame) -> ChurnPredictionResponse:
        """Compute degraded-mode prediction using fallback model or deterministic prior heuristic."""
        try:
            if self.fallback_model is not None:
                prob = float(self.fallback_model.predict_proba(df)[0, 1])
                source = "fallback_dummy_prior_model"
            else:
                # Deterministic heuristic fallback: month-to-month + short tenure raises risk modestly
                is_mtm = 1.0 if str(payload.Contract).lower() == "month-to-month" else 0.0
                tenure_factor = max(0.0, 1.0 - (float(payload.tenure) / 72.0))
                prob = float(np.clip(0.18 + 0.22 * is_mtm + 0.15 * tenure_factor, 0.05, 0.85))
                source = "fallback_prior_heuristic"
        except Exception:  # noqa: BLE001
            prob = 0.265
            source = "fallback_static_prior"

        prob = float(np.clip(prob, 0.0, 1.0))
        pred = int(prob >= self.threshold)
        return ChurnPredictionResponse(
            customer_id=payload.customer_id,
            churn_probability=round(prob, 6),
            churn_prediction=pred,
            risk_tier=classify_risk_tier(prob, self.threshold),  # type: ignore[arg-type]
            threshold_used=self.threshold,
            model_source=source,
            model_version=self.model_version,
            degraded_mode=True,
        )


registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    registry.load(auto_train_if_missing=True)
    yield


app = FastAPI(
    title="Customer Churn Prediction Service",
    version=MODEL_VERSION,
    description="Real-time churn scoring API with Pydantic v2 validation and graceful fallback.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return service readiness, loaded artifact status, and decision threshold."""
    if registry.primary_model is None and registry.fallback_model is None:
        registry.load(auto_train_if_missing=False)

    primary_ok = registry.primary_model is not None and not registry.force_primary_failure
    fallback_ok = registry.fallback_model is not None
    status = "ok" if primary_ok else ("degraded" if fallback_ok else "degraded")

    if not primary_ok and not fallback_ok:
        raise HTTPException(status_code=503, detail="Neither primary nor fallback model is available.")

    return HealthResponse(
        status=status,
        model_version=registry.model_version,
        primary_model_loaded=primary_ok,
        fallback_model_loaded=fallback_ok,
        decision_threshold=registry.threshold,
    )


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    """Score a single validated customer payload for churn probability and risk tier."""
    if registry.primary_model is None and registry.fallback_model is None:
        registry.load(auto_train_if_missing=True)
    return registry.predict_single(payload)