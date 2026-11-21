"""Strict Pydantic v2 request and response contracts for churn inference."""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChurnPredictionRequest(BaseModel):
    """Validated subscriber input payload for real-time churn scoring."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    customer_id: str = Field(..., min_length=1, max_length=64, description="Unique subscriber identifier")
    gender: Optional[str] = Field(default="Female", description="Subscriber gender")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="Whether subscriber is a senior citizen (0 or 1)")
    Partner: Optional[str] = Field(default="No", description="Whether subscriber has a partner")
    Dependents: Optional[str] = Field(default="No", description="Whether subscriber has dependents")
    tenure: float = Field(..., ge=0.0, le=120.0, description="Months with the provider (0 to 120)")
    PhoneService: Optional[str] = Field(default="Yes", description="Whether subscriber has phone service")
    MultipleLines: Optional[str] = Field(default="No", description="Whether subscriber has multiple lines")
    InternetService: Optional[str] = Field(default="DSL", description="Internet service type")
    OnlineSecurity: Optional[str] = Field(default="No", description="Online security add-on status")
    OnlineBackup: Optional[str] = Field(default="No", description="Online backup add-on status")
    DeviceProtection: Optional[str] = Field(default="No", description="Device protection add-on status")
    TechSupport: Optional[str] = Field(default="No", description="Tech support add-on status")
    StreamingTV: Optional[str] = Field(default="No", description="Streaming TV add-on status")
    StreamingMovies: Optional[str] = Field(default="No", description="Streaming movies add-on status")
    Contract: Optional[str] = Field(default="Month-to-month", description="Contract term")
    PaperlessBilling: Optional[str] = Field(default="Yes", description="Paperless billing flag")
    PaymentMethod: Optional[str] = Field(default="Electronic check", description="Payment method")
    MonthlyCharges: float = Field(..., ge=0.0, le=1000.0, description="Current monthly charge in USD")
    TotalCharges: Optional[float] = Field(default=None, ge=0.0, le=100000.0, description="Cumulative charges in USD")

    @field_validator("tenure", "MonthlyCharges")
    @classmethod
    def validate_finite_float(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Numeric fields must be finite.")
        return float(v)

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if math.isnan(v) or math.isinf(v):
            raise ValueError("TotalCharges must be finite when provided.")
        return float(v)

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("customer_id cannot be blank.")
        return v.strip()


class ChurnPredictionResponse(BaseModel):
    """Structured prediction response returned by POST /predict."""

    customer_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: int = Field(..., ge=0, le=1)
    risk_tier: Literal["Low", "Medium", "High"]
    threshold_used: float = Field(..., ge=0.0, le=1.0)
    model_source: str
    model_version: str
    degraded_mode: bool = False


class HealthResponse(BaseModel):
    """Service health and artifact readiness payload for GET /health."""

    status: Literal["ok", "degraded"]
    model_version: str
    primary_model_loaded: bool
    fallback_model_loaded: bool
    decision_threshold: float