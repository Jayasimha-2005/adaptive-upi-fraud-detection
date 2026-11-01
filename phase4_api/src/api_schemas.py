"""
phase4_api/src/api_schemas.py
=============================
Pydantic v2 Request & Response Schemas for Phase 7 FastAPI Serving.

Part of Member 3 — API Serving for Adaptive UPI / BAF Fraud Detection.

Defines:
  1. SingleTransactionRequest: Schema for POST /api/v1/predict/single
  2. BatchTransactionRequest: Schema for POST /api/v1/predict/batch
  3. HealthResponse: Schema for GET /health
  4. ModelInfoResponse: Schema for GET /api/v1/model/info
  5. StandardizedPredictionResponse: Schema for single prediction responses
  6. BatchPredictionResponseModel: Schema for batch prediction responses
  7. ErrorResponseModel: Standardized client error representation
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from phase4_api.src.batch_inference_engine import DEFAULT_MAX_BATCH_SIZE
from phase4_api.src.prediction_schema import ReasonCodeModel


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------
class SingleTransactionRequest(BaseModel):
    """
    Request payload for a single BAF transaction.
    Enforces presence of all 29 canonical features and optional transaction_id.
    Strictly forbids unexpected extra features.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional client or system transaction identifier",
    )

    # 24 Canonical Numerical Features
    income: float = Field(..., description="Income level")
    name_email_similarity: float = Field(..., description="Name-email similarity score")
    prev_address_months_count: float = Field(..., description="Months at previous address")
    current_address_months_count: float = Field(..., description="Months at current address")
    customer_age: float = Field(..., description="Customer age")
    days_since_request: float = Field(..., description="Days since application request")
    intended_balcon_amount: float = Field(..., description="Intended initial balance transfer amount")
    zip_count_4w: float = Field(..., description="Zip code application count in past 4 weeks")
    velocity_6h: float = Field(..., description="Application velocity in past 6 hours")
    velocity_24h: float = Field(..., description="Application velocity in past 24 hours")
    velocity_4w: float = Field(..., description="Application velocity in past 4 weeks")
    bank_branch_count_8w: float = Field(..., description="Bank branch count in past 8 weeks")
    date_of_birth_distinct_emails_4w: float = Field(..., description="Distinct emails with same DOB in 4 weeks")
    credit_risk_score: float = Field(..., description="Internal credit risk score")
    email_is_free: float = Field(..., description="Binary flag indicating free email domain")
    phone_home_valid: float = Field(..., description="Binary flag indicating valid home phone")
    phone_mobile_valid: float = Field(..., description="Binary flag indicating valid mobile phone")
    bank_months_count: float = Field(..., description="Months with current bank")
    has_other_cards: float = Field(..., description="Binary flag for other credit cards")
    proposed_credit_limit: float = Field(..., description="Proposed credit limit")
    foreign_request: float = Field(..., description="Binary flag for foreign IP/origin")
    session_length_in_minutes: float = Field(..., description="Web/mobile session length in minutes")
    keep_alive_session: float = Field(..., description="Binary flag indicating active keep-alive")
    device_distinct_emails_8w: float = Field(..., description="Distinct emails observed on device in 8 weeks")

    # 5 Canonical Categorical Features
    payment_type: str = Field(..., description="Payment method category (AA, AB, AC, AD, AE)")
    employment_status: str = Field(..., description="Employment category (CA..CG)")
    housing_status: str = Field(..., description="Housing status category (BA..BG)")
    source: str = Field(..., description="Application origin channel (INTERNET, TELEAPP)")
    device_os: str = Field(..., description="Device operating system (windows, macintosh, linux, x11, other)")

    @field_validator("transaction_id")
    @classmethod
    def validate_tid(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("transaction_id cannot be an empty or whitespace-only string.")
        return v.strip() if v is not None else None


class BatchTransactionRequest(BaseModel):
    """
    Request payload for batch inference.
    Accepts an ordered list of transaction records.
    """

    model_config = ConfigDict(extra="forbid")

    transactions: List[Any] = Field(
        ...,
        description="Ordered list of transaction dictionaries (minimum 1)",
    )

    @field_validator("transactions")
    @classmethod
    def validate_transactions_list(cls, v: List[Any]) -> List[Any]:
        if not isinstance(v, list):
            raise ValueError("transactions must be a list.")
        if len(v) == 0:
            raise ValueError("transactions batch cannot be empty. Must contain at least 1 record.")
        return v


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------
class SinglePredictionResponse(BaseModel):
    """Structured response for single-transaction prediction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = Field(..., description="Execution status: 'SUCCESS' or 'ERROR'")
    transaction_id: Optional[str] = Field(None, description="Transaction identifier")
    prediction: str = Field(..., description="Authoritative decision: 'FRAUD' or 'LEGIT'")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="LightGBM model fraud probability")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Authoritative frozen threshold (0.8964321608040201)")
    risk_level: str = Field(..., description="Descriptive risk tier: LOW, MEDIUM, HIGH, CRITICAL")
    model_version: str = Field(..., description="Active model version tag, e.g. 'v3'")
    model_status: str = Field(..., description="Operational status, e.g. 'ACTIVE_LOADED'")
    feature_count: int = Field(..., description="Canonical feature count (29)")
    reason_codes: List[ReasonCodeModel] = Field(default_factory=list, description="Ordered auditable explanation tokens")
    explanation_metadata: Optional[Dict[str, Any]] = Field(None, description="Operational metadata and risk cutoffs")
    execution_time_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds")


class BatchRecordResponse(BaseModel):
    """Individual record outcome within a batch response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_index: int = Field(..., description="0-based original sequence position in batch")
    transaction_id: Optional[str] = Field(None, description="Transaction identifier")
    status: str = Field(..., description="Record outcome: 'success' or 'error'")
    prediction: Optional[str] = Field(None, description="'FRAUD', 'LEGIT', or null on error")
    fraud_probability: Optional[float] = Field(None, description="Fraud probability or null on error")
    threshold: Optional[float] = Field(None, description="Authoritative frozen threshold or null on error")
    risk_level: Optional[str] = Field(None, description="Descriptive risk tier or null on error")
    model_version: Optional[str] = Field(None, description="Model version tag or null on error")
    feature_count: Optional[int] = Field(None, description="Canonical feature count or null on error")
    reason_codes: Optional[List[ReasonCodeModel]] = Field(None, description="Auditable reason codes or null on error")
    explanation_metadata: Optional[Dict[str, Any]] = Field(None, description="Explanation metadata or null on error")
    error: Optional[Dict[str, str]] = Field(None, description="Error details if status is 'error'")


class BatchPredictionResponseModel(BaseModel):
    """Aggregated batch prediction response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_status: str = Field(..., description="Overall status: 'all_success', 'partial_success', 'all_failed'")
    total_transactions: int = Field(..., ge=0, description="Total transactions in batch")
    successful_transactions: int = Field(..., ge=0, description="Number of successfully predicted transactions")
    failed_transactions: int = Field(..., ge=0, description="Number of failed transactions")
    model_version: str = Field(..., description="Active model version tag")
    threshold: float = Field(..., description="Authoritative frozen threshold")
    results: List[BatchRecordResponse] = Field(..., description="Ordered list of per-record results")
    execution_time_ms: float = Field(..., ge=0.0, description="Total batch inference latency in milliseconds")


class HealthResponse(BaseModel):
    """Service health and readiness telemetry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = Field(..., description="'healthy', 'degraded', or 'unhealthy'")
    model_loaded: bool = Field(..., description="True if LightGBM booster is actively resident in memory")
    ready: bool = Field(..., description="True if inference engine is fully operational")
    model_version: str = Field(..., description="Active model version tag")
    frozen_threshold: float = Field(..., description="Authoritative frozen threshold")
    uptime_seconds: float = Field(..., ge=0.0, description="Service uptime in seconds")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")


class ModelInfoResponse(BaseModel):
    """Safe model metadata and configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_version: str = Field(..., description="Model version tag ('v3')")
    model_status: str = Field(..., description="Operational status ('ACTIVE_LOADED')")
    model_type: str = Field(..., description="Model architecture ('LightGBM Booster')")
    dataset: str = Field(..., description="Benchmark dataset ('Bank Account Fraud - BAF Base')")
    feature_count: int = Field(..., description="Number of canonical features (29)")
    num_trees: Optional[int] = Field(None, description="Number of decision trees (500)")
    frozen_threshold: float = Field(..., description="Authoritative frozen threshold (0.8964321608040201)")
    supported_inference_modes: List[str] = Field(..., description="Supported modes: ['single', 'batch']")
    risk_policy_version: str = Field(..., description="Active risk policy version tag")
    risk_cutoffs: Dict[str, List[float]] = Field(..., description="Configured risk probability bands")


class ErrorDetail(BaseModel):
    """Structured error payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Safe human-readable error description")
    details: Optional[List[Any]] = Field(default=None, description="Optional granular validation details")


class ErrorResponseModel(BaseModel):
    """Centralized error envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    error: ErrorDetail
