"""
phase4_api/src/prediction_schema.py
===================================
Standardized Prediction Result Schema & Validation Models.

Part of Member 3 — API Serving for Adaptive UPI / BAF Fraud Detection.

Provides:
  1. StandardizedPredictionResult: Canonical, fully JSON-serializable container
     compatible with both single-transaction and batch-transaction inference.
  2. PredictionResultModel: Pydantic v2 validation schema preparing for Phase 7
     FastAPI integration without requiring external endpoints.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from phase4_api.src.risk_classifier import (
    ReasonCode,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class PredictionSchemaError(Exception):
    """Base error for prediction schema validation or serialization failures."""


class InvalidSchemaDataError(PredictionSchemaError):
    """Raised when data fails schema validation rules."""


# ---------------------------------------------------------------------------
# Pydantic v2 Models for Validation & Phase 7 Readiness
# ---------------------------------------------------------------------------
class ReasonCodeModel(BaseModel):
    """Pydantic model for a single structured reason code."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(..., description="Stable, unique reason code token")
    description: str = Field(..., description="Human-readable description of the reason")
    category: str = Field(..., description="Category group of the reason code")


class PredictionResultModel(BaseModel):
    """
    Pydantic v2 schema for standardized prediction results.
    Validates types, ranges, and invariants across both single and batch records.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    transaction_id: Optional[str] = Field(None, description="Client or generated transaction identifier")
    prediction: Optional[str] = Field(None, description="'FRAUD' or 'LEGIT' decision based on frozen threshold")
    fraud_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="LightGBM model fraud probability")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Authoritative frozen threshold")
    risk_level: Optional[str] = Field(None, description="Descriptive risk tier: LOW, MEDIUM, HIGH, CRITICAL")
    model_version: Optional[str] = Field(None, description="Model version tag, e.g. 'v3'")
    model_status: Optional[str] = Field(None, description="Operational status, e.g. 'ACTIVE_LOADED'")
    feature_count: Optional[int] = Field(None, ge=1, description="Number of model features, canonical 29")
    reason_codes: List[ReasonCodeModel] = Field(default_factory=list, description="Ordered auditable reason codes")
    explanation_metadata: Optional[Dict[str, Any]] = Field(None, description="Operational context metadata")
    execution_time_ms: Optional[float] = Field(None, ge=0.0, description="Inference latency in milliseconds")
    status: str = Field(..., description="'SUCCESS' or 'ERROR'")
    input_index: Optional[int] = Field(None, ge=0, description="Original 0-based input position in a batch")
    error: Optional[Dict[str, str]] = Field(None, description="Structured error payload if status=='ERROR'")

    @field_validator("prediction")
    @classmethod
    def validate_prediction(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("FRAUD", "LEGIT"):
            raise ValueError(f"prediction must be 'FRAUD', 'LEGIT', or None, got: {v}")
        return v

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            raise ValueError(f"risk_level must be one of LOW, MEDIUM, HIGH, CRITICAL, or None, got: {v}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        norm = v.upper()
        if norm not in ("SUCCESS", "ERROR"):
            raise ValueError(f"status must be 'SUCCESS' or 'ERROR', got: {v}")
        return norm


# ---------------------------------------------------------------------------
# Standardized Dataclass Container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StandardizedPredictionResult:
    """
    Unified result object supporting single-transaction responses and individual
    batch record outcomes.
    """

    status: str  # "SUCCESS" | "ERROR"
    transaction_id: Optional[str] = None
    prediction: Optional[str] = None  # "FRAUD" | "LEGIT" | None
    fraud_probability: Optional[float] = None
    threshold: Optional[float] = None
    risk_level: Optional[str] = None  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | None
    model_version: Optional[str] = None
    model_status: Optional[str] = None
    feature_count: Optional[int] = None
    reason_codes: List[Union[ReasonCode, Dict[str, str]]] = field(default_factory=list)
    explanation_metadata: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    input_index: Optional[int] = None  # Populated in batch inference
    error: Optional[Dict[str, str]] = None  # Populated on error

    def __post_init__(self) -> None:
        """Validate core invariants on construction."""
        norm_status = self.status.upper()
        if norm_status not in ("SUCCESS", "ERROR"):
            raise InvalidSchemaDataError(f"status must be 'SUCCESS' or 'ERROR', got: {self.status}")

        if norm_status == "SUCCESS":
            if self.prediction not in ("FRAUD", "LEGIT"):
                raise InvalidSchemaDataError(
                    f"Successful prediction must have prediction in ('FRAUD', 'LEGIT'), got: {self.prediction}"
                )
            if self.fraud_probability is None or not (0.0 <= self.fraud_probability <= 1.0):
                raise InvalidSchemaDataError(
                    f"Successful prediction must have fraud_probability in [0.0, 1.0], got: {self.fraud_probability}"
                )
            if self.threshold is None or not (0.0 <= self.threshold <= 1.0):
                raise InvalidSchemaDataError(
                    f"Successful prediction must have threshold in [0.0, 1.0], got: {self.threshold}"
                )
            if self.risk_level not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
                raise InvalidSchemaDataError(
                    f"Successful prediction must have risk_level in ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'), "
                    f"got: {self.risk_level}"
                )
        else:  # ERROR
            # Ensure failed transactions are NOT classified as low or valid risk
            if self.risk_level is not None:
                raise InvalidSchemaDataError(
                    f"Failed result must NOT have a risk_level assigned, got: {self.risk_level}"
                )
            if self.error is None:
                raise InvalidSchemaDataError("Error result must contain an error dictionary.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the result to a clean, JSON-serializable dictionary.
        Does NOT expose internal file paths or raw transaction features.
        """
        serialized_reasons: List[Dict[str, str]] = []
        for rc in self.reason_codes:
            if isinstance(rc, ReasonCode):
                serialized_reasons.append(rc.to_dict())
            elif isinstance(rc, dict):
                serialized_reasons.append(dict(rc))
            else:
                serialized_reasons.append({"code": str(rc), "description": str(rc), "category": "GENERAL"})

        return {
            "status": self.status.upper(),
            "transaction_id": self.transaction_id,
            "prediction": self.prediction,
            "fraud_probability": self.fraud_probability,
            "threshold": self.threshold,
            "risk_level": self.risk_level,
            "model_version": self.model_version,
            "model_status": self.model_status,
            "feature_count": self.feature_count,
            "reason_codes": serialized_reasons,
            "explanation_metadata": self.explanation_metadata,
            "execution_time_ms": self.execution_time_ms,
            "input_index": self.input_index,
            "error": self.error,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serializes the result dictionary to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_pydantic(self) -> PredictionResultModel:
        """Converts into a validated Pydantic model."""
        return PredictionResultModel(**self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StandardizedPredictionResult:
        """Constructs an instance from a dictionary."""
        # Extract reasons if present
        raw_reasons = data.get("reason_codes", [])
        parsed_reasons: List[ReasonCode] = []
        for r in raw_reasons:
            if isinstance(r, dict):
                parsed_reasons.append(
                    ReasonCode(
                        code=r.get("code", "UNKNOWN"),
                        description=r.get("description", ""),
                        category=r.get("category", "GENERAL"),
                    )
                )
            elif isinstance(r, ReasonCode):
                parsed_reasons.append(r)

        return cls(
            status=data.get("status", "SUCCESS"),
            transaction_id=data.get("transaction_id"),
            prediction=data.get("prediction"),
            fraud_probability=data.get("fraud_probability"),
            threshold=data.get("threshold"),
            risk_level=data.get("risk_level"),
            model_version=data.get("model_version"),
            model_status=data.get("model_status"),
            feature_count=data.get("feature_count"),
            reason_codes=parsed_reasons,
            explanation_metadata=data.get("explanation_metadata"),
            execution_time_ms=data.get("execution_time_ms"),
            input_index=data.get("input_index"),
            error=data.get("error"),
        )
