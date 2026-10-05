"""
api/schemas.py
Pydantic request and response schemas for FastAPI Model Serving.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    model_name: str = Field(..., example="E1_LightGBM")
    model_version: str = Field(..., example="E1")
    feature_count: int = Field(..., example=406)
    decision_threshold: float = Field(..., example=0.616521)


class TransactionRequest(BaseModel):
    """
    Raw transaction request schema.
    
    Accepts raw transaction fields (TransactionID, TransactionDT, TransactionAmt, etc.).
    Allows extra dynamic raw IEEE-CIS columns passed in JSON payload.
    """
    model_config = ConfigDict(extra="allow")

    transaction_id: Optional[Union[int, str]] = Field(None, alias="TransactionID")
    TransactionDT: Optional[float] = Field(None, description="Transaction timestamp in seconds")
    TransactionAmt: Optional[float] = Field(None, description="Transaction amount")
    ProductCD: Optional[str] = Field(None, description="Product code")
    card1: Optional[Union[int, float]] = Field(None)
    card2: Optional[Union[int, float]] = Field(None)
    card3: Optional[Union[int, float]] = Field(None)
    card4: Optional[str] = Field(None)
    card5: Optional[Union[int, float]] = Field(None)
    card6: Optional[str] = Field(None)
    addr1: Optional[Union[int, float]] = Field(None)
    addr2: Optional[Union[int, float]] = Field(None)
    dist1: Optional[Union[int, float]] = Field(None)
    P_emaildomain: Optional[str] = Field(None)
    R_emaildomain: Optional[str] = Field(None)


class BatchTransactionRequest(BaseModel):
    """Batch transaction request schema."""
    transactions: List[TransactionRequest] = Field(..., min_items=1)


class PredictionResponse(BaseModel):
    """Fraud prediction response schema."""
    transaction_id: Union[int, str] = Field(..., example="3544193")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, example=0.003993)
    decision: str = Field(..., example="LEGIT")
    actual_label: Optional[str] = Field("UNKNOWN", example="LEGIT")
    preprocessing_time_ms: float = Field(..., example=0.594)
    model_prediction_time_ms: float = Field(..., example=0.042)
    total_inference_time_ms: float = Field(..., example=0.636)


class BatchPredictionResponse(BaseModel):
    """Batch fraud prediction response schema."""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[Any] = None


class VelocityMetricsSchema(BaseModel):
    """Real-time velocity features computed by Flink / Spark streaming windows."""
    model_config = ConfigDict(extra="allow")

    count_5m: Optional[int] = Field(None, description="Number of transactions in trailing 5 minutes")
    amount_5m: Optional[float] = Field(None, description="Total amount spent in trailing 5 minutes")
    avg_amount_5m: Optional[float] = Field(None, description="Average amount in trailing 5 minutes")
    count_10m: Optional[int] = Field(None, description="Number of transactions in trailing 10 minutes")
    amount_10m: Optional[float] = Field(None, description="Total amount spent in trailing 10 minutes")
    velocity_ratio: Optional[float] = Field(None, description="Velocity ratio count_5m / count_10m")


class HydratedPredictionRequest(BaseModel):
    """Compact streaming event to be hydrated via OnlineFeatureHydrationAdapter."""
    model_config = ConfigDict(extra="allow")

    transaction_id: Union[int, str] = Field(..., alias="TransactionID")
    card_id: str = Field(...)
    amount: float = Field(..., alias="TransactionAmt")
    event_time: float = Field(..., alias="TransactionDT")
    product_code: Optional[str] = Field("W", alias="ProductCD")
    merchant_id: Optional[str] = Field(None)
    device_type: Optional[str] = Field(None, alias="DeviceType")
    country: Optional[str] = Field(None)
    velocity: Optional[VelocityMetricsSchema] = Field(None)

