"""
serving/hydration/schemas.py
Pydantic schemas and dataclasses for the Feature Hydration Layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class VelocityMetrics(BaseModel):
    """Real-time velocity features computed by Flink / Spark streaming windows."""
    model_config = ConfigDict(extra="allow")

    count_5m: Optional[int] = Field(None, description="Number of transactions in trailing 5 minutes")
    amount_5m: Optional[float] = Field(None, description="Total amount spent in trailing 5 minutes")
    avg_amount_5m: Optional[float] = Field(None, description="Average amount in trailing 5 minutes")
    count_10m: Optional[int] = Field(None, description="Number of transactions in trailing 10 minutes")
    amount_10m: Optional[float] = Field(None, description="Total amount spent in trailing 10 minutes")
    velocity_ratio: Optional[float] = Field(None, description="Velocity ratio count_5m / count_10m")


class StreamingTransactionPayload(BaseModel):
    """
    Compact incoming streaming event (from Member 1 Kafka / Member 2 Flink).
    """
    model_config = ConfigDict(extra="allow")

    transaction_id: Union[int, str] = Field(..., alias="TransactionID", description="Unique transaction ID")
    card_id: str = Field(..., description="Cardholder or card partition key")
    amount: float = Field(..., alias="TransactionAmt", description="Transaction amount")
    event_time: float = Field(..., alias="TransactionDT", description="Timestamp in seconds or epoch ms")
    product_code: Optional[str] = Field("W", alias="ProductCD", description="Product code (e.g. 'W', 'H', 'C')")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    device_type: Optional[str] = Field(None, alias="DeviceType", description="Client device type")
    country: Optional[str] = Field(None, description="Country code")
    velocity: Optional[VelocityMetrics] = Field(None, description="Flink 5m/10m velocity metrics")


@dataclass
class HydrationResult:
    """
    Structured outcome of the Feature Hydration process.
    Acts as the hard gate: scoring is only permitted if `complete == True`.
    """
    complete: bool
    point_in_time_valid: bool
    dataframe: Optional[pd.DataFrame] = None
    missing_features: List[str] = field(default_factory=list)
    feature_provenance: Dict[str, str] = field(default_factory=dict)
    rejection_reason: Optional[str] = None
    card_id: Optional[str] = None
    transaction_id: Optional[str] = None
