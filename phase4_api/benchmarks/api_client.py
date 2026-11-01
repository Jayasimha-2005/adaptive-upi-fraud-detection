"""
phase4_api/benchmarks/api_client.py
===================================
Robust HTTP Client for Phase 8 Performance Benchmarking & Parity.

Part of Member 3 — API Serving Layer for Adaptive UPI / BAF Fraud Detection.
Phase 8: Performance Testing, Benchmarking & Parity Validation.

Strictly distinguishes:
- Client-observed end-to-end HTTP latency (ms) via high-resolution monotonic timer
- Server-reported inference latency (execution_time_ms)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
import urllib.request
import urllib.error

import httpx

logger = logging.getLogger("phase4_api.benchmarks.api_client")


@dataclass
class SingleRequestResult:
    """Latency and response payload for a single prediction request."""
    transaction_id: Optional[str]
    status_code: int
    is_success: bool
    client_latency_ms: float
    server_execution_time_ms: Optional[float] = None
    prediction: Optional[str] = None
    fraud_probability: Optional[float] = None
    risk_level: Optional[str] = None
    reason_codes: List[str] = field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class BatchRecordResult:
    """Per-record outcome inside a batch response."""
    input_index: int
    transaction_id: Optional[str]
    status: str
    prediction: Optional[str] = None
    fraud_probability: Optional[float] = None
    risk_level: Optional[str] = None
    reason_codes: List[str] = field(default_factory=list)
    error: Optional[Dict[str, str]] = None


@dataclass
class BatchRequestResult:
    """Latency and response payload for a batch prediction request."""
    batch_size: int
    status_code: int
    is_success: bool
    client_latency_ms: float
    server_execution_time_ms: Optional[float] = None
    batch_status: Optional[str] = None
    total_transactions: int = 0
    successful_transactions: int = 0
    failed_transactions: int = 0
    records: List[BatchRecordResult] = field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class BenchmarkApiClient:
    """
    Dedicated client for calling the FastAPI fraud detection service.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        # Reusable HTTP client with persistent connection pool
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout_seconds, connect=10.0),
        )

    def close(self) -> None:
        """Closes the underlying HTTP client session."""
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self) -> BenchmarkApiClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def check_health(self) -> Dict[str, Any]:
        """
        Queries GET /health to verify service readiness and LightGBM model residency.
        """
        url = f"{self.base_url}/health"
        try:
            resp = self._client.get("/health")
            if resp.status_code == 200:
                data = resp.json()
                return {"is_ready": data.get("ready", False), "details": data, "status_code": 200}
            return {
                "is_ready": False,
                "error": f"Health check returned HTTP {resp.status_code}",
                "status_code": resp.status_code,
            }
        except Exception as e:
            logger.error("Health check failed: %s", str(e))
            return {"is_ready": False, "error": str(e), "status_code": 0}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Queries GET /api/v1/model/info for active model version and threshold metadata.
        """
        try:
            resp = self._client.get("/api/v1/model/info")
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            return {"error": str(e)}

    def predict_single(self, transaction_id: Optional[str], features: Dict[str, Any]) -> SingleRequestResult:
        """
        Sends a single transaction prediction request to POST /api/v1/predict/single.
        Measures exact client latency with time.perf_counter().
        """
        payload = dict(features)
        if transaction_id:
            payload["transaction_id"] = transaction_id

        start_time = time.perf_counter()
        try:
            resp = self._client.post("/api/v1/predict/single", json=payload)
            client_latency_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                reasons = [rc.get("code", "") for rc in data.get("reason_codes", []) if isinstance(rc, dict)]
                return SingleRequestResult(
                    transaction_id=data.get("transaction_id", transaction_id),
                    status_code=resp.status_code,
                    is_success=True,
                    client_latency_ms=client_latency_ms,
                    server_execution_time_ms=data.get("execution_time_ms"),
                    prediction=data.get("prediction"),
                    fraud_probability=data.get("fraud_probability"),
                    risk_level=data.get("risk_level"),
                    reason_codes=reasons,
                    raw_response=data,
                )
            else:
                return SingleRequestResult(
                    transaction_id=transaction_id,
                    status_code=resp.status_code,
                    is_success=False,
                    client_latency_ms=client_latency_ms,
                    error_message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )
        except Exception as e:
            client_latency_ms = (time.perf_counter() - start_time) * 1000.0
            return SingleRequestResult(
                transaction_id=transaction_id,
                status_code=0,
                is_success=False,
                client_latency_ms=client_latency_ms,
                error_message=str(e),
            )

    def predict_batch(self, batch_records: List[Dict[str, Any]]) -> BatchRequestResult:
        """
        Sends a batch prediction request to POST /api/v1/predict/batch.
        Measures client latency and preserves input-to-output ordering.
        """
        payload = {"transactions": batch_records}
        batch_size = len(batch_records)

        start_time = time.perf_counter()
        try:
            resp = self._client.post("/api/v1/predict/batch", json=payload)
            client_latency_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                record_results: List[BatchRecordResult] = []
                for item in data.get("results", []):
                    reasons = [rc.get("code", "") for rc in item.get("reason_codes", []) if isinstance(rc, dict)]
                    record_results.append(
                        BatchRecordResult(
                            input_index=item.get("input_index", 0),
                            transaction_id=item.get("transaction_id"),
                            status=item.get("status", "unknown"),
                            prediction=item.get("prediction"),
                            fraud_probability=item.get("fraud_probability"),
                            risk_level=item.get("risk_level"),
                            reason_codes=reasons,
                            error=item.get("error"),
                        )
                    )

                return BatchRequestResult(
                    batch_size=batch_size,
                    status_code=resp.status_code,
                    is_success=True,
                    client_latency_ms=client_latency_ms,
                    server_execution_time_ms=data.get("execution_time_ms"),
                    batch_status=data.get("batch_status"),
                    total_transactions=data.get("total_transactions", batch_size),
                    successful_transactions=data.get("successful_transactions", 0),
                    failed_transactions=data.get("failed_transactions", 0),
                    records=record_results,
                    raw_response=data,
                )
            else:
                return BatchRequestResult(
                    batch_size=batch_size,
                    status_code=resp.status_code,
                    is_success=False,
                    client_latency_ms=client_latency_ms,
                    error_message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )
        except Exception as e:
            client_latency_ms = (time.perf_counter() - start_time) * 1000.0
            return BatchRequestResult(
                batch_size=batch_size,
                status_code=0,
                is_success=False,
                client_latency_ms=client_latency_ms,
                error_message=str(e),
            )
