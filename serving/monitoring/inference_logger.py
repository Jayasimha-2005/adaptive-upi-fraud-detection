"""
monitoring/inference_logger.py
Phase 8 — Structured per-request inference logging for the E1 serving system.

Emits one structured log line per prediction using the existing Python logging
infrastructure. No new logging framework is introduced.

PRIVACY RULE:
    Raw transaction payloads are NOT logged.
    Only the minimum information required for monitoring and traceability is recorded:
      - timestamp (via logging)
      - endpoint
      - transaction_id
      - model_version
      - fraud_probability
      - decision
      - preprocessing_time_ms, model_prediction_time_ms, total_inference_time_ms
      - http_status

Usage:
    from monitoring.inference_logger import log_inference_event

    log_inference_event(
        endpoint="/predict",
        transaction_id="3544193",
        model_version="E1",
        fraud_probability=0.003993,
        decision="LEGIT",
        preprocessing_ms=53.5,
        model_ms=2.4,
        total_ms=55.9,
        http_status=200,
    )
"""
from __future__ import annotations

import logging
from typing import Optional, Union

# Use a dedicated logger for inference events — separable from the FastAPI app logger
inference_logger = logging.getLogger("inference_monitor")


def log_inference_event(
    endpoint: str,
    transaction_id: Union[str, int],
    model_version: str,
    fraud_probability: float,
    decision: str,
    preprocessing_ms: float,
    model_ms: float,
    total_ms: float,
    http_status: int,
    batch_size: Optional[int] = None,
) -> None:
    """
    Emit a structured inference log line.

    Parameters
    ----------
    endpoint : str
        The API endpoint path (e.g. "/predict", "/predict/batch").
    transaction_id : str or int
        Transaction identifier for traceability.
    model_version : str
        Model version string (e.g. "E1").
    fraud_probability : float
        Raw fraud probability produced by the model [0.0, 1.0].
    decision : str
        Classification decision: "FRAUD" or "LEGIT".
    preprocessing_ms : float
        Preprocessing time in milliseconds.
    model_ms : float
        Model prediction time in milliseconds.
    total_ms : float
        Total inference time in milliseconds.
    http_status : int
        HTTP response status code.
    batch_size : int, optional
        Number of transactions in batch (only for /predict/batch).
    """
    parts = [
        f"endpoint={endpoint}",
        f"transaction_id={transaction_id}",
        f"model_version={model_version}",
        f"fraud_probability={fraud_probability:.6f}",
        f"decision={decision}",
        f"preprocessing_ms={preprocessing_ms:.3f}",
        f"model_ms={model_ms:.3f}",
        f"total_ms={total_ms:.3f}",
        f"http_status={http_status}",
    ]
    if batch_size is not None:
        parts.append(f"batch_size={batch_size}")

    inference_logger.info(" | ".join(parts))


def log_inference_error(
    endpoint: str,
    http_status: int,
    error_type: str,
    transaction_id: Optional[Union[str, int]] = None,
) -> None:
    """
    Emit a structured inference error log line (no payload, no stack trace).

    Parameters
    ----------
    endpoint : str
        API endpoint where the error occurred.
    http_status : int
        HTTP response status code.
    error_type : str
        Short error category (e.g. "validation_error", "model_unavailable", "internal_error").
    transaction_id : str or int, optional
        Transaction ID if available at the point of failure.
    """
    parts = [
        f"endpoint={endpoint}",
        f"http_status={http_status}",
        f"error_type={error_type}",
    ]
    if transaction_id is not None:
        parts.append(f"transaction_id={transaction_id}")

    inference_logger.warning(" | ".join(parts))
