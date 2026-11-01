"""
phase4_api/src/routes.py
========================
FastAPI Route Handlers for Phase 7 API Serving.

Part of Member 3 — API Serving for Adaptive UPI / BAF Fraud Detection.

Exposes:
  - GET  /health                  -> Health and readiness checks
  - GET  /api/v1/model/info       -> Safe model metadata & configuration
  - POST /api/v1/predict/single   -> Single transaction prediction
  - POST /api/v1/predict/batch    -> Batch transaction prediction
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, status

from phase4_api.src.api_schemas import (
    BatchPredictionResponseModel,
    BatchTransactionRequest,
    HealthResponse,
    ModelInfoResponse,
    SinglePredictionResponse,
    SingleTransactionRequest,
)
from phase4_api.src.batch_inference_engine import (
    BatchSizeLimitExceededError,
    BatchTransactionInferenceEngine,
)
from phase4_api.src.dependencies import (
    get_active_bundle,
    get_batch_engine,
    get_single_engine,
)
from phase4_api.src.inference_engine import SingleTransactionInferenceEngine
from phase4_api.src.model_loader import (
    APPROVED_DEFAULT_MODEL_VERSION,
    EXPECTED_FEATURE_COUNT,
    LoadedModelBundle,
)
from phase4_api.src.risk_classifier import (
    DEFAULT_RISK_POLICY_VERSION,
    get_risk_classifier,
)

logger = logging.getLogger("phase4_api.routes")
router = APIRouter()

# Service start timestamp for uptime calculation
START_TIME = time.time()


# ---------------------------------------------------------------------------
# Root Service Index
# ---------------------------------------------------------------------------
@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="API Root Index",
    description="Returns service metadata and navigation endpoints for documentation and health checks.",
    tags=["Service"],
)
def root_index() -> Dict[str, Any]:
    """Provides navigation links to the API documentation and health status."""
    return {
        "service": "Adaptive UPI Fraud Detection API",
        "member": "Member 3 — API Serving Layer",
        "status": "online",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
        "endpoints": {
            "health": "/health",
            "model_info": "/api/v1/model/info",
            "predict_single": "/api/v1/predict/single",
            "predict_batch": "/api/v1/predict/batch",
        },
    }


# ---------------------------------------------------------------------------
# Health & Readiness Check
# ---------------------------------------------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health & Readiness Check",
    description="Returns service liveness and readiness status, confirming LightGBM booster memory residency.",
    tags=["Monitoring"],
)
def check_health(
    engine: SingleTransactionInferenceEngine = Depends(get_single_engine),
) -> HealthResponse:
    """Verifies that the API service is alive and the inference engine is ready."""
    model_loaded = False
    ready = False
    threshold = 0.0

    try:
        threshold = engine.threshold
        model_loaded = engine.booster is not None
        ready = model_loaded and threshold > 0.0
    except Exception as e:
        logger.warning("Health check detected degraded engine: %s", str(e))

    uptime = time.time() - START_TIME
    now_iso = datetime.now(timezone.utc).isoformat()

    return HealthResponse(
        status="healthy" if ready else "degraded",
        model_loaded=model_loaded,
        ready=ready,
        model_version=engine.model_version,
        frozen_threshold=threshold,
        uptime_seconds=round(uptime, 2),
        timestamp=now_iso,
    )


# ---------------------------------------------------------------------------
# Model Information & Metadata
# ---------------------------------------------------------------------------
@router.get(
    "/api/v1/model/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Model Telemetry & Metadata",
    description="Returns authoritative model configuration without exposing local filesystem paths or credentials.",
    tags=["Model Information"],
)
def get_model_information(
    bundle: LoadedModelBundle = Depends(get_active_bundle),
) -> ModelInfoResponse:
    """Returns safe, non-sensitive model metadata."""
    classifier = get_risk_classifier()
    high_cutoff = classifier.policy.resolve_high_threshold(bundle.frozen_threshold)

    return ModelInfoResponse(
        model_version=bundle.info.model_version,
        model_status="ACTIVE_LOADED",
        model_type="LightGBM Booster",
        dataset="Bank Account Fraud - BAF Base",
        feature_count=bundle.info.num_features,
        num_trees=bundle.info.num_trees,
        frozen_threshold=bundle.frozen_threshold,
        supported_inference_modes=["single", "batch"],
        risk_policy_version=classifier.policy.version,
        risk_cutoffs={
            "low": [0.0, classifier.policy.low_threshold],
            "medium": [classifier.policy.low_threshold, classifier.policy.medium_threshold],
            "high": [classifier.policy.medium_threshold, high_cutoff],
            "critical": [high_cutoff, 1.0],
        },
    )


# ---------------------------------------------------------------------------
# Single Transaction Prediction
# ---------------------------------------------------------------------------
@router.post(
    "/api/v1/predict/single",
    response_model=SinglePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Single Transaction Fraud Inference",
    description="Evaluates a single BAF transaction against the approved LightGBM v3 model using the frozen threshold.",
    tags=["Inference"],
)
def predict_single_transaction(
    request: SingleTransactionRequest,
    engine: SingleTransactionInferenceEngine = Depends(get_single_engine),
) -> SinglePredictionResponse:
    """Executes single-transaction inference with Phase 6 risk classification."""
    data = request.model_dump()
    tid = data.get("transaction_id")

    # Delegate to the Phase 4/6 SingleTransactionInferenceEngine
    resp = engine.predict(data, transaction_id=tid)

    # Convert to Pydantic response model
    return SinglePredictionResponse(
        status=resp.status,
        transaction_id=resp.transaction_id,
        prediction=resp.prediction,
        fraud_probability=resp.fraud_probability,
        threshold=resp.threshold,
        risk_level=resp.risk_level or "LOW",
        model_version=resp.model_version,
        model_status=resp.model_status,
        feature_count=resp.feature_count,
        reason_codes=[rc.to_dict() if hasattr(rc, "to_dict") else rc for rc in resp.reason_codes],
        explanation_metadata=resp.explanation_metadata,
        execution_time_ms=resp.execution_time_ms,
    )


# ---------------------------------------------------------------------------
# Batch Transaction Prediction
# ---------------------------------------------------------------------------
@router.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponseModel,
    status_code=status.HTTP_200_OK,
    summary="Batch Transaction Fraud Inference",
    description="Executes high-performance vectorized batch inference with partial-success tolerance.",
    tags=["Inference"],
)
def predict_batch_transactions(
    request: BatchTransactionRequest,
    batch_engine: BatchTransactionInferenceEngine = Depends(get_batch_engine),
) -> BatchPredictionResponseModel:
    """Executes vectorized batch inference with Phase 6 risk classification."""
    records = request.transactions

    # Validate batch size limit against engine configuration
    if len(records) > batch_engine.max_batch_size:
        raise BatchSizeLimitExceededError(
            f"Batch size of {len(records)} exceeds maximum allowed limit of {batch_engine.max_batch_size} records."
        )

    # Delegate to Phase 5/6 BatchTransactionInferenceEngine
    batch_resp = batch_engine.predict_batch(records)

    # Convert to Pydantic response model
    return BatchPredictionResponseModel(
        batch_status=batch_resp.batch_status,
        total_transactions=batch_resp.total_transactions,
        successful_transactions=batch_resp.successful_transactions,
        failed_transactions=batch_resp.failed_transactions,
        model_version=batch_resp.model_version,
        threshold=batch_resp.threshold,
        results=[
            {
                "input_index": r.input_index,
                "transaction_id": r.transaction_id,
                "status": r.status,
                "prediction": r.prediction,
                "fraud_probability": r.fraud_probability,
                "threshold": r.threshold,
                "risk_level": r.risk_level,
                "model_version": r.model_version,
                "feature_count": r.feature_count,
                "reason_codes": [rc.to_dict() if hasattr(rc, "to_dict") else rc for rc in r.reason_codes]
                if r.reason_codes
                else None,
                "explanation_metadata": r.explanation_metadata,
                "error": r.error,
            }
            for r in batch_resp.results
        ],
        execution_time_ms=batch_resp.execution_time_ms,
    )
