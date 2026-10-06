"""
api/main.py
FastAPI application for real-time E1 LightGBM fraud detection model serving.

Loads the approved E1 LightGBM model and preprocessor once at application startup
via the FastAPI lifespan context manager and serves real-time prediction endpoints.
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Add project root and serving directory to sys.path
SERVING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVING_DIR.parent
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from api.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    ErrorResponse,
    HealthResponse,
    HydratedPredictionRequest,
    PredictionResponse,
    TransactionRequest,
)
from inference.offline_inference import (
    E1_DECISION_THRESHOLD,
    OfflineInferenceEngine,
)
from monitoring.model_metadata import MODEL_METADATA
from monitoring.api_monitor import monitor
from monitoring.inference_logger import log_inference_event, log_inference_error
from serving.hydration.adapter import OnlineFeatureHydrationAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fastapi_serving")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Loads the approved canonical E1 model and preprocessor once at application startup.
    Initializes the OnlineFeatureHydrationAdapter.
    """
    logger.info("--- Starting FastAPI Model Serving Application ---")
    try:
        engine = OfflineInferenceEngine(
            threshold=E1_DECISION_THRESHOLD,
        )
        app.state.engine = engine
        app.state.hydration_adapter = OnlineFeatureHydrationAdapter()
        logger.info("E1 Model Engine and Hydration Adapter loaded successfully into memory.")
    except Exception as e:
        logger.error("Failed to load E1 Model Engine on startup: %s", str(e))
        app.state.engine = None
        app.state.hydration_adapter = None

    yield

    logger.info("--- Shutting down FastAPI Model Serving Application ---")
    app.state.engine = None
    app.state.hydration_adapter = None


app = FastAPI(
    title="Financial Fraud Model Serving API",
    description="Real-time ML Serving API for E1 LightGBM Fraud Detection Model",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Exception Handlers ────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic request validation errors cleanly (HTTP 422)."""
    logger.warning("Request validation failed for %s: %s", request.url.path, exc.errors())
    monitor.record(http_status=422)
    log_inference_error(
        endpoint=request.url.path,
        http_status=422,
        error_type="validation_error",
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Request Validation Error",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions cleanly."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "detail": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected server exceptions gracefully without exposing raw stack trace."""
    logger.error("Unhandled internal exception on %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "Prediction service encountered an internal error.",
        },
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="API Root", tags=["Health"])
async def root():
    """
    Root welcome endpoint with links to documentation and health check.
    """
    return {
        "message": "Welcome to Financial Fraud Model Serving API (E1 LightGBM)",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
        "metrics_url": "/metrics",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API Health & Model Load Status",
    tags=["Health"],
)
async def health_check():
    """
    Check if the API service is healthy and the E1 model is loaded in memory.
    Returns model name, version, feature count, and decision threshold from MODEL_METADATA.
    """
    engine_loaded = getattr(app.state, "engine", None) is not None
    return HealthResponse(
        status="healthy",
        model_loaded=engine_loaded,
        model_name=MODEL_METADATA["model_name"],
        model_version=MODEL_METADATA["model_version"],
        feature_count=MODEL_METADATA["feature_count"],
        decision_threshold=MODEL_METADATA["decision_threshold"],
    )


@app.get(
    "/metrics",
    summary="Get In-Memory API Performance & Monitoring Metrics",
    tags=["Monitoring"],
)
async def get_metrics():
    """
    Returns aggregated in-memory metrics summary from the API monitor.
    Includes request counts, HTTP status code buckets, and inference latency percentiles.
    """
    return monitor.get_summary()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
    },
    summary="Predict Fraud Probability for a Single Transaction",
    tags=["Inference"],
)
async def predict_single(payload: TransactionRequest):
    """
    Predict fraud probability and classification decision for a single transaction.
    """
    engine: OfflineInferenceEngine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="E1 Model Engine is not loaded.",
        )

    try:
        # Convert request payload to single-row dictionary
        raw_dict = payload.model_dump(by_alias=True, exclude_none=True)

        # Handle alias mapping if transaction_id passed
        if "transaction_id" in raw_dict and "TransactionID" not in raw_dict:
            raw_dict["TransactionID"] = raw_dict.pop("transaction_id")

        # Convert to Pandas DataFrame (1 row)
        df_raw = pd.DataFrame([raw_dict])

        # Execute inference through Phase 1 serving preprocessor & model
        res_df, meta = engine.predict_transaction(df_raw)
        row = res_df.iloc[0]

        # ── Phase 8: Monitoring & Logging ──────────────────────────────────
        monitor.record(http_status=200, latency_ms=float(row["total_inference_time_ms"]))
        log_inference_event(
            endpoint="/predict",
            transaction_id=str(row["transaction_id"]),
            model_version=MODEL_METADATA["model_version"],
            fraud_probability=float(row["fraud_probability"]),
            decision=str(row["decision"]),
            preprocessing_ms=float(row["preprocessing_time_ms"]),
            model_ms=float(row["model_prediction_time_ms"]),
            total_ms=float(row["total_inference_time_ms"]),
            http_status=200,
        )
        # ───────────────────────────────────────────────────────────────────

        return PredictionResponse(
            transaction_id=str(row["transaction_id"]),
            fraud_probability=float(row["fraud_probability"]),
            decision=str(row["decision"]),
            actual_label=str(row["actual_label"]),
            preprocessing_time_ms=float(row["preprocessing_time_ms"]),
            model_prediction_time_ms=float(row["model_prediction_time_ms"]),
            total_inference_time_ms=float(row["total_inference_time_ms"]),
        )

    except ValueError as ve:
        logger.warning("Invalid request data for predict: %s", str(ve))
        monitor.record(http_status=400)
        log_inference_error(endpoint="/predict", http_status=400, error_type="value_error")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error("Inference execution error: %s", str(e), exc_info=True)
        monitor.record(http_status=500)
        log_inference_error(endpoint="/predict", http_status=500, error_type="internal_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed due to an internal error.",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict Fraud Probability for a Batch of Transactions",
    tags=["Inference"],
)
async def predict_batch(payload: BatchTransactionRequest):
    """
    Predict fraud probabilities and classification decisions for a batch of transactions.
    """
    engine: OfflineInferenceEngine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="E1 Model Engine is not loaded.",
        )

    try:
        raw_dicts = []
        for item in payload.transactions:
            d = item.model_dump(by_alias=True, exclude_none=True)
            if "transaction_id" in d and "TransactionID" not in d:
                d["TransactionID"] = d.pop("transaction_id")
            raw_dicts.append(d)

        df_raw = pd.DataFrame(raw_dicts)
        res_df, summary_meta = engine.predict_transaction(df_raw)

        predictions = []
        for _, row in res_df.iterrows():
            predictions.append(
                PredictionResponse(
                    transaction_id=str(row["transaction_id"]),
                    fraud_probability=float(row["fraud_probability"]),
                    decision=str(row["decision"]),
                    actual_label=str(row["actual_label"]),
                    preprocessing_time_ms=float(row["preprocessing_time_ms"]),
                    model_prediction_time_ms=float(row["model_prediction_time_ms"]),
                    total_inference_time_ms=float(row["total_inference_time_ms"]),
                )
            )

        # ── Phase 8: Monitoring & Logging (per transaction in batch) ───────
        batch_size = len(predictions)
        for pred in predictions:
            monitor.record(http_status=200, latency_ms=pred.total_inference_time_ms)
            log_inference_event(
                endpoint="/predict/batch",
                transaction_id=pred.transaction_id,
                model_version=MODEL_METADATA["model_version"],
                fraud_probability=pred.fraud_probability,
                decision=pred.decision,
                preprocessing_ms=pred.preprocessing_time_ms,
                model_ms=pred.model_prediction_time_ms,
                total_ms=pred.total_inference_time_ms,
                http_status=200,
                batch_size=batch_size,
            )
        # ───────────────────────────────────────────────────────────────────

        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary_meta,
        )

    except Exception as e:
        logger.error("Batch inference error: %s", str(e), exc_info=True)
        monitor.record(http_status=500)
        log_inference_error(endpoint="/predict/batch", http_status=500, error_type="internal_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch inference execution failed due to an internal error.",
        )


@app.post(
    "/predict/hydrated",
    summary="Predict Fraud with Online Feature Hydration Gate",
    tags=["Inference"],
)
async def predict_hydrated(payload: HydratedPredictionRequest):
    """
    Predict fraud probability for a compact streaming event after passing
    through the Online Feature Hydration Adapter.
    Enforces the Hard Hydration Gate: if complete != True, returns HTTP 422 with
    rejection reason rather than scoring with missing features.
    """
    engine: OfflineInferenceEngine = getattr(app.state, "engine", None)
    adapter: OnlineFeatureHydrationAdapter = getattr(app.state, "hydration_adapter", None)
    if engine is None or adapter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serving Engine or Hydration Adapter is not loaded.",
        )

    score_result = adapter.score_or_reject(payload.model_dump(by_alias=True), engine)

    if not score_result.get("scoreable", False):
        monitor.record(http_status=422)
        log_inference_error(
            endpoint="/predict/hydrated",
            http_status=422,
            error_type="hydration_gate_rejection",
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=score_result,
        )

    monitor.record(http_status=200, latency_ms=float(score_result["total_inference_time_ms"]))
    log_inference_event(
        endpoint="/predict/hydrated",
        transaction_id=str(score_result["transaction_id"]),
        model_version=MODEL_METADATA["model_version"],
        fraud_probability=float(score_result["fraud_probability"]),
        decision=str(score_result["decision"]),
        preprocessing_ms=float(score_result["preprocessing_time_ms"]),
        model_ms=float(score_result["model_prediction_time_ms"]),
        total_ms=float(score_result["total_inference_time_ms"]),
        http_status=200,
    )

    return score_result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)

