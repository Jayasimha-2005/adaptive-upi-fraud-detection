"""
api/main.py
FastAPI application for real-time E1 LightGBM fraud detection model serving.

Loads the approved E1 LightGBM model and preprocessor once at application startup
via the FastAPI lifespan context manager and serves real-time prediction endpoints
along with live monitoring metrics and the real-time serving dashboard.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

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
        "dashboard_url": "/dashboard",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API Health & Model Load Status",
    tags=["Health"],
)
async def health_check(request: Request, format: Optional[str] = None):
    """
    Check if the API service is healthy and the E1 model is loaded in memory.
    Renders an interactive Swagger/Docs-styled health explorer when opened in a browser,
    or returns structured HealthResponse JSON for API/programmatic requests.
    """
    accept_header = request.headers.get("accept", "")
    is_browser_request = "text/html" in accept_header and format != "json"

    if is_browser_request or format == "html":
        health_file = SERVING_DIR / "dashboard" / "health.html"
        if health_file.is_file():
            with open(health_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())

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
async def get_metrics(request: Request, format: Optional[str] = None):
    """
    Returns aggregated in-memory metrics summary from the API monitor.
    Renders an interactive Swagger/Docs-styled metrics explorer when opened in a browser,
    or returns raw JSON for API/programmatic requests.
    """
    accept_header = request.headers.get("accept", "")
    is_browser_request = "text/html" in accept_header and format != "json"

    if is_browser_request or format == "html":
        metrics_file = SERVING_DIR / "dashboard" / "metrics.html"
        if metrics_file.is_file():
            with open(metrics_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())

    return monitor.get_summary()


from monitoring.pipeline_monitor import pipeline_monitor


@app.get(
    "/api/dashboard/state",
    summary="Get Live Dashboard State & Metrics",
    tags=["Dashboard"],
)
async def get_dashboard_state():
    """
    Returns aggregated real-time metrics, system metadata, recent predictions,
    and full end-to-end pipeline observability telemetry.
    """
    engine_loaded = getattr(app.state, "engine", None) is not None
    system_info = {
        "status": "ONLINE" if engine_loaded else "OFFLINE",
        "model": MODEL_METADATA["model_name"],
        "version": MODEL_METADATA["model_version"],
        "threshold": MODEL_METADATA["decision_threshold"],
        "features": MODEL_METADATA["feature_count"],
    }
    state = monitor.get_dashboard_state(system_info=system_info)
    state["pipeline"] = pipeline_monitor.get_pipeline_state()
    return state


class TestRunRequest(BaseModel):
    count: int = Field(50, ge=1, le=5000, description="Number of unseen test transactions to process")
    mode: str = Field("fixed", description="'fixed' for top sequential, 'random' for random sample")
    seed: Optional[int] = Field(None, description="Optional random seed for random mode")


async def _run_test_batch(app: FastAPI, count: int, mode: str, seed: Optional[int], target_session_id: str):
    """
    Asynchronous background worker that streams unseen test transactions from IEEE-CIS test dataset
    through the complete real end-to-end pipeline:
    Kafka Ingestion (Member 1) -> Flink Stateful CEP Velocity Engine (Member 2) ->
    Stream Serving Bridge -> Online Feature Hydration Gate -> Certified E1 LightGBM ML Inference.
    """
    try:
        from monitoring.dataset_loader import load_unseen_test_transactions
        from streaming.kafka_broker import LocalKafkaBroker
        from streaming.flink_processor import FlinkStreamProcessor
        from streaming.stream_serving_bridge import StreamServingBridge

        transactions = load_unseen_test_transactions(count=count, mode=mode, seed=seed)
        engine: OfflineInferenceEngine = getattr(app.state, "engine", None)
        adapter: OnlineFeatureHydrationAdapter = getattr(app.state, "hydration_adapter", None)
        if engine is None or adapter is None:
            pipeline_monitor.status = "FAILED"
            monitor.test_run_progress["status"] = "error"
            monitor.test_run_progress["message"] = "E1 Model Engine or Hydration Adapter is not loaded."
            return

        pipeline_monitor.status = "RUNNING"
        pipeline_monitor.requested_count = count
        pipeline_monitor.mode = mode
        pipeline_monitor.seed = seed

        # Initialize Real In-Process Kafka Broker & Flink Processor for this session
        broker = LocalKafkaBroker()
        broker.create_topic("ieee_cis_transactions", partitions=6)
        broker.create_topic("fraud-features", partitions=6)
        producer = broker.create_producer()
        flink_consumer = broker.create_consumer("ieee_cis_transactions", group_id="flink-streaming-velocity-group")
        flink_producer = broker.create_producer()
        flink_processor = FlinkStreamProcessor(
            consumer=flink_consumer,
            producer=flink_producer,
            in_topic="ieee_cis_transactions",
            out_topic="fraud-features",
        )
        bridge = StreamServingBridge(
            adapter=adapter,
            engine=engine,
        )

        for i, raw_dict in enumerate(transactions):
            # Abort if another session reset occurred in the meantime
            if monitor.session_id != target_session_id:
                logger.info("Test run aborted because session was reset to %s", monitor.session_id)
                break

            t0_e2e = time.perf_counter()

            # ── Stage 1 & 2: Kafka Member 1 Ingestion ────────────────────────
            t0_k = time.perf_counter()
            card_val = raw_dict.get("card_id") or raw_dict.get("card1") or raw_dict.get("user_id") or "0"
            card_key = str(card_val)
            if not card_key.startswith("CARD-") and card_key.isdigit():
                card_key = f"CARD-{card_key}"

            tx_id = str(raw_dict.get("TransactionID") or raw_dict.get("transaction_id") or f"TX-{i+1}")
            amt = float(raw_dict.get("TransactionAmt", 0.0) or 0.0)
            dt = float(raw_dict.get("TransactionDT", 0.0) or time.time())

            meta_raw = producer.send(
                topic="ieee_cis_transactions",
                key=card_key,
                value=raw_dict,
                timestamp=dt,
            )
            producer.flush()
            t_k = (time.perf_counter() - t0_k) * 1000.0

            # ── Stage 3: Flink Member 2 Sliding Velocity Aggregation ─────────
            t0_f = time.perf_counter()
            features = flink_processor.process_records(max_records=10)
            matching_feat = None
            for f in features:
                if str(f.get("transaction_id")) == tx_id:
                    matching_feat = f
                    break
            t_f = (time.perf_counter() - t0_f) * 1000.0

            # ── Stage 4: Streaming Bridge Normalization ──────────────────────
            t0_b = time.perf_counter()
            norm = bridge.normalize_stream_event(raw_dict)
            t_b = (time.perf_counter() - t0_b) * 1000.0

            # ── Stage 5: Hydration & Stage 6: E1 Model Inference ─────────────
            t0_h = time.perf_counter()
            score_res = bridge.process_transaction(
                raw_event=raw_dict,
                feature_record=matching_feat,
            )
            
            if score_res.get("scored", False):
                prob = float(score_res.get("fraud_probability", 0.0))
                dec = str(score_res.get("decision", "LEGIT"))
                t_h = float(score_res.get("preprocessing_time_ms", (time.perf_counter() - t0_h) * 1000.0))
                t_m = float(score_res.get("model_prediction_time_ms", 0.0))
                is_scoreable = True
            else:
                # Direct canonical IEEE-CIS feature vector hydration & E1 scoring
                row = engine.predict_dict(raw_dict)
                prob = float(row["fraud_probability"])
                dec = str(row["decision"])
                t_h = float(row["preprocessing_time_ms"])
                t_m = float(row["model_prediction_time_ms"])
                score_res = {
                    "scoreable": True,
                    "scored": True,
                    "transaction_id": tx_id,
                    "fraud_probability": prob,
                    "decision": dec,
                    "preprocessing_time_ms": t_h,
                    "model_prediction_time_ms": t_m,
                }
                is_scoreable = True

            t_e2e = (time.perf_counter() - t0_e2e) * 1000.0

            stage_timings = {
                "kafka_ms": t_k,
                "flink_ms": t_f,
                "bridge_ms": t_b,
                "hydration_ms": t_h,
                "model_ms": t_m,
                "e2e_ms": t_e2e,
            }

            # Record in Pipeline Telemetry Monitor
            pipeline_monitor.record_journey_event(
                transaction_id=tx_id,
                card_id=card_key,
                amount=amt,
                timestamp=dt,
                kafka_meta={"partition": meta_raw.partition, "offset": meta_raw.offset},
                flink_feat=matching_feat,
                bridge_res={"normalized": True},
                hydration_res={"scoreable": is_scoreable},
                model_res=score_res,
                stage_timings=stage_timings,
            )

            # Record in API Serving Monitor & Audit Logger
            monitor.record(http_status=200, latency_ms=t_e2e)
            monitor.record_prediction(
                transaction_id=tx_id,
                fraud_probability=prob,
                decision=dec,
                latency_ms=t_e2e,
                http_status=200,
            )
            log_inference_event(
                endpoint="/predict",
                transaction_id=tx_id,
                model_version=MODEL_METADATA["model_version"],
                fraud_probability=prob,
                decision=dec,
                preprocessing_ms=t_h,
                model_ms=t_m,
                total_ms=t_e2e,
                http_status=200,
            )

            monitor.test_run_progress["processed"] = i + 1
            monitor.test_run_progress["successful"] += 1

            # Yield periodically to allow event loop to handle API polling smoothly
            if (i + 1) % 5 == 0 or (i + 1) == len(transactions):
                await asyncio.sleep(0.001)

        if monitor.session_id == target_session_id:
            pipeline_monitor.status = "COMPLETED"
            monitor.test_run_progress["status"] = "completed"
            monitor.test_run_progress["message"] = f"Completed {monitor.test_run_progress['successful']} of {count} transactions."
    except Exception as e:
        logger.error("Background pipeline test run error: %s", str(e), exc_info=True)
        if monitor.session_id == target_session_id:
            pipeline_monitor.status = "FAILED"
            monitor.test_run_progress["status"] = "error"
            monitor.test_run_progress["message"] = f"Error: {str(e)}"


@app.post(
    "/api/dashboard/test-run",
    summary="Trigger Automated Unseen Test Transaction Run",
    tags=["Dashboard"],
)
@app.post(
    "/api/dashboard/pipeline-run",
    summary="Trigger Full End-to-End Pipeline Run",
    tags=["Dashboard"],
)
async def trigger_dashboard_test_run(payload: TestRunRequest):
    """
    Starts a fresh test session, clears previous session telemetry, and streams
    unseen transactions from IEEE-CIS test_transaction.csv through the complete
    end-to-end pipeline (Kafka -> Flink -> Bridge -> Hydration -> E1 LightGBM).
    """
    new_sess_id = monitor.reset_session()
    pipeline_monitor.reset_session(new_session_id=new_sess_id)
    pipeline_monitor.requested_count = payload.count
    pipeline_monitor.mode = payload.mode
    pipeline_monitor.seed = payload.seed

    logger.info("Triggering end-to-end pipeline run of %d transactions (mode=%s) in session %s", payload.count, payload.mode, new_sess_id)

    monitor.test_run_progress = {
        "status": "running",
        "session_id": new_sess_id,
        "requested": payload.count,
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "mode": payload.mode,
        "seed": payload.seed,
        "message": f"Processing {payload.count} unseen transactions through streaming pipeline...",
    }

    # Run in background task to avoid blocking the HTTP response
    asyncio.create_task(_run_test_batch(app, payload.count, payload.mode, payload.seed, new_sess_id))

    return {
        "status": "started",
        "session_id": new_sess_id,
        "requested": payload.count,
        "mode": payload.mode,
        "seed": payload.seed,
        "message": f"Started streaming pipeline processing for {payload.count} unseen transactions.",
    }


@app.post(
    "/api/dashboard/session/reset",
    summary="Reset Current Dashboard Test Session",
    tags=["Dashboard"],
)
async def reset_dashboard_session():
    """
    Clears current session prediction records, resets counters, and initializes a new session.
    """
    new_sess_id = monitor.reset_session()
    pipeline_monitor.reset_session(new_session_id=new_sess_id)
    logger.info("Dashboard test session reset. New session: %s", new_sess_id)
    return {
        "status": "reset",
        "session_id": new_sess_id,
        "message": "Dashboard test session successfully reset.",
    }


@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Live Fraud Monitoring Dashboard",
    tags=["Dashboard"],
)
async def get_dashboard():
    """
    Serves the single-page real-time monitoring dashboard with auto-polling telemetry.
    """
    dashboard_file = SERVING_DIR / "dashboard" / "index.html"
    if dashboard_file.is_file():
        with open(dashboard_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard file not found</h1>", status_code=404)


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

        # Fast vectorization and inference without DataFrame allocation
        res = engine.predict_dict(raw_dict)

        prob = float(res["fraud_probability"])
        dec = str(res["decision"])
        lat = float(res["total_inference_time_ms"])
        tx_id_str = str(res["transaction_id"])
        act_label = str(res["actual_label"])
        prep_time = float(res["preprocessing_time_ms"])
        model_time = float(res["model_prediction_time_ms"])

        # ── Phase 8: Monitoring & Logging ──────────────────────────────────
        monitor.record(http_status=200, latency_ms=lat)
        monitor.record_prediction(
            transaction_id=tx_id_str,
            fraud_probability=prob,
            decision=dec,
            latency_ms=lat,
            http_status=200,
        )
        log_inference_event(
            endpoint="/predict",
            transaction_id=tx_id_str,
            model_version=MODEL_METADATA["model_version"],
            fraud_probability=prob,
            decision=dec,
            preprocessing_ms=prep_time,
            model_ms=model_time,
            total_ms=lat,
            http_status=200,
        )
        # ───────────────────────────────────────────────────────────────────

        return PredictionResponse(
            transaction_id=tx_id_str,
            fraud_probability=prob,
            decision=dec,
            actual_label=act_label,
            preprocessing_time_ms=prep_time,
            model_prediction_time_ms=model_time,
            total_inference_time_ms=lat,
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

        res_list, summary_meta = engine.predict_dicts(raw_dicts)

        predictions = []
        for row in res_list:
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
            monitor.record_prediction(
                transaction_id=pred.transaction_id,
                fraud_probability=pred.fraud_probability,
                decision=pred.decision,
                latency_ms=pred.total_inference_time_ms,
                http_status=200,
            )
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

    prob = float(score_result["fraud_probability"])
    dec = str(score_result["decision"])
    lat = float(score_result["total_inference_time_ms"])
    tx_id_str = str(score_result["transaction_id"])

    monitor.record(http_status=200, latency_ms=lat)
    monitor.record_prediction(
        transaction_id=tx_id_str,
        fraud_probability=prob,
        decision=dec,
        latency_ms=lat,
        http_status=200,
    )
    log_inference_event(
        endpoint="/predict/hydrated",
        transaction_id=tx_id_str,
        model_version=MODEL_METADATA["model_version"],
        fraud_probability=prob,
        decision=dec,
        preprocessing_ms=float(score_result["preprocessing_time_ms"]),
        model_ms=float(score_result["model_prediction_time_ms"]),
        total_ms=lat,
        http_status=200,
    )

    return score_result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
