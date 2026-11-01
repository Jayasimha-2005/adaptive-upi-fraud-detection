"""
phase4_api/src/main.py
======================
FastAPI Application Entry Point for Phase 4 BAF Fraud Detection Serving.

Part of Member 3 — API Serving for Adaptive UPI / BAF Fraud Detection.

Orchestrates:
  1. Safe application lifecycle management via async lifespan context.
  2. Centralized exception handling across validation, inference, and service errors.
  3. OpenAPI / Swagger / ReDoc interactive documentation.
  4. Decoupled router registration.
  5. Security safeguards preventing stack trace leakage and PII exposure.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import pathlib
import sys
from typing import Any, AsyncGenerator, Dict, List

# Ensure repository root is on sys.path so 'phase4_api' is always resolvable
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from phase4_api.src.batch_inference_engine import (
    BatchInferenceError,
    BatchSizeLimitExceededError,
    InvalidBatchInputError,
    get_batch_inference_engine,
)
from phase4_api.src.inference_engine import (
    InferenceEngineError,
    InvalidModelOutputError,
    InvalidTransactionIdError,
    ModelNotReadyError,
    ModelPredictionError,
    get_inference_engine,
)
from phase4_api.src.model_loader import (
    APPROVED_DEFAULT_MODEL_VERSION,
    ModelArtifactNotFoundError,
    ModelCorruptedError,
    ModelLoaderError,
    ModelMetadataError,
    ThresholdConfigurationError,
    get_model_bundle,
)
from phase4_api.src.routes import router as api_router
from phase4_api.src.transaction_preprocessor import (
    EmptyTransactionError,
    ExtraFeatureError,
    InvalidCategoricalValueError,
    InvalidNumericalValueError,
    InvalidTransactionFormatError,
    MissingFeatureError,
    PreprocessedOutputValidationError,
    PreprocessorError,
)

# Configure Safe Structured Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase4_api.main")


# ---------------------------------------------------------------------------
# Lifespan Context Manager
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages application startup and shutdown.
    Pre-warms the approved LightGBM v3 model and caches inference engines.
    """
    logger.info("Initializing Phase 4 Adaptive Fraud Detection API...")
    try:
        # Pre-warm model bundle and inference engines
        bundle = get_model_bundle(model_version=APPROVED_DEFAULT_MODEL_VERSION)
        get_inference_engine(model_version=APPROVED_DEFAULT_MODEL_VERSION)
        get_batch_inference_engine(model_version=APPROVED_DEFAULT_MODEL_VERSION)
        logger.info(
            "Model pre-warm successful: Version=%s | Trees=%s | Threshold=%.6f",
            bundle.info.model_version,
            bundle.info.num_trees,
            bundle.frozen_threshold,
        )
    except Exception as e:
        logger.error(
            "Failed to pre-warm model on startup: %s. Health checks will report degraded status.",
            str(e),
        )

    yield

    logger.info("Shutting down Phase 4 Adaptive Fraud Detection API...")


# ---------------------------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Constructs and configures the FastAPI application instance."""
    app = FastAPI(
        title="Adaptive Financial Fraud Detection API",
        description=(
            "High-performance REST serving layer for concept-drift-adapted LightGBM "
            "fraud detection on the Bank Account Fraud (BAF) benchmark dataset. "
            "Features frozen threshold decisioning, decoupled risk categorization, "
            "auditable reason codes, and vectorized batch inference."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS configuration with safe defaults
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(api_router)

    # -----------------------------------------------------------------------
    # Centralized Exception Handlers
    # -----------------------------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handles Pydantic input validation errors without leaking sensitive data."""
        clean_errors: List[Dict[str, Any]] = []
        for err in exc.errors():
            clean_errors.append(
                {
                    "loc": [str(loc_item) for loc_item in err.get("loc", [])],
                    "msg": err.get("msg", "Validation error"),
                    "type": err.get("type", "value_error"),
                }
            )
        logger.warning("Request validation failed for %s: %s", request.url.path, clean_errors)
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "REQUEST_VALIDATION_ERROR",
                    "message": "The incoming request payload failed schema validation.",
                    "details": clean_errors,
                }
            },
        )

    @app.exception_handler(BatchSizeLimitExceededError)
    async def batch_size_limit_handler(
        request: Request, exc: BatchSizeLimitExceededError
    ) -> JSONResponse:
        """Handles batch size limit exceedance with HTTP 413 Payload Too Large."""
        logger.warning("Batch limit exceeded: %s", str(exc))
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "code": "BATCH_SIZE_LIMIT_EXCEEDED",
                    "message": str(exc),
                    "details": None,
                }
            },
        )

    async def preprocessing_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handles transaction preprocessing, feature mismatch, and schema contract errors."""
        code = getattr(exc, "__class__", type(exc)).__name__
        logger.warning("Preprocessing rejection (%s): %s", code, str(exc))
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": code,
                    "message": str(exc),
                    "details": None,
                }
            },
        )

    for exc_cls in [
        MissingFeatureError,
        ExtraFeatureError,
        InvalidNumericalValueError,
        InvalidCategoricalValueError,
        InvalidTransactionFormatError,
        EmptyTransactionError,
        InvalidBatchInputError,
        InvalidTransactionIdError,
        PreprocessedOutputValidationError,
    ]:
        app.add_exception_handler(exc_cls, preprocessing_exception_handler)

    async def model_availability_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handles model unavailability and configuration errors with HTTP 503."""
        code = getattr(exc, "__class__", type(exc)).__name__
        logger.error("Model service unavailable (%s): %s", code, str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": f"Inference model service is currently unavailable: {exc}",
                    "details": None,
                }
            },
        )

    for exc_cls in [
        ModelNotReadyError,
        ModelLoaderError,
        ModelArtifactNotFoundError,
        ModelCorruptedError,
        ModelMetadataError,
        ThresholdConfigurationError,
    ]:
        app.add_exception_handler(exc_cls, model_availability_handler)

    async def inference_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handles internal prediction computation failures with HTTP 500."""
        logger.error("Model prediction failure: %s", str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INFERENCE_ERROR",
                    "message": f"An error occurred during model prediction: {exc}",
                    "details": None,
                }
            },
        )

    for exc_cls in [
        ModelPredictionError,
        InvalidModelOutputError,
        InferenceEngineError,
        BatchInferenceError,
    ]:
        app.add_exception_handler(exc_cls, inference_error_handler)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Preserves HTTP exceptions with standardized envelope."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": str(exc.detail),
                    "details": None,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catches all unhandled exceptions, sanitizing output to prevent trace leakage."""
        logger.critical("Unhandled internal server exception: %s", str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred while processing the request.",
                    "details": None,
                }
            },
        )

    return app


# Application Singleton for ASGI servers (e.g. Uvicorn)
app = create_app()
