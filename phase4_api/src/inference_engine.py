"""
phase4_api/src/inference_engine.py
==================================
Reusable Single-Transaction Inference Engine for Phase 4 BAF Fraud Detection.

Orchestrates:
1. Reusable model loading & caching via Phase 2 Phase4ModelLoader.
2. Invariant feature preprocessing & validation via Phase 3 TransactionPreprocessor.
3. LightGBM booster probability prediction & validation.
4. Binary fraud decisioning using the authoritative frozen threshold.
5. Structured, sanitized prediction response formatting.
6. Safe structured logging without raw feature or PII leakage.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from phase4_api.src.model_loader import (
    APPROVED_DEFAULT_MODEL_VERSION,
    EXPECTED_FEATURE_COUNT,
    LoadedModelBundle,
    ModelLoaderError,
    Phase4ModelLoader,
    ThresholdConfigurationError,
    get_model_bundle,
)
from phase4_api.src.transaction_preprocessor import (
    EmptyTransactionError,
    ExtraFeatureError,
    InvalidCategoricalValueError,
    InvalidNumericalValueError,
    InvalidTransactionFormatError,
    MissingFeatureError,
    PreprocessedOutputValidationError,
    PreprocessorError,
    TransactionPreprocessor,
)
from phase4_api.src.risk_classifier import (
    ReasonCode,
    RiskClassifier,
    RiskEvaluationResult,
    RiskLevel,
    get_risk_classifier,
)

# ---------------------------------------------------------------------------
# Configure Safe Structured Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("phase4_api.inference_engine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [InferenceEngine] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Inference Exception Hierarchy
# ---------------------------------------------------------------------------
class InferenceEngineError(Exception):
    """Base exception for all inference engine failures."""


class InvalidTransactionIdError(InferenceEngineError):
    """Raised when an invalid transaction identifier format is supplied."""


class ModelNotReadyError(InferenceEngineError):
    """Raised when the underlying model bundle cannot be initialized or loaded."""


class ModelPredictionError(InferenceEngineError):
    """Raised when the LightGBM booster raises an unexpected error during prediction."""


class InvalidModelOutputError(InferenceEngineError):
    """Raised when the model outputs non-finite, out-of-bounds, or malformed predictions."""


# ---------------------------------------------------------------------------
# Structured Prediction Response
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictionResponse:
    """
    Public, structured representation of single-transaction inference results.
    Does not expose sensitive raw transaction features or local filesystem paths.
    """

    transaction_id: Optional[str]
    prediction: str  # "FRAUD" | "LEGIT"
    fraud_probability: float  # Display probability rounded to 6 decimal places
    threshold: float  # Authoritative frozen threshold
    model_version: str  # e.g., "v3"
    model_status: str  # "ACTIVE_LOADED"
    feature_count: int  # 29
    execution_time_ms: float  # Duration of preprocessing + inference
    risk_level: Optional[str] = None  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    reason_codes: List[ReasonCode] = field(default_factory=list)
    explanation_metadata: Optional[Dict[str, Any]] = None
    status: str = "SUCCESS"

    def to_dict(self) -> Dict[str, Any]:
        """Converts to a clean, JSON-serializable dictionary."""
        serialized_reasons = [
            rc.to_dict() if hasattr(rc, "to_dict") else rc
            for rc in self.reason_codes
        ]
        return {
            "status": self.status,
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
        }


# ---------------------------------------------------------------------------
# Single-Transaction Inference Engine
# ---------------------------------------------------------------------------
class SingleTransactionInferenceEngine:
    """
    Core inference engine for real-time BAF transaction fraud detection.
    Connects the model loader, preprocessor, and frozen threshold decision logic.
    """

    def __init__(
        self,
        model_version: str = APPROVED_DEFAULT_MODEL_VERSION,
        model_bundle: Optional[LoadedModelBundle] = None,
        preprocessor: Optional[TransactionPreprocessor] = None,
        risk_classifier: Optional[RiskClassifier] = None,
    ) -> None:
        """
        Parameters
        ----------
        model_version : str, default="v3"
            The approved model version to load.
        model_bundle : Optional[LoadedModelBundle]
            Optional pre-loaded model bundle for dependency injection / mocking.
        preprocessor : Optional[TransactionPreprocessor]
            Optional preprocessor instance for dependency injection / mocking.
        risk_classifier : Optional[RiskClassifier]
            Optional risk classification engine for Phase 6 evaluation.
        """
        self.model_version = model_version
        self.preprocessor = preprocessor or TransactionPreprocessor(
            allow_unseen_categories=False,
            allow_missing_numerical=True,
        )
        self.risk_classifier = risk_classifier or get_risk_classifier()

        # Initialize or assign model bundle
        if model_bundle is not None:
            self._bundle = model_bundle
        else:
            try:
                self._bundle = get_model_bundle(model_version=self.model_version)
            except Exception as e:
                logger.error(
                    "Model initialization failed for version '%s': %s",
                    self.model_version,
                    str(e),
                )
                raise ModelNotReadyError(
                    f"Failed to initialize model '{self.model_version}': {e}"
                ) from e

        # Validate threshold from bundle
        self._validate_threshold(self._bundle.frozen_threshold)

    @property
    def bundle(self) -> LoadedModelBundle:
        """Returns the loaded model bundle."""
        return self._bundle

    @property
    def threshold(self) -> float:
        """Returns the authoritative frozen threshold."""
        return self._bundle.frozen_threshold

    @property
    def booster(self) -> Any:
        """Returns the active LightGBM Booster."""
        return self._bundle.booster

    @staticmethod
    def _validate_threshold(threshold: Any) -> float:
        """Asserts that the threshold is a finite float in (0.0, 1.0)."""
        try:
            val = float(threshold)
        except (ValueError, TypeError) as e:
            raise ThresholdConfigurationError(
                f"Configured threshold is not a valid float: {threshold!r}"
            ) from e

        if not math.isfinite(val) or not (0.0 < val < 1.0):
            raise ThresholdConfigurationError(
                f"Configured threshold {val} is outside valid probability interval (0.0, 1.0)"
            )
        return val

    @staticmethod
    def validate_transaction_id(transaction_id: Optional[str]) -> Optional[str]:
        """
        Validates the optional transaction identifier.
        Must be a non-empty string <= 128 characters without control characters, or None.
        """
        if transaction_id is None:
            return None

        if not isinstance(transaction_id, str):
            raise InvalidTransactionIdError(
                f"transaction_id must be a string, received {type(transaction_id).__name__}."
            )

        tid = transaction_id.strip()
        if not tid:
            raise InvalidTransactionIdError("transaction_id cannot be empty or whitespace.")

        if len(tid) > 128:
            raise InvalidTransactionIdError(
                f"transaction_id exceeds maximum length of 128 characters ({len(tid)} chars)."
            )

        # Disallow control characters
        if any(ord(c) < 32 or ord(c) == 127 for c in tid):
            raise InvalidTransactionIdError("transaction_id contains invalid control characters.")

        return tid

    def _validate_model_output(self, raw_pred: Any) -> float:
        """
        Validates raw booster prediction output:
        Must be a 1D array/sequence of length 1 containing a finite float in [0.0, 1.0].
        """
        # Check empty or None
        if raw_pred is None:
            raise InvalidModelOutputError("Model prediction output is None.")

        # Convert to numpy array or sequence
        if isinstance(raw_pred, np.ndarray):
            pred_flat = raw_pred.flatten()
            count = len(pred_flat)
        elif isinstance(raw_pred, (list, tuple)):
            pred_flat = list(raw_pred)
            count = len(pred_flat)
        elif isinstance(raw_pred, (float, int, np.floating)):
            pred_flat = [raw_pred]
            count = 1
        else:
            raise InvalidModelOutputError(
                f"Unexpected model prediction type: {type(raw_pred).__name__}"
            )

        if count == 0:
            raise InvalidModelOutputError("Model returned empty prediction output (0 elements).")

        if count > 1:
            raise InvalidModelOutputError(
                f"Expected exactly 1 prediction output, received {count} elements."
            )

        val = pred_flat[0]

        # Check numeric
        try:
            prob = float(val)
        except (ValueError, TypeError) as e:
            raise InvalidModelOutputError(
                f"Model output cannot be converted to float: {val!r}"
            ) from e

        # Check finite
        if math.isnan(prob) or math.isinf(prob):
            raise InvalidModelOutputError(
                f"Model returned non-finite probability value: {prob!r}"
            )

        # Check probability bounds [0.0, 1.0]
        if prob < 0.0 or prob > 1.0:
            raise InvalidModelOutputError(
                f"Model returned probability outside valid range [0.0, 1.0]: {prob!r}"
            )

        return prob

    def predict(
        self,
        transaction: Mapping[str, Any],
        transaction_id: Optional[str] = None,
    ) -> PredictionResponse:
        """
        Executes single-transaction inference:
        1. Validates transaction_id.
        2. Preprocesses input transaction into a 1x29 DataFrame.
        3. Evaluates fraud probability with LightGBM Booster.
        4. Applies authoritative frozen threshold.
        5. Returns structured PredictionResponse.

        Parameters
        ----------
        transaction : Mapping[str, Any]
            Raw transaction record with 29 canonical features.
        transaction_id : Optional[str]
            Optional tracking identifier.

        Returns
        -------
        PredictionResponse
            Structured inference decision, probability, and metadata.
        """
        start_time = time.perf_counter()

        # Isolate transaction_id if provided inside transaction mapping
        if "transaction_id" in transaction:
            transaction_id = transaction_id or transaction["transaction_id"]
            transaction_features = {k: v for k, v in transaction.items() if k != "transaction_id"}
        else:
            transaction_features = transaction

        valid_tid = self.validate_transaction_id(transaction_id)

        # 1. Preprocess Transaction
        try:
            df = self.preprocessor.transform(transaction_features)
        except PreprocessorError:
            # Let domain preprocessing errors bubble up cleanly
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during transaction preprocessing (TID=%s): %s",
                valid_tid or "N/A",
                str(e),
            )
            raise PreprocessorError(f"Transaction preprocessing failed: {e}") from e

        # Validate preprocessor output shape
        if len(df) != 1 or len(df.columns) != EXPECTED_FEATURE_COUNT:
            raise PreprocessedOutputValidationError(
                f"Preprocessor output shape invalid: expected (1, {EXPECTED_FEATURE_COUNT}), "
                f"got ({len(df)}, {len(df.columns)})"
            )

        # 2. Execute LightGBM Prediction
        try:
            raw_output = self.booster.predict(df)
        except Exception as e:
            logger.error(
                "LightGBM prediction execution failed (TID=%s, Model=%s): %s",
                valid_tid or "N/A",
                self.model_version,
                str(e),
            )
            raise ModelPredictionError(f"LightGBM prediction failed: {e}") from e

        # 3. Validate Probability Output
        prob_full = self._validate_model_output(raw_output)

        # 4. Apply Frozen Threshold Decision Logic
        # Keep full precision for decisioning
        is_fraud = prob_full >= self.threshold
        decision = "FRAUD" if is_fraud else "LEGIT"

        # 5. Evaluate Risk Classification, Reason Codes, and Metadata (Phase 6)
        risk_eval = self.risk_classifier.evaluate(
            probability=prob_full,
            frozen_threshold=self.threshold,
            decision=decision,
            model_version=self.model_version,
            model_status="ACTIVE_LOADED",
            feature_count=EXPECTED_FEATURE_COUNT,
            inference_mode="single",
        )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Safe structured logging (no raw feature values or PII logged)
        logger.info(
            "Inference complete: TID=%s | Version=%s | Decision=%s | Risk=%s | Prob=%.4f | Time=%.2fms",
            valid_tid or "N/A",
            self.model_version,
            decision,
            risk_eval.risk_level.value,
            prob_full,
            duration_ms,
        )

        return PredictionResponse(
            transaction_id=valid_tid,
            prediction=decision,
            fraud_probability=round(prob_full, 6),
            threshold=self.threshold,
            model_version=self.model_version,
            model_status="ACTIVE_LOADED",
            feature_count=EXPECTED_FEATURE_COUNT,
            execution_time_ms=round(duration_ms, 3),
            risk_level=risk_eval.risk_level.value,
            reason_codes=risk_eval.reason_codes,
            explanation_metadata=risk_eval.explanation_metadata,
            status="SUCCESS",
        )


# ---------------------------------------------------------------------------
# Default Engine Singleton
# ---------------------------------------------------------------------------
_DEFAULT_ENGINE: Optional[SingleTransactionInferenceEngine] = None


def get_inference_engine(
    model_version: str = APPROVED_DEFAULT_MODEL_VERSION,
) -> SingleTransactionInferenceEngine:
    """Returns the default singleton inference engine instance."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None or _DEFAULT_ENGINE.model_version != model_version:
        _DEFAULT_ENGINE = SingleTransactionInferenceEngine(model_version=model_version)
    return _DEFAULT_ENGINE
