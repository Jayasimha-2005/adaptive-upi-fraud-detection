"""
phase4_api/src/batch_inference_engine.py
========================================
Reusable Batch Transaction Inference Engine for Phase 4 BAF Fraud Detection.

Orchestrates:
1. Ordered batch ingestion with configurable batch size limits.
2. Per-record validation with partial-success architecture.
3. Separation of envelope metadata (transaction_id) from 29 model features.
4. Vectorized batch preprocessing using Phase 3 TransactionPreprocessor logic.
5. Exactly one vectorized LightGBM Booster prediction call per valid batch.
6. Authoritative frozen threshold decisioning (0.8964321608040201).
7. Strict preservation of original input sequence and transaction IDs.
8. Structured, sanitized batch response formatting with execution metrics.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from phase4_api.src.model_loader import (
    APPROVED_DEFAULT_MODEL_VERSION,
    CANONICAL_CATEGORICAL_COLS,
    CANONICAL_FEATURE_COLS,
    CANONICAL_NUMERICAL_COLS,
    EXPECTED_FEATURE_COUNT,
    LoadedModelBundle,
    ModelLoaderError,
    Phase4ModelLoader,
    ThresholdConfigurationError,
    get_model_bundle,
)
from phase4_api.src.transaction_preprocessor import (
    CANONICAL_CATEGORICAL_LEVELS,
    CANONICAL_FEATURE_SET,
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
logger = logging.getLogger("phase4_api.batch_inference_engine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [BatchInferenceEngine] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default batch limit (prevents memory exhaustion under extreme concurrency)
DEFAULT_MAX_BATCH_SIZE: int = 5000


# ---------------------------------------------------------------------------
# Batch Exception Hierarchy
# ---------------------------------------------------------------------------
class BatchInferenceError(Exception):
    """Base exception for all batch inference failures."""


class InvalidBatchInputError(BatchInferenceError):
    """Raised when the batch container is not an ordered list/sequence."""


class BatchSizeLimitExceededError(BatchInferenceError):
    """Raised when an incoming batch exceeds the configured max_batch_size."""


# ---------------------------------------------------------------------------
# Structured Result Representations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BatchRecordResult:
    """Individual prediction outcome corresponding to a single input index."""

    input_index: int
    transaction_id: Optional[str]
    status: str  # "success" | "error"
    prediction: Optional[str]  # "FRAUD" | "LEGIT" | None
    fraud_probability: Optional[float]  # Display probability rounded to 6 decimal places | None
    threshold: Optional[float]  # Authoritative frozen threshold | None
    model_version: Optional[str]  # e.g., "v3" | None
    feature_count: Optional[int]  # 29 | None
    error: Optional[Dict[str, str]]  # {"code": ..., "message": ...} | None
    risk_level: Optional[str] = None  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | None
    reason_codes: Optional[List[ReasonCode]] = None
    explanation_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts to a JSON-serializable dictionary."""
        serialized_reasons = (
            [rc.to_dict() if hasattr(rc, "to_dict") else rc for rc in self.reason_codes]
            if self.reason_codes is not None
            else None
        )
        return {
            "input_index": self.input_index,
            "transaction_id": self.transaction_id,
            "status": self.status,
            "prediction": self.prediction,
            "fraud_probability": self.fraud_probability,
            "threshold": self.threshold,
            "risk_level": self.risk_level,
            "model_version": self.model_version,
            "feature_count": self.feature_count,
            "reason_codes": serialized_reasons,
            "explanation_metadata": self.explanation_metadata,
            "error": self.error,
        }


@dataclass(frozen=True)
class BatchPredictionResponse:
    """Aggregated batch response with summary telemetry and ordered records."""

    batch_status: str  # "all_success" | "partial_success" | "all_failed" | "empty"
    total_transactions: int
    successful_transactions: int
    failed_transactions: int
    model_version: str
    threshold: float
    results: List[BatchRecordResult]
    execution_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts to a JSON-serializable dictionary."""
        return {
            "batch_status": self.batch_status,
            "total_transactions": self.total_transactions,
            "successful_transactions": self.successful_transactions,
            "failed_transactions": self.failed_transactions,
            "model_version": self.model_version,
            "threshold": self.threshold,
            "results": [r.to_dict() for r in self.results],
            "execution_time_ms": self.execution_time_ms,
        }


# ---------------------------------------------------------------------------
# Batch Transaction Inference Engine
# ---------------------------------------------------------------------------
class BatchTransactionInferenceEngine:
    """
    High-performance batch inference engine for real-time BAF fraud detection.
    Preprocesses valid records into an aggregated DataFrame, executes one
    vectorized LightGBM prediction call, and applies the frozen threshold.
    """

    def __init__(
        self,
        model_version: str = APPROVED_DEFAULT_MODEL_VERSION,
        model_bundle: Optional[LoadedModelBundle] = None,
        preprocessor: Optional[TransactionPreprocessor] = None,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
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
        max_batch_size : int, default=5000
            Configurable maximum batch size limit.
        risk_classifier : Optional[RiskClassifier]
            Optional risk classification engine for Phase 6 evaluation.
        """
        self.model_version = model_version
        self.max_batch_size = max_batch_size
        self.preprocessor = preprocessor or TransactionPreprocessor(
            allow_unseen_categories=False,
            allow_missing_numerical=True,
        )
        self.risk_classifier = risk_classifier or get_risk_classifier()

        # Initialize or assign model bundle
        if model_bundle is not None:
            self._bundle = model_bundle
        else:
            self._bundle = get_model_bundle(model_version=self.model_version)

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
            raise ValueError(
                f"transaction_id must be a string, received {type(transaction_id).__name__}."
            )

        tid = transaction_id.strip()
        if not tid:
            raise ValueError("transaction_id cannot be empty or whitespace.")

        if len(tid) > 128:
            raise ValueError(
                f"transaction_id exceeds maximum length of 128 characters ({len(tid)} chars)."
            )

        # Disallow control characters
        if any(ord(c) < 32 or ord(c) == 127 for c in tid):
            raise ValueError("transaction_id contains invalid control characters.")

        return tid

    def _validate_batch_container(self, batch: Any) -> None:
        """Validates that batch is a non-None list/sequence within max_batch_size."""
        if batch is None:
            raise InvalidBatchInputError("Batch input cannot be None.")

        # Must be a list or tuple (reject dict, set, string, generator)
        if not isinstance(batch, (list, tuple)):
            raise InvalidBatchInputError(
                f"Batch input must be a list or sequence, received {type(batch).__name__}."
            )

        if len(batch) > self.max_batch_size:
            raise BatchSizeLimitExceededError(
                f"Batch size {len(batch)} exceeds maximum allowed limit of {self.max_batch_size}."
            )

    def _validate_record_features(
        self,
        record: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Validates individual record features using Phase 3 preprocessor rules.
        Returns a dictionary of validated values.
        """
        record_keys = set(record.keys())

        # Check missing features
        missing = CANONICAL_FEATURE_SET - record_keys
        if missing:
            sorted_missing = sorted(missing)
            raise MissingFeatureError(
                f"Missing {len(missing)} required features: {sorted_missing}"
            )

        # Check extra features
        extra = record_keys - CANONICAL_FEATURE_SET
        if extra:
            sorted_extra = sorted(extra)
            raise ExtraFeatureError(
                f"Unexpected extra features: {sorted_extra}. Expected only 29 canonical features."
            )

        validated: Dict[str, Any] = {}

        # Validate numerical features
        for col in CANONICAL_NUMERICAL_COLS:
            val = record[col]
            validated[col] = self.preprocessor.process_numerical(col, val)

        # Validate categorical features
        for col in CANONICAL_CATEGORICAL_COLS:
            val = record[col]
            validated[col] = self.preprocessor.process_categorical(col, val)

        return validated

    def _validate_vectorized_output(
        self,
        raw_pred: Any,
        expected_len: int,
    ) -> np.ndarray:
        """
        Validates raw vectorized booster prediction output:
        Must be a 1D numeric array of length expected_len with finite floats in [0.0, 1.0].
        """
        if raw_pred is None:
            raise ValueError("Model prediction output is None.")

        if isinstance(raw_pred, (list, tuple)):
            pred_arr = np.array(raw_pred, dtype=float)
        elif isinstance(raw_pred, np.ndarray):
            pred_arr = raw_pred.flatten()
        elif isinstance(raw_pred, (float, int, np.floating)):
            pred_arr = np.array([float(raw_pred)])
        else:
            raise ValueError(f"Unexpected prediction output type: {type(raw_pred).__name__}")

        if len(pred_arr) != expected_len:
            raise ValueError(
                f"Output length mismatch: expected {expected_len} probabilities, got {len(pred_arr)}."
            )

        # Check numeric, finite, bounds
        if not np.issubdtype(pred_arr.dtype, np.number):
            raise ValueError("Model output contains non-numeric values.")

        if np.any(np.isnan(pred_arr)) or np.any(np.isinf(pred_arr)):
            raise ValueError("Model output contains non-finite values (NaN or Inf).")

        if np.any((pred_arr < 0.0) | (pred_arr > 1.0)):
            raise ValueError("Model output contains probabilities outside [0.0, 1.0].")

        return pred_arr

    def predict_batch(
        self,
        batch: Sequence[Any],
        transaction_ids: Optional[Sequence[Optional[str]]] = None,
    ) -> BatchPredictionResponse:
        """
        Executes batch transaction inference:
        1. Validates batch container and size constraints.
        2. Validates records with partial-success error capture.
        3. Assembles one aggregated DataFrame for all valid records.
        4. Invokes one vectorized booster.predict() call.
        5. Applies authoritative frozen threshold decisioning.
        6. Reassembles results in exact original input sequence.

        Parameters
        ----------
        batch : Sequence[Any]
            List of transaction dictionaries.
        transaction_ids : Optional[Sequence[Optional[str]]]
            Optional external sequence of transaction IDs matching batch length.

        Returns
        -------
        BatchPredictionResponse
            Summary telemetry and per-record results.
        """
        start_time = time.perf_counter()

        # 1. Validate container
        self._validate_batch_container(batch)
        total_count = len(batch)

        # Handle empty batch gracefully
        if total_count == 0:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return BatchPredictionResponse(
                batch_status="empty",
                total_transactions=0,
                successful_transactions=0,
                failed_transactions=0,
                model_version=self.model_version,
                threshold=self.threshold,
                results=[],
                execution_time_ms=round(duration_ms, 3),
            )

        # Validate transaction_ids length if provided externally
        if transaction_ids is not None and len(transaction_ids) != total_count:
            raise InvalidBatchInputError(
                f"transaction_ids length ({len(transaction_ids)}) must match batch length ({total_count})."
            )

        results_map: Dict[int, BatchRecordResult] = {}
        valid_indices: List[int] = []
        valid_tids: List[Optional[str]] = []
        valid_records_data: List[Dict[str, Any]] = []

        # 2. Per-record validation & envelope isolation
        for idx, item in enumerate(batch):
            # A. Check dictionary format
            if not isinstance(item, Mapping):
                results_map[idx] = BatchRecordResult(
                    input_index=idx,
                    transaction_id=None,
                    status="error",
                    prediction=None,
                    fraud_probability=None,
                    threshold=None,
                    model_version=None,
                    feature_count=None,
                    error={
                        "code": "INVALID_RECORD_FORMAT",
                        "message": f"Record must be a dictionary/mapping, got {type(item).__name__}.",
                    },
                )
                continue

            # B. Extract and validate transaction_id
            # Can be inside record or supplied externally
            raw_tid: Optional[str] = None
            if "transaction_id" in item:
                raw_tid = item["transaction_id"]
                # Create a shallow copy without transaction_id so preprocessor does not see extra key
                item_features = {k: v for k, v in item.items() if k != "transaction_id"}
            else:
                raw_tid = transaction_ids[idx] if transaction_ids is not None else None
                item_features = dict(item)

            try:
                valid_tid = self.validate_transaction_id(raw_tid)
            except ValueError as e:
                results_map[idx] = BatchRecordResult(
                    input_index=idx,
                    transaction_id=str(raw_tid) if raw_tid is not None else None,
                    status="error",
                    prediction=None,
                    fraud_probability=None,
                    threshold=None,
                    model_version=None,
                    feature_count=None,
                    error={
                        "code": "INVALID_TRANSACTION_ID",
                        "message": str(e),
                    },
                )
                continue

            # C. Validate feature schema and values
            try:
                validated_vals = self._validate_record_features(item_features)
                valid_indices.append(idx)
                valid_tids.append(valid_tid)
                valid_records_data.append(validated_vals)
            except MissingFeatureError as e:
                results_map[idx] = BatchRecordResult(
                    input_index=idx,
                    transaction_id=valid_tid,
                    status="error",
                    prediction=None,
                    fraud_probability=None,
                    threshold=None,
                    model_version=None,
                    feature_count=None,
                    error={"code": "MISSING_FEATURE", "message": str(e)},
                )
            except ExtraFeatureError as e:
                results_map[idx] = BatchRecordResult(
                    input_index=idx,
                    transaction_id=valid_tid,
                    status="error",
                    prediction=None,
                    fraud_probability=None,
                    threshold=None,
                    model_version=None,
                    feature_count=None,
                    error={"code": "EXTRA_FEATURE", "message": str(e)},
                )
            except InvalidNumericalValueError as e:
                results_map[idx] = BatchRecordResult(
                    input_index=idx,
                    transaction_id=valid_tid,
                    status="error",
                    prediction=None,
                    fraud_probability=None,
                    threshold=None,
                    model_version=None,
                    feature_count=None,
                    error={"code": "INVALID_NUMERICAL_VALUE", "message": str(e)},
                )
            except InvalidCategoricalValueError as e:
                results_map[idx] = BatchRecordResult(
                    input_index=idx,
                    transaction_id=valid_tid,
                    status="error",
                    prediction=None,
                    fraud_probability=None,
                    threshold=None,
                    model_version=None,
                    feature_count=None,
                    error={"code": "INVALID_CATEGORICAL_VALUE", "message": str(e)},
                )
            except Exception as e:
                results_map[idx] = BatchRecordResult(
                    input_index=idx,
                    transaction_id=valid_tid,
                    status="error",
                    prediction=None,
                    fraud_probability=None,
                    threshold=None,
                    model_version=None,
                    feature_count=None,
                    error={"code": "VALIDATION_ERROR", "message": str(e)},
                )

        # 3. Vectorized batch preprocessing & inference for valid records
        num_valid = len(valid_records_data)
        if num_valid > 0:
            # Construct column-oriented aggregated DataFrame
            row_dict: Dict[str, Any] = {}
            for col in CANONICAL_NUMERICAL_COLS:
                row_dict[col] = [rec[col] for rec in valid_records_data]

            for col in CANONICAL_CATEGORICAL_COLS:
                levels = list(CANONICAL_CATEGORICAL_LEVELS[col])
                row_dict[col] = pd.Categorical(
                    [rec[col] for rec in valid_records_data],
                    categories=levels,
                )

            df_batch = pd.DataFrame(row_dict)[list(CANONICAL_FEATURE_COLS)]

            # 4. Vectorized Booster Prediction
            try:
                raw_preds = self.booster.predict(df_batch)
                validated_probs = self._validate_vectorized_output(raw_preds, num_valid)

                # 5. Threshold decisioning and Phase 6 Risk Evaluation
                for sub_idx, (orig_idx, tid) in enumerate(zip(valid_indices, valid_tids)):
                    prob = float(validated_probs[sub_idx])
                    is_fraud = prob >= self.threshold
                    decision = "FRAUD" if is_fraud else "LEGIT"

                    risk_eval = self.risk_classifier.evaluate(
                        probability=prob,
                        frozen_threshold=self.threshold,
                        decision=decision,
                        model_version=self.model_version,
                        model_status="ACTIVE_LOADED",
                        feature_count=EXPECTED_FEATURE_COUNT,
                        inference_mode="batch",
                    )

                    results_map[orig_idx] = BatchRecordResult(
                        input_index=orig_idx,
                        transaction_id=tid,
                        status="success",
                        prediction=decision,
                        fraud_probability=round(prob, 6),
                        threshold=self.threshold,
                        model_version=self.model_version,
                        feature_count=EXPECTED_FEATURE_COUNT,
                        risk_level=risk_eval.risk_level.value,
                        reason_codes=risk_eval.reason_codes,
                        explanation_metadata=risk_eval.explanation_metadata,
                        error=None,
                    )
            except Exception as e:
                logger.error("Vectorized booster prediction failed: %s", str(e))
                for orig_idx, tid in zip(valid_indices, valid_tids):
                    results_map[orig_idx] = BatchRecordResult(
                        input_index=orig_idx,
                        transaction_id=tid,
                        status="error",
                        prediction=None,
                        fraud_probability=None,
                        threshold=None,
                        model_version=None,
                        feature_count=None,
                        error={"code": "PREDICTION_ERROR", "message": f"LightGBM model inference failed: {e}"},
                    )

        # 6. Reassemble results in exact original input order
        ordered_results = [results_map[i] for i in range(total_count)]
        success_count = sum(1 for r in ordered_results if r.status == "success")
        failed_count = total_count - success_count

        if success_count == total_count:
            batch_status = "all_success"
        elif success_count == 0:
            batch_status = "all_failed"
        else:
            batch_status = "partial_success"

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Safe structured logging (no raw transaction features or PII logged)
        logger.info(
            "Batch inference complete: Total=%d | Success=%d | Failed=%d | Status=%s | Time=%.2fms",
            total_count,
            success_count,
            failed_count,
            batch_status,
            duration_ms,
        )

        return BatchPredictionResponse(
            batch_status=batch_status,
            total_transactions=total_count,
            successful_transactions=success_count,
            failed_transactions=failed_count,
            model_version=self.model_version,
            threshold=self.threshold,
            results=ordered_results,
            execution_time_ms=round(duration_ms, 3),
        )


# ---------------------------------------------------------------------------
# Default Batch Engine Singleton
# ---------------------------------------------------------------------------
_DEFAULT_BATCH_ENGINE: Optional[BatchTransactionInferenceEngine] = None


def get_batch_inference_engine(
    model_version: str = APPROVED_DEFAULT_MODEL_VERSION,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> BatchTransactionInferenceEngine:
    """Returns the default singleton batch inference engine instance."""
    global _DEFAULT_BATCH_ENGINE
    if (
        _DEFAULT_BATCH_ENGINE is None
        or _DEFAULT_BATCH_ENGINE.model_version != model_version
        or _DEFAULT_BATCH_ENGINE.max_batch_size != max_batch_size
    ):
        _DEFAULT_BATCH_ENGINE = BatchTransactionInferenceEngine(
            model_version=model_version,
            max_batch_size=max_batch_size,
        )
    return _DEFAULT_BATCH_ENGINE
