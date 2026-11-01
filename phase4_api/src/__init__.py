"""
Phase 4 BAF Adaptive Fraud Detection Service package.
"""
import pathlib
import sys

# Ensure repository root is on sys.path so 'phase4_api' is always resolvable
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from phase4_api.src.batch_inference_engine import (
    BatchInferenceError,
    BatchPredictionResponse,
    BatchRecordResult,
    BatchSizeLimitExceededError,
    BatchTransactionInferenceEngine,
    DEFAULT_MAX_BATCH_SIZE,
    InvalidBatchInputError,
    get_batch_inference_engine,
)
from phase4_api.src.inference_engine import (
    InferenceEngineError,
    InvalidModelOutputError,
    InvalidTransactionIdError,
    ModelNotReadyError,
    ModelPredictionError,
    PredictionResponse,
    SingleTransactionInferenceEngine,
    get_inference_engine,
)
from phase4_api.src.model_loader import (
    APPROVED_DEFAULT_MODEL_VERSION,
    CANONICAL_CATEGORICAL_COLS,
    CANONICAL_FEATURE_COLS,
    CANONICAL_NUMERICAL_COLS,
    EXPECTED_FEATURE_COUNT,
    FeatureSchemaMismatchError,
    LoadedModelBundle,
    ModelArtifactNotFoundError,
    ModelCorruptedError,
    ModelInfo,
    ModelLoaderError,
    ModelMetadataError,
    Phase4ModelLoader,
    ThresholdConfigurationError,
    UnsupportedModelVersionError,
    clear_model_cache,
    find_project_root,
    get_model_bundle,
    get_model_info,
)
from phase4_api.src.prediction_schema import (
    InvalidSchemaDataError,
    PredictionResultModel,
    PredictionSchemaError,
    ReasonCodeModel,
    StandardizedPredictionResult,
)
from phase4_api.src.risk_classifier import (
    DEFAULT_LOW_THRESHOLD,
    DEFAULT_MEDIUM_THRESHOLD,
    DEFAULT_RISK_POLICY_VERSION,
    InvalidProbabilityError,
    InvalidRiskPolicyError,
    ReasonCode,
    ReasonCodeGenerationError,
    RiskClassificationError,
    RiskClassifier,
    RiskEvaluationResult,
    RiskLevel,
    RiskPolicy,
    get_risk_classifier,
)
from phase4_api.src.main import app, create_app
from phase4_api.src.transaction_preprocessor import (
    CANONICAL_CATEGORICAL_LEVELS,
    EmptyTransactionError,
    ExtraFeatureError,
    InvalidCategoricalValueError,
    InvalidNumericalValueError,
    InvalidTransactionFormatError,
    MissingFeatureError,
    PreprocessedOutputValidationError,
    PreprocessorError,
    TransactionPreprocessor,
    preprocess_transaction,
)

__version__ = "1.0.0"

__all__ = [
    # FastAPI Application exports (Phase 7)
    "app",
    "create_app",
    # Risk classification & explanation exports (Phase 6)
    "RiskLevel",
    "ReasonCode",
    "RiskPolicy",
    "RiskClassifier",
    "RiskEvaluationResult",
    "get_risk_classifier",
    "RiskClassificationError",
    "InvalidProbabilityError",
    "InvalidRiskPolicyError",
    "ReasonCodeGenerationError",
    "DEFAULT_RISK_POLICY_VERSION",
    "DEFAULT_LOW_THRESHOLD",
    "DEFAULT_MEDIUM_THRESHOLD",
    "StandardizedPredictionResult",
    "PredictionResultModel",
    "ReasonCodeModel",
    "PredictionSchemaError",
    "InvalidSchemaDataError",
    # Batch inference exports (Phase 5)
    "BatchTransactionInferenceEngine",
    "BatchPredictionResponse",
    "BatchRecordResult",
    "get_batch_inference_engine",
    "BatchInferenceError",
    "InvalidBatchInputError",
    "BatchSizeLimitExceededError",
    "DEFAULT_MAX_BATCH_SIZE",
    # Single inference engine exports (Phase 4)
    "SingleTransactionInferenceEngine",
    "PredictionResponse",
    "get_inference_engine",
    "InferenceEngineError",
    "InvalidTransactionIdError",
    "ModelNotReadyError",
    "ModelPredictionError",
    "InvalidModelOutputError",
    # Model loader exports (Phase 2)
    "CANONICAL_CATEGORICAL_COLS",
    "CANONICAL_FEATURE_COLS",
    "CANONICAL_NUMERICAL_COLS",
    "EXPECTED_FEATURE_COUNT",
    "APPROVED_DEFAULT_MODEL_VERSION",
    "FeatureSchemaMismatchError",
    "LoadedModelBundle",
    "ModelArtifactNotFoundError",
    "ModelCorruptedError",
    "ModelInfo",
    "ModelLoaderError",
    "ModelMetadataError",
    "Phase4ModelLoader",
    "ThresholdConfigurationError",
    "UnsupportedModelVersionError",
    "clear_model_cache",
    "find_project_root",
    "get_model_bundle",
    "get_model_info",
    # Preprocessor exports (Phase 3)
    "CANONICAL_CATEGORICAL_LEVELS",
    "EmptyTransactionError",
    "ExtraFeatureError",
    "InvalidCategoricalValueError",
    "InvalidNumericalValueError",
    "InvalidTransactionFormatError",
    "MissingFeatureError",
    "PreprocessedOutputValidationError",
    "PreprocessorError",
    "TransactionPreprocessor",
    "preprocess_transaction",
]
