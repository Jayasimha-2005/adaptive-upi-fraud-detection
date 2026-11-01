"""
phase4_api/src/model_loader.py
==============================
Reusable, production-grade LightGBM model loader for Phase 4 BAF models.

Key Capabilities:
1. Cross-platform loading with automatic Windows CRLF -> LF normalization.
2. Canonical 29-feature schema & ordering validation.
3. Authoritative frozen threshold ingestion (0.8964321608040201).
4. Metadata integrity verification.
5. In-memory caching / singleton pattern to prevent redundant disk I/O.
6. Explicit, structured error hierarchy.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb

# ---------------------------------------------------------------------------
# Canonical BAF Feature Schema (Protocol v1.1 Locked Definition)
# ---------------------------------------------------------------------------
CANONICAL_NUMERICAL_COLS: Tuple[str, ...] = (
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "session_length_in_minutes",
    "keep_alive_session",
    "device_distinct_emails_8w",
)

CANONICAL_CATEGORICAL_COLS: Tuple[str, ...] = (
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
)

CANONICAL_FEATURE_COLS: Tuple[str, ...] = (
    CANONICAL_NUMERICAL_COLS + CANONICAL_CATEGORICAL_COLS
)
EXPECTED_FEATURE_COUNT: int = len(CANONICAL_FEATURE_COLS)  # 29
APPROVED_DEFAULT_MODEL_VERSION: str = "v3"
SUPPORTED_MODEL_VERSIONS: Tuple[str, ...] = ("v1", "v2", "v3")


# ---------------------------------------------------------------------------
# Error Hierarchy
# ---------------------------------------------------------------------------
class ModelLoaderError(Exception):
    """Base exception for all model loading errors in Phase 4 API."""


class ModelArtifactNotFoundError(ModelLoaderError):
    """Raised when the specified model text file cannot be located."""


class ModelCorruptedError(ModelLoaderError):
    """Raised when the model file cannot be parsed or initialized by LightGBM."""


class ModelMetadataError(ModelLoaderError):
    """Raised when model metadata file is missing, unreadable, or invalid."""


class FeatureSchemaMismatchError(ModelLoaderError):
    """Raised when the loaded booster's features or ordering diverge from canonical schema."""


class ThresholdConfigurationError(ModelLoaderError):
    """Raised when the frozen threshold artifact is missing or invalid."""


class UnsupportedModelVersionError(ModelLoaderError):
    """Raised when an unrecognized model version label is requested."""


# ---------------------------------------------------------------------------
# Structured Model Representation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelInfo:
    """Public, safe representation of model telemetry and configuration."""

    model_version: str
    model_artifact_path: str
    num_features: int
    feature_names: List[str]
    num_trees: int
    frozen_threshold: float
    model_loading_status: str
    training_months: List[int]
    model_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    def to_public_dict(self) -> Dict[str, Any]:
        """Convert to sanitized dictionary omitting detailed filesystem paths."""
        d = self.to_dict()
        d["model_artifact_path"] = pathlib.Path(self.model_artifact_path).name
        return d


@dataclass
class LoadedModelBundle:
    """Internal container holding the active LightGBM Booster and verified metadata."""

    booster: lgb.Booster
    info: ModelInfo
    metadata: Dict[str, Any]
    frozen_threshold: float


# ---------------------------------------------------------------------------
# Path Discovery Helpers
# ---------------------------------------------------------------------------
def find_project_root(start_path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """
    Locates the project repository root by ascending directories until
    the 'experiments' or 'phase4_api' markers are found.
    """
    current = (start_path or pathlib.Path(__file__)).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "experiments" / "phase4_drift_adaptation").is_dir():
            return parent
        if (parent / "phase4_api").is_dir() and (parent / "README.md").is_file():
            return parent
    raise ModelArtifactNotFoundError(
        f"Could not locate project repository root ascending from {current}"
    )


# ---------------------------------------------------------------------------
# Core Model Loader
# ---------------------------------------------------------------------------
class Phase4ModelLoader:
    """
    Production-grade loader for Phase 4 LightGBM models.
    Supports caching, validation, and authoritative threshold loading.
    """

    def __init__(
        self,
        project_root: Optional[pathlib.Path] = None,
        model_version: str = APPROVED_DEFAULT_MODEL_VERSION,
    ) -> None:
        if model_version not in SUPPORTED_MODEL_VERSIONS:
            raise UnsupportedModelVersionError(
                f"Model version '{model_version}' is not supported. "
                f"Allowed versions: {SUPPORTED_MODEL_VERSIONS}"
            )
        self.project_root = project_root or find_project_root()
        self.model_version = model_version
        self._lock = threading.Lock()
        self._cached_bundle: Optional[LoadedModelBundle] = None

    @property
    def models_dir(self) -> pathlib.Path:
        return (
            self.project_root
            / "experiments"
            / "phase4_drift_adaptation"
            / "artifacts"
            / "models"
        )

    @property
    def thresholds_dir(self) -> pathlib.Path:
        return (
            self.project_root
            / "experiments"
            / "phase4_drift_adaptation"
            / "artifacts"
            / "thresholds"
        )

    @property
    def model_artifact_path(self) -> pathlib.Path:
        return self.models_dir / f"lgbm_{self.model_version}.txt"

    @property
    def metadata_path(self) -> pathlib.Path:
        return self.models_dir / f"lgbm_{self.model_version}_meta.json"

    @property
    def threshold_path(self) -> pathlib.Path:
        return self.thresholds_dir / "frozen_threshold_v1.json"

    def load_frozen_threshold(self) -> float:
        """
        Loads and validates the permanently frozen prediction threshold.
        """
        t_path = self.threshold_path
        if not t_path.is_file():
            raise ThresholdConfigurationError(
                f"Authoritative threshold artifact missing at: {t_path}"
            )

        try:
            with open(t_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ThresholdConfigurationError(
                f"Failed to read threshold artifact at {t_path}: {e}"
            ) from e

        if "threshold" not in data:
            raise ThresholdConfigurationError(
                f"Threshold artifact at {t_path} missing 'threshold' key."
            )

        try:
            val = float(data["threshold"])
        except (ValueError, TypeError) as e:
            raise ThresholdConfigurationError(
                f"Threshold value in {t_path} is not a valid float: {data.get('threshold')}"
            ) from e

        if not math.isfinite(val) or not (0.0 < val < 1.0):
            raise ThresholdConfigurationError(
                f"Threshold value {val} is outside valid probability interval (0.0, 1.0)"
            )

        return val

    def load_metadata(self) -> Dict[str, Any]:
        """
        Loads and verifies the model's companion JSON metadata file.
        """
        m_path = self.metadata_path
        if not m_path.is_file():
            raise ModelMetadataError(
                f"Model metadata file missing at: {m_path}"
            )

        try:
            with open(m_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            raise ModelMetadataError(
                f"Failed to parse model metadata JSON at {m_path}: {e}"
            ) from e

        if not isinstance(meta, dict):
            raise ModelMetadataError(
                f"Model metadata at {m_path} is not a valid JSON dictionary."
            )

        # Validate required metadata fields
        required_fields = ("version_label", "training_months", "n_train", "model_hash")
        missing_fields = [f for f in required_fields if f not in meta]
        if missing_fields:
            raise ModelMetadataError(
                f"Model metadata at {m_path} is missing required fields: {missing_fields}"
            )

        if meta.get("version_label") != self.model_version:
            raise ModelMetadataError(
                f"Metadata version mismatch in {m_path}: expected '{self.model_version}', "
                f"found '{meta.get('version_label')}'"
            )

        return meta

    def load_booster(self, artifact_path: Optional[pathlib.Path] = None) -> lgb.Booster:
        """
        Loads LightGBM Booster from text file, normalizing Windows CRLF to LF.
        """
        a_path = artifact_path or self.model_artifact_path
        if not a_path.is_file():
            raise ModelArtifactNotFoundError(
                f"LightGBM model artifact not found at: {a_path}"
            )

        try:
            raw_bytes = a_path.read_bytes()
            if not raw_bytes:
                raise ModelCorruptedError(f"Model file at {a_path} is empty (0 bytes).")
            # Normalize CRLF to LF to guarantee cross-platform binary parser consistency
            normalized_bytes = raw_bytes.replace(b"\r\n", b"\n")
            model_str = normalized_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ModelCorruptedError(
                f"Model file at {a_path} is not valid UTF-8 text: {e}"
            ) from e
        except Exception as e:
            raise ModelCorruptedError(
                f"Failed to read model artifact at {a_path}: {e}"
            ) from e

        try:
            booster = lgb.Booster(model_str=model_str)
        except Exception as e:
            raise ModelCorruptedError(
                f"LightGBM Booster initialization failed for {a_path}: {e}"
            ) from e

        return booster

    def validate_features(self, booster: lgb.Booster) -> List[str]:
        """
        Validates feature count, exact names, and sequence against the canonical schema.
        """
        actual_count = booster.num_feature()
        if actual_count != EXPECTED_FEATURE_COUNT:
            raise FeatureSchemaMismatchError(
                f"Feature count mismatch: model has {actual_count} features, "
                f"expected exactly {EXPECTED_FEATURE_COUNT}."
            )

        actual_names = booster.feature_name()
        if len(actual_names) != EXPECTED_FEATURE_COUNT:
            raise FeatureSchemaMismatchError(
                f"Reported feature_name() count {len(actual_names)} != {EXPECTED_FEATURE_COUNT}."
            )

        # Check exact ordering
        if list(actual_names) != list(CANONICAL_FEATURE_COLS):
            diffs = []
            for i, (act, exp) in enumerate(zip(actual_names, CANONICAL_FEATURE_COLS)):
                if act != exp:
                    diffs.append(f"Index {i}: actual '{act}' != expected '{exp}'")
            raise FeatureSchemaMismatchError(
                f"Feature ordering mismatch in model ({len(diffs)} disparities): "
                f"{'; '.join(diffs[:5])}"
            )

        return list(actual_names)

    def load(self, force_reload: bool = False) -> LoadedModelBundle:
        """
        Loads, validates, and caches the LightGBM model bundle.
        Thread-safe singleton behavior per loader instance.
        """
        with self._lock:
            if self._cached_bundle is not None and not force_reload:
                return self._cached_bundle

            # 1. Load frozen threshold
            threshold = self.load_frozen_threshold()

            # 2. Load model metadata
            metadata = self.load_metadata()

            # 3. Load booster with CRLF normalization
            booster = self.load_booster()

            # 4. Validate schema and ordering
            feature_names = self.validate_features(booster)

            # 5. Build ModelInfo
            info = ModelInfo(
                model_version=self.model_version,
                model_artifact_path=str(self.model_artifact_path.resolve()),
                num_features=len(feature_names),
                feature_names=feature_names,
                num_trees=booster.num_trees(),
                frozen_threshold=threshold,
                model_loading_status="ACTIVE_LOADED",
                training_months=list(metadata.get("training_months", [])),
                model_hash=str(metadata.get("model_hash", "")),
            )

            bundle = LoadedModelBundle(
                booster=booster,
                info=info,
                metadata=metadata,
                frozen_threshold=threshold,
            )
            self._cached_bundle = bundle
            return bundle


# ---------------------------------------------------------------------------
# Global Singleton Cache & Factory API
# ---------------------------------------------------------------------------
_GLOBAL_LOADER_REGISTRY: Dict[str, Phase4ModelLoader] = {}
_REGISTRY_LOCK = threading.Lock()


def get_model_bundle(
    model_version: str = APPROVED_DEFAULT_MODEL_VERSION,
    project_root: Optional[pathlib.Path] = None,
    force_reload: bool = False,
) -> LoadedModelBundle:
    """
    Global factory returning the cached LoadedModelBundle for a specified version.
    """
    with _REGISTRY_LOCK:
        key = f"{model_version}_{project_root or 'default'}"
        if key not in _GLOBAL_LOADER_REGISTRY:
            _GLOBAL_LOADER_REGISTRY[key] = Phase4ModelLoader(
                project_root=project_root,
                model_version=model_version,
            )
        loader = _GLOBAL_LOADER_REGISTRY[key]

    return loader.load(force_reload=force_reload)


def get_model_info(
    model_version: str = APPROVED_DEFAULT_MODEL_VERSION,
    project_root: Optional[pathlib.Path] = None,
) -> ModelInfo:
    """
    Returns the structured ModelInfo without reloading if already cached.
    """
    bundle = get_model_bundle(model_version=model_version, project_root=project_root)
    return bundle.info


def clear_model_cache() -> None:
    """Clears all globally cached model instances (useful for testing)."""
    with _REGISTRY_LOCK:
        _GLOBAL_LOADER_REGISTRY.clear()
