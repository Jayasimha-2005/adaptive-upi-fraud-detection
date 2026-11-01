"""
phase4_api/src/dependencies.py
==============================
FastAPI Dependency Injection Providers.

Part of Member 3 — API Serving for Adaptive UPI / BAF Fraud Detection.

Provides decoupled dependencies for:
  1. SingleTransactionInferenceEngine
  2. BatchTransactionInferenceEngine
  3. LoadedModelBundle

Allows test suites to override dependencies cleanly without touching disk artifacts.
"""
from __future__ import annotations

from typing import Optional

from phase4_api.src.batch_inference_engine import (
    BatchTransactionInferenceEngine,
    get_batch_inference_engine,
)
from phase4_api.src.inference_engine import (
    SingleTransactionInferenceEngine,
    get_inference_engine,
)
from phase4_api.src.model_loader import (
    APPROVED_DEFAULT_MODEL_VERSION,
    LoadedModelBundle,
    get_model_bundle,
)


def get_single_engine() -> SingleTransactionInferenceEngine:
    """Dependency provider for SingleTransactionInferenceEngine."""
    return get_inference_engine(model_version=APPROVED_DEFAULT_MODEL_VERSION)


def get_batch_engine() -> BatchTransactionInferenceEngine:
    """Dependency provider for BatchTransactionInferenceEngine."""
    return get_batch_inference_engine(model_version=APPROVED_DEFAULT_MODEL_VERSION)


def get_active_bundle() -> LoadedModelBundle:
    """Dependency provider for LoadedModelBundle."""
    return get_model_bundle(model_version=APPROVED_DEFAULT_MODEL_VERSION)
