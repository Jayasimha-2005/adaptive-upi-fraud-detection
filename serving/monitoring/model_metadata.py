"""
monitoring/model_metadata.py
Phase 8 — Single source of truth for E1 model identity and configuration.

This module is the canonical reference for all model version information.
api/main.py, health endpoints, and inference loggers all read from here —
no values are duplicated or hardcoded elsewhere.

IMPORTANT: Do NOT modify E1_DECISION_THRESHOLD here directly.
It is imported from inference.offline_inference to avoid duplication.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root and serving directory to sys.path
SERVING_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVING_DIR.parent
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from inference.offline_inference import E1_DECISION_THRESHOLD

# ── E1 Model Metadata (Single Source of Truth) ───────────────────────────────
#
# model_version:  Short version tag used in logs and monitoring records.
# model_name:     Full name used in /health responses and experiment records.
# All artifact paths point to the canonical research artifacts.
#
MODEL_METADATA: dict = {
    "model_version":        "E1",
    "model_name":           "E1_LightGBM",
    "feature_count":        406,
    "decision_threshold":   E1_DECISION_THRESHOLD,   # 0.616521 — imported, not duplicated
    "artifact_model":       "experiments/E1_lightgbm/model.txt",
    "artifact_preprocessor":"experiments/E1_lightgbm/preprocessing.joblib",
    "artifact_features":    "experiments/E1_lightgbm/feature_names.json",
    "source_repository":    "adaptive-upi-fraud-detection",
    "serving_repository":   "fraud-model-serving",
}

