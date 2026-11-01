# pyright: reportMissingImports=false
# type: ignore
"""
experiments/phase4_drift_adaptation/src/member2_drift_adapter.py
==================================================================
Lightweight Integration Adapter between Member 2 (Spark/Flink PSI Drift Detection)
and Phase 4 (BAF Adaptive ML & Drift-Aware Model Management).

Strict Invariants Enforced:
1. Preserves the full 29-feature BAF schema (no features dropped, filtered, or renamed).
2. Directly ingests Member 2's verified drift JSON reports (zero duplicate PSI recalculation).
3. Validates schema completeness, unknown/missing features, and decision consistency.
4. Performs drift-aware feature stratification (Critical >= 0.25, Moderate 0.10..0.25, Stable < 0.10)
   to focus Phase 4 drift attention while keeping all non-drifted features active in the pipeline.
5. Fails closed with descriptive errors on missing files, malformed JSON, or schema mismatch.
6. Zero model retraining, zero parameter changes, zero dependencies on external frameworks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Canonical 29 BAF Feature Schema from Protocol v1.1
CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

NUMERICAL_COLS = [
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
]

CANONICAL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS  # Exactly 29 features
CANONICAL_FEATURE_SET = set(CANONICAL_FEATURE_COLS)
EXCLUDED_COLS = {"fraud_bool", "month", "device_fraud_count"}

# Default repository paths
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPARK_DRIFT_DIR = REPO_ROOT / "spark" / "reports" / "drift"
DEFAULT_FLINK_DRIFT_DIR = REPO_ROOT / "flink" / "reports" / "drift"


@dataclass(frozen=True)
class Member2DriftReport:
    """
    Structured, immutable container for verified Member 2 drift detection results.
    """
    window_label: str
    engine: str
    psi_values: Dict[str, float]
    drifted_features: Dict[str, float]
    stable_features: Dict[str, float]
    drifted_feature_names: List[str]
    drifted_count: int
    stable_count: int
    triggered: bool
    psi_threshold: float
    trigger_count_required: int
    feature_denominator: int
    drift_risk_score: float
    action_recommendation: str
    top_drifted_features: List[Tuple[str, float]]
    retained_features: List[str] = field(default_factory=lambda: list(CANONICAL_FEATURE_COLS))


class DriftAwareFeatureAnalysis:
    """
    Drift-aware feature stratification and attention engine for Phase 4.
    
    Categorizes features into 3 attention tiers without altering model feature dimensions:
    - Critical Drift Tier (PSI >= 0.25): High-priority monitoring for distribution collapse.
    - Moderate Drift Tier (0.10 <= PSI < 0.25): Trending shift monitoring.
    - Stable Tier (PSI < 0.10): Retains nominal baseline representation.
    """

    def __init__(self, report: Member2DriftReport):
        self.report = report

        # Stratify features based on standard severity boundaries
        self.critical_features = {
            f: psi for f, psi in report.psi_values.items() if psi >= 0.25
        }
        self.moderate_features = {
            f: psi for f, psi in report.psi_values.items() if report.psi_threshold <= psi < 0.25
        }
        self.stable_features = {
            f: psi for f, psi in report.psi_values.items() if psi < report.psi_threshold
        }

    def get_attention_weights(self) -> Dict[str, float]:
        """
        Computes analytical attention weights for drift investigation.
        All 29 features receive a non-zero weight (stable = 1.0, moderate = 2.0, critical = 5.0).
        """
        weights: Dict[str, float] = {}
        for feat in self.report.retained_features:
            psi = self.report.psi_values[feat]
            if psi >= 0.25:
                weights[feat] = 5.0
            elif psi >= self.report.psi_threshold:
                weights[feat] = 2.0
            else:
                weights[feat] = 1.0
        return weights

    def get_model_feature_schema(self) -> List[str]:
        """
        Guarantees that all 29 canonical features remain present and strictly ordered.
        """
        return list(CANONICAL_FEATURE_COLS)

    def should_trigger_retraining(self) -> bool:
        """
        Decision rule evaluated by Member 2: triggered if >= 6 features have PSI >= 0.10.
        """
        return self.report.triggered

    def summary(self) -> Dict[str, Any]:
        """Returns structured diagnostic telemetry for Phase 4 logging."""
        return {
            "window_label": self.report.window_label,
            "engine": self.report.engine,
            "total_features": len(self.report.retained_features),
            "drifted_count": self.report.drifted_count,
            "stable_count": self.report.stable_count,
            "critical_features_count": len(self.critical_features),
            "moderate_features_count": len(self.moderate_features),
            "stable_features_count": len(self.stable_features),
            "triggered": self.report.triggered,
            "action_recommendation": self.report.action_recommendation,
            "drift_risk_score": self.report.drift_risk_score,
            "top_3_shifted": self.report.top_drifted_features[:3],
        }


def resolve_report_path(
    window: str | int,
    engine: Literal["spark", "flink"] = "spark",
    custom_dir: Optional[Path | str] = None,
) -> Path:
    """Resolves Member 2 drift report file path."""
    w_label = f"month{window}" if isinstance(window, int) or not str(window).startswith("month") else str(window)
    base_dir = Path(custom_dir) if custom_dir else (DEFAULT_SPARK_DRIFT_DIR if engine == "spark" else DEFAULT_FLINK_DRIFT_DIR)
    return base_dir / f"{engine}_psi_{w_label}.json"


def load_member2_drift_report(
    window_or_path: Optional[str | int | Path] = None,
    engine: Literal["spark", "flink"] = "spark",
    custom_dir: Optional[Path | str] = None,
    window: Optional[str | int] = None,
) -> Member2DriftReport:
    """
    Loads, validates, and standardizes a Member 2 BAF drift report for Phase 4.

    Raises:
        FileNotFoundError: If the report file does not exist (fail-closed, never assumes no-drift).
        ValueError: If JSON is malformed, missing required keys, or feature set does not match 29 canonical columns.
    """
    target = window if window is not None else window_or_path
    if target is None:
        raise ValueError("Must provide either 'window_or_path' or 'window' argument.")

    if isinstance(target, Path) or ("/" in str(target) or "\\" in str(target)):
        target_path = Path(target)
    else:
        target_path = resolve_report_path(window=target, engine=engine, custom_dir=custom_dir)

    if not target_path.exists():
        raise FileNotFoundError(
            f"Member 2 drift report not found at: {target_path}. "
            f"Phase 4 cannot assume 'no drift' when telemetry is missing."
        )

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(f"Malformed JSON in Member 2 drift report '{target_path}': {err}") from err

    # 1. Validate required root fields
    required_keys = {"psi_values", "triggered_features", "triggered_count", "triggered", "psi_threshold"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Member 2 drift report '{target_path}' is missing required fields: {sorted(missing_keys)}")

    raw_psi_values = data["psi_values"]
    if not isinstance(raw_psi_values, dict):
        raise ValueError(f"Field 'psi_values' must be a dictionary, got {type(raw_psi_values).__name__}")

    # 2. Strict 29-feature integrity check
    report_features = set(raw_psi_values.keys())
    if report_features != CANONICAL_FEATURE_SET:
        missing_features = CANONICAL_FEATURE_SET - report_features
        extra_features = report_features - CANONICAL_FEATURE_SET
        err_msg = f"Feature schema mismatch in Member 2 drift report '{target_path}'."
        if missing_features:
            err_msg += f" Missing {len(missing_features)} features: {sorted(missing_features)}."
        if extra_features:
            err_msg += f" Unknown {len(extra_features)} features: {sorted(extra_features)}."
        raise ValueError(err_msg)

    # 3. Ensure target labels are strictly excluded
    for excluded in EXCLUDED_COLS:
        if excluded in report_features:
            raise ValueError(f"Label leakage risk: excluded column '{excluded}' found in drift report features!")

    psi_threshold = float(data["psi_threshold"])
    trigger_count_required = int(data.get("trigger_count_required", 6))

    # 4. Partition into drifted vs stable features
    drifted_features = {f: float(raw_psi_values[f]) for f in CANONICAL_FEATURE_COLS if float(raw_psi_values[f]) >= psi_threshold}
    stable_features = {f: float(raw_psi_values[f]) for f in CANONICAL_FEATURE_COLS if float(raw_psi_values[f]) < psi_threshold}

    drifted_feature_names = sorted(drifted_features.keys(), key=lambda f: -drifted_features[f])
    drifted_count = len(drifted_features)

    # Verify reconciliation with report metadata
    if drifted_count != int(data["triggered_count"]):
        raise ValueError(
            f"Internal inconsistency in drift report: recalculated drifted count ({drifted_count}) "
            f"does not match reported 'triggered_count' ({data['triggered_count']})"
        )

    expected_triggered = drifted_count >= trigger_count_required
    if bool(data["triggered"]) != expected_triggered:
        raise ValueError(
            f"Decision inconsistency: report 'triggered' is {data['triggered']} "
            f"but condition ({drifted_count} >= {trigger_count_required}) is {expected_triggered}"
        )

    top_drifted = sorted(raw_psi_values.items(), key=lambda x: -x[1])

    # Infer or extract window label
    window_label = data.get("window_label", target_path.stem.replace("spark_psi_", "").replace("flink_psi_", ""))
    detected_engine = "spark" if "spark" in target_path.name.lower() else ("flink" if "flink" in target_path.name.lower() else engine)

    return Member2DriftReport(
        window_label=window_label,
        engine=detected_engine,
        psi_values={f: float(raw_psi_values[f]) for f in CANONICAL_FEATURE_COLS},
        drifted_features=drifted_features,
        stable_features=stable_features,
        drifted_feature_names=drifted_feature_names,
        drifted_count=drifted_count,
        stable_count=len(stable_features),
        triggered=expected_triggered,
        psi_threshold=psi_threshold,
        trigger_count_required=trigger_count_required,
        feature_denominator=len(CANONICAL_FEATURE_COLS),
        drift_risk_score=float(data.get("drift_risk_score", 0.0)),
        action_recommendation=str(data.get("action_recommendation", "RETRAIN_MODEL" if expected_triggered else "MAINTAIN_STATIC")),
        top_drifted_features=top_drifted,
        retained_features=list(CANONICAL_FEATURE_COLS),
    )


def validate_and_handoff_drift(
    window: str | int,
    engine: Literal["spark", "flink"] = "spark",
    custom_dir: Optional[Path | str] = None,
) -> Tuple[Member2DriftReport, DriftAwareFeatureAnalysis]:
    """
    High-level integration entry point for Phase 4.
    
    Reads Member 2's report, verifies data integrity, ensures all 29 features
    are retained, and creates a drift-aware feature analysis object.
    """
    report = load_member2_drift_report(window=window, engine=engine, custom_dir=custom_dir)
    analysis = DriftAwareFeatureAnalysis(report)
    return report, analysis
