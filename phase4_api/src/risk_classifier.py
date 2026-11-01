"""
phase4_api/src/risk_classifier.py
=================================
Standardized Risk Classification, Reason Codes, and Explanation Metadata.

Part of Member 3 — API Serving for Adaptive UPI / BAF Fraud Detection.

This module provides:
  1. RiskLevel: Enumeration of standardized risk tiers (LOW, MEDIUM, HIGH, CRITICAL).
  2. ReasonCode: Structured, auditable reason-code container.
  3. RiskPolicy: Explicitly configurable, documented risk-banding policy.
  4. RiskClassifier: Decoupled evaluator linking fraud probabilities, frozen
     thresholds, risk tiers, and auditable reason codes.

CRITICAL SEPARATION OF CONCERNS:
  - Fraud Probability: LightGBM model score in [0.0, 1.0].
  - Fraud Decision: Authoritative frozen-threshold binary outcome ("FRAUD" | "LEGIT").
  - Risk Level: Configurable descriptive categorization ("LOW" | "MEDIUM" | "HIGH" | "CRITICAL").
    A transaction may have HIGH risk while remaining below the frozen decision threshold
    (resulting in "LEGIT" decision with "HIGH" risk level).
  - Reason Codes: Stable, auditable justification tokens. Does NOT invent feature-level
    attributions (e.g., SHAP/LIME) when not present.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RiskClassificationError(Exception):
    """Base exception for risk classification and reasoning errors."""


class InvalidProbabilityError(RiskClassificationError):
    """Raised when a probability value is non-numeric, NaN, infinite, or outside [0.0, 1.0]."""


class InvalidRiskPolicyError(RiskClassificationError):
    """Raised when a RiskPolicy configuration has invalid, overlapping, or non-monotonic cutoffs."""


class ReasonCodeGenerationError(RiskClassificationError):
    """Raised when structured reason code generation fails."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class RiskLevel(str, Enum):
    """
    Standardized descriptive risk levels for BAF transactions.
    Inherits from str for seamless JSON serialization and dictionary lookup.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Structured Reason Code
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReasonCode:
    """
    Standardized, auditable explanation token.
    Never leaks raw transaction attributes or personal identifying information (PII).
    """

    code: str
    description: str
    category: str  # e.g., "MODEL_DECISION", "RISK_LEVEL", "THRESHOLD", "EXPLANATION"

    def to_dict(self) -> Dict[str, str]:
        """Converts to a JSON-serializable dictionary."""
        return {
            "code": self.code,
            "description": self.description,
            "category": self.category,
        }


# Standard Reason Code Definitions
REASON_DECISION_FRAUD = ReasonCode(
    code="MODEL_DECISION_FRAUD",
    description="Transaction probability equals or exceeds the frozen fraud decision threshold.",
    category="MODEL_DECISION",
)

REASON_DECISION_LEGIT = ReasonCode(
    code="MODEL_DECISION_LEGIT",
    description="Transaction probability is strictly below the frozen fraud decision threshold.",
    category="MODEL_DECISION",
)

REASON_RISK_LOW = ReasonCode(
    code="RISK_LEVEL_LOW",
    description="Transaction probability falls within the low descriptive risk band.",
    category="RISK_LEVEL",
)

REASON_RISK_MEDIUM = ReasonCode(
    code="RISK_LEVEL_MEDIUM",
    description="Transaction probability falls within the medium descriptive risk band.",
    category="RISK_LEVEL",
)

REASON_RISK_HIGH = ReasonCode(
    code="RISK_LEVEL_HIGH",
    description="Transaction probability falls within the high descriptive risk band. "
    "Note: Decision may remain LEGIT if below the frozen decision threshold.",
    category="RISK_LEVEL",
)

REASON_RISK_CRITICAL = ReasonCode(
    code="RISK_LEVEL_CRITICAL",
    description="Transaction probability falls within the critical descriptive risk band.",
    category="RISK_LEVEL",
)

REASON_THRESHOLD_MET = ReasonCode(
    code="THRESHOLD_EXCEEDED",
    description="Calculated fraud probability meets or exceeds the frozen decision threshold.",
    category="THRESHOLD",
)

REASON_THRESHOLD_NOT_MET = ReasonCode(
    code="THRESHOLD_NOT_MET",
    description="Calculated fraud probability does not meet the frozen decision threshold.",
    category="THRESHOLD",
)

REASON_EXPLANATION_UNAVAILABLE = ReasonCode(
    code="FEATURE_EXPLANATION_UNAVAILABLE",
    description="Feature-level attribution (e.g., SHAP/LIME) is not enabled in Phase 6 serving.",
    category="EXPLANATION",
)


# ---------------------------------------------------------------------------
# Configurable Risk Policy
# ---------------------------------------------------------------------------
DEFAULT_RISK_POLICY_VERSION: str = "v1.0-provisional"
DEFAULT_LOW_THRESHOLD: float = 0.20
DEFAULT_MEDIUM_THRESHOLD: float = 0.50


class RiskPolicy:
    """
    Configurable risk-banding policy for categorizing fraud probabilities.

    POLICY SPECIFICATION & JUSTIFICATION:
      - This policy is a business risk-management configuration, NOT a model parameter.
      - Default thresholds:
          LOW:      [0.0, low_threshold)              (Default: [0.0, 0.20))
          MEDIUM:   [low_threshold, medium_threshold) (Default: [0.20, 0.50))
          HIGH:     [medium_threshold, high_threshold)(Default: [0.50, frozen_threshold))
          CRITICAL: [high_threshold, 1.0]             (Default: [frozen_threshold, 1.0])
      - If high_threshold is None, it dynamically anchors to the model's frozen threshold
        (e.g., 0.8964321608040201 for LightGBM v3).
      - Boundary rule: Half-open intervals [lower, upper) with upper endpoint belonging
        to the higher category, except 1.0 which is inclusive in CRITICAL.
    """

    def __init__(
        self,
        low_threshold: float = DEFAULT_LOW_THRESHOLD,
        medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
        high_threshold: Optional[float] = None,
        version: str = DEFAULT_RISK_POLICY_VERSION,
    ) -> None:
        self.version = str(version)
        self.low_threshold = self._validate_cutoff(low_threshold, "low_threshold")
        self.medium_threshold = self._validate_cutoff(medium_threshold, "medium_threshold")
        self.high_threshold = (
            self._validate_cutoff(high_threshold, "high_threshold")
            if high_threshold is not None
            else None
        )

        # Validate monotonic ordering if high_threshold is explicitly provided
        if not (0.0 <= self.low_threshold < self.medium_threshold):
            raise InvalidRiskPolicyError(
                f"RiskPolicy thresholds must satisfy 0.0 <= low_threshold < medium_threshold: "
                f"got low={self.low_threshold}, medium={self.medium_threshold}"
            )
        if self.high_threshold is not None:
            if not (self.medium_threshold < self.high_threshold <= 1.0):
                raise InvalidRiskPolicyError(
                    f"RiskPolicy thresholds must satisfy medium_threshold < high_threshold <= 1.0: "
                    f"got medium={self.medium_threshold}, high={self.high_threshold}"
                )

    @staticmethod
    def _validate_cutoff(val: Any, name: str) -> float:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise InvalidRiskPolicyError(
                f"RiskPolicy {name} must be a numeric float, got {type(val).__name__}: {val}"
            )
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            raise InvalidRiskPolicyError(f"RiskPolicy {name} cannot be NaN or Infinite: {val}")
        if not (0.0 <= f_val <= 1.0):
            raise InvalidRiskPolicyError(
                f"RiskPolicy {name} must be in [0.0, 1.0], got {f_val}"
            )
        return f_val

    def resolve_high_threshold(self, frozen_threshold: float) -> float:
        """
        Resolves the effective high threshold, using the configured high_threshold
        if set, or dynamically anchoring to the model's authoritative frozen threshold.
        """
        if self.high_threshold is not None:
            return self.high_threshold
        # Anchor to frozen_threshold
        f_thresh = self._validate_cutoff(frozen_threshold, "frozen_threshold")
        if f_thresh <= self.medium_threshold:
            raise InvalidRiskPolicyError(
                f"Model frozen threshold ({f_thresh}) must be strictly greater than "
                f"RiskPolicy medium_threshold ({self.medium_threshold})"
            )
        return f_thresh

    def classify(self, probability: float, frozen_threshold: float) -> RiskLevel:
        """
        Classifies a fraud probability into a standardized RiskLevel.

        Parameters
        ----------
        probability : float
            LightGBM fraud probability in [0.0, 1.0].
        frozen_threshold : float
            Authoritative frozen model decision threshold.

        Returns
        -------
        RiskLevel
            One of LOW, MEDIUM, HIGH, CRITICAL.
        """
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise InvalidProbabilityError(
                f"Probability must be a numeric float, got {type(probability).__name__}: {probability}"
            )
        prob = float(probability)
        if math.isnan(prob) or math.isinf(prob):
            raise InvalidProbabilityError(f"Probability cannot be NaN or Infinite: {prob}")
        if not (0.0 <= prob <= 1.0):
            raise InvalidProbabilityError(f"Probability must be in [0.0, 1.0], got {prob}")

        effective_high = self.resolve_high_threshold(frozen_threshold)

        if prob < self.low_threshold:
            return RiskLevel.LOW
        elif prob < self.medium_threshold:
            return RiskLevel.MEDIUM
        elif prob < effective_high:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def generate_reason_codes(
        self,
        probability: float,
        frozen_threshold: float,
        decision: str,
        risk_level: RiskLevel,
    ) -> List[ReasonCode]:
        """
        Generates deterministic, auditable reason codes for a prediction outcome.
        Never fabricates feature attributions.
        """
        codes: List[ReasonCode] = []

        # 1. Model Decision Code
        if decision == "FRAUD":
            codes.append(REASON_DECISION_FRAUD)
            codes.append(REASON_THRESHOLD_MET)
        else:
            codes.append(REASON_DECISION_LEGIT)
            codes.append(REASON_THRESHOLD_NOT_MET)

        # 2. Risk Level Code
        if risk_level == RiskLevel.LOW:
            codes.append(REASON_RISK_LOW)
        elif risk_level == RiskLevel.MEDIUM:
            codes.append(REASON_RISK_MEDIUM)
        elif risk_level == RiskLevel.HIGH:
            codes.append(REASON_RISK_HIGH)
        elif risk_level == RiskLevel.CRITICAL:
            codes.append(REASON_RISK_CRITICAL)

        # 3. Neutral Explanation Availability Code
        codes.append(REASON_EXPLANATION_UNAVAILABLE)

        return codes

    def generate_explanation_metadata(
        self,
        model_version: str,
        model_status: str,
        feature_count: int,
        frozen_threshold: float,
        inference_mode: str,
    ) -> Dict[str, Any]:
        """
        Generates structured metadata explaining the operational context
        and policy rules used to derive the result.
        """
        effective_high = self.resolve_high_threshold(frozen_threshold)
        return {
            "model_version": str(model_version),
            "model_status": str(model_status),
            "feature_count": int(feature_count),
            "threshold": float(frozen_threshold),
            "risk_policy_version": self.version,
            "risk_cutoffs": {
                "low": [0.0, self.low_threshold],
                "medium": [self.low_threshold, self.medium_threshold],
                "high": [self.medium_threshold, effective_high],
                "critical": [effective_high, 1.0],
            },
            "inference_mode": str(inference_mode),
            "explanation_method": "threshold_and_risk_policy",
            "feature_explanations_available": False,
        }


# ---------------------------------------------------------------------------
# Risk Evaluation Container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskEvaluationResult:
    """Encapsulates the complete evaluation produced by RiskClassifier."""

    risk_level: RiskLevel
    reason_codes: List[ReasonCode]
    explanation_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "reason_codes": [rc.to_dict() for rc in self.reason_codes],
            "explanation_metadata": self.explanation_metadata,
        }


# ---------------------------------------------------------------------------
# Risk Classifier Engine
# ---------------------------------------------------------------------------
class RiskClassifier:
    """
    Decoupled risk classification engine for Phase 6.
    Applies the configured RiskPolicy to evaluate prediction outputs.
    """

    def __init__(self, policy: Optional[RiskPolicy] = None) -> None:
        self.policy = policy or RiskPolicy()

    def classify(self, probability: float, frozen_threshold: float) -> RiskLevel:
        """Classifies a probability into a RiskLevel."""
        return self.policy.classify(probability, frozen_threshold)

    def evaluate(
        self,
        probability: float,
        frozen_threshold: float,
        decision: str,
        model_version: str,
        model_status: str,
        feature_count: int,
        inference_mode: str = "single",
    ) -> RiskEvaluationResult:
        """
        Performs full risk evaluation, deriving risk level, structured reason codes,
        and explanation metadata.
        """
        risk_lvl = self.classify(probability, frozen_threshold)
        reason_codes = self.policy.generate_reason_codes(
            probability=probability,
            frozen_threshold=frozen_threshold,
            decision=decision,
            risk_level=risk_lvl,
        )
        metadata = self.policy.generate_explanation_metadata(
            model_version=model_version,
            model_status=model_status,
            feature_count=feature_count,
            frozen_threshold=frozen_threshold,
            inference_mode=inference_mode,
        )
        return RiskEvaluationResult(
            risk_level=risk_lvl,
            reason_codes=reason_codes,
            explanation_metadata=metadata,
        )


# Default Singleton Instance
_DEFAULT_CLASSIFIER: Optional[RiskClassifier] = None


def get_risk_classifier(policy: Optional[RiskPolicy] = None) -> RiskClassifier:
    """Returns the singleton or newly instantiated RiskClassifier."""
    global _DEFAULT_CLASSIFIER
    if policy is not None:
        return RiskClassifier(policy=policy)
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = RiskClassifier()
    return _DEFAULT_CLASSIFIER
