"""
phase4_api/benchmarks/parity_checker.py
=======================================
Offline vs. Online Prediction Parity Verification Engine.

Part of Member 3 — API Serving Layer for Adaptive UPI / BAF Fraud Detection.
Phase 8: Performance Testing, Benchmarking & Parity Validation.

Validates that predictions generated directly by the Python inference engine (Offline)
and predictions served through the HTTP REST API (Online) achieve bitwise / numerical
parity within floating-point tolerance (default 1e-8).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from phase4_api.benchmarks.api_client import BenchmarkApiClient, SingleRequestResult
from phase4_api.benchmarks.dataset_loader import ValidatedTransactionRecord
from phase4_api.src.inference_engine import (
    PredictionResponse,
    SingleTransactionInferenceEngine,
)

logger = logging.getLogger("phase4_api.benchmarks.parity_checker")


@dataclass
class ParityRecordComparison:
    """Detailed comparison for a single transaction record."""
    transaction_id: str
    offline_probability: float
    online_probability: float
    probability_diff: float
    probability_match: bool
    offline_decision: str
    online_decision: str
    decision_match: bool
    offline_risk_level: str
    online_risk_level: str
    risk_level_match: bool
    offline_reason_codes: List[str]
    online_reason_codes: List[str]
    reason_codes_match: bool
    reconciled: bool
    error_details: Optional[str] = None


@dataclass
class ParitySummary:
    """Aggregated parity metrics across all evaluated records."""
    total_evaluated: int
    reconciled_count: int
    missing_count: int
    failed_count: int
    reconciliation_rate: float

    # Parity percentages
    probability_parity_percentage: float
    decision_agreement_percentage: float
    risk_level_agreement_percentage: float
    reason_code_agreement_percentage: float

    # Numerical precision metrics
    tolerance_used: float
    max_absolute_probability_difference: float
    mean_absolute_probability_difference: float

    # Mismatch records
    mismatched_transaction_ids: List[str] = field(default_factory=list)


class OfflineOnlineParityChecker:
    """
    Executes offline predictions via SingleTransactionInferenceEngine and compares
    them against online responses from BenchmarkApiClient.
    """

    def __init__(
        self,
        api_client: BenchmarkApiClient,
        offline_engine: Optional[SingleTransactionInferenceEngine] = None,
        tolerance: float = 1e-8,
    ) -> None:
        self.api_client = api_client
        self.offline_engine = offline_engine or SingleTransactionInferenceEngine()
        self.tolerance = tolerance

    def compare_records(
        self,
        records: List[ValidatedTransactionRecord],
    ) -> Tuple[ParitySummary, List[Dict[str, Any]]]:
        """
        Iterates through records in exact sequence, running offline inference and
        online API prediction, then computing agreement metrics.
        """
        if not records:
            raise ValueError("No records supplied for parity comparison.")

        logger.info(
            "Starting Offline vs. Online Parity Evaluation: %d records, tolerance=%e",
            len(records),
            self.tolerance,
        )

        detailed_rows: List[Dict[str, Any]] = []
        prob_matches = 0
        dec_matches = 0
        risk_matches = 0
        reason_matches = 0
        reconciled_count = 0
        failed_count = 0
        missing_count = 0

        prob_diffs: List[float] = []
        mismatched_ids: List[str] = []

        for rec in records:
            tx_id = rec.transaction_id
            
            # 1. Offline Prediction
            try:
                offline_res = self.offline_engine.predict(
                    transaction=rec.features, transaction_id=tx_id
                )
                offline_prob = offline_res.fraud_probability
                offline_dec = offline_res.prediction
                offline_risk = offline_res.risk_level
                offline_reasons = [
                    rc.code if hasattr(rc, "code") else (rc.get("code", str(rc)) if isinstance(rc, dict) else str(rc))
                    for rc in (offline_res.reason_codes or [])
                ]
            except Exception as e:
                failed_count += 1
                logger.error("Offline prediction failed for %s: %s", tx_id, e)
                detailed_rows.append({
                    "transaction_id": tx_id,
                    "reconciled": False,
                    "error_details": f"Offline inference error: {e}",
                })
                mismatched_ids.append(tx_id)
                continue

            # 2. Online Prediction
            online_res: SingleRequestResult = self.api_client.predict_single(tx_id, rec.features)
            if not online_res.is_success or online_res.fraud_probability is None:
                failed_count += 1
                logger.error("Online prediction failed for %s: %s", tx_id, online_res.error_message)
                detailed_rows.append({
                    "transaction_id": tx_id,
                    "reconciled": False,
                    "error_details": f"Online API error: {online_res.error_message}",
                })
                mismatched_ids.append(tx_id)
                continue

            online_prob = online_res.fraud_probability
            online_dec = online_res.prediction or ""
            online_risk = online_res.risk_level or ""
            online_reasons = online_res.reason_codes

            # 3. Precision & Agreement Comparison
            diff = abs(offline_prob - online_prob)
            prob_diffs.append(diff)

            prob_match = diff <= self.tolerance
            dec_match = offline_dec == online_dec
            risk_match = offline_risk == online_risk
            reasons_match = offline_reasons == online_reasons

            if prob_match:
                prob_matches += 1
            if dec_match:
                dec_matches += 1
            if risk_match:
                risk_matches += 1
            if reasons_match:
                reason_matches += 1

            reconciled = prob_match and dec_match and risk_match
            if reconciled:
                reconciled_count += 1
            else:
                mismatched_ids.append(tx_id)

            detailed_rows.append({
                "transaction_id": tx_id,
                "offline_probability": round(offline_prob, 8),
                "online_probability": round(online_prob, 8),
                "probability_diff": round(diff, 10),
                "probability_match": prob_match,
                "offline_decision": offline_dec,
                "online_decision": online_dec,
                "decision_match": dec_match,
                "offline_risk_level": offline_risk,
                "online_risk_level": online_risk,
                "risk_level_match": risk_match,
                "offline_reason_codes": ";".join(offline_reasons),
                "online_reason_codes": ";".join(online_reasons),
                "reason_codes_match": reasons_match,
                "reconciled": reconciled,
                "error_details": None,
            })

        total = len(records)
        max_diff = max(prob_diffs) if prob_diffs else 0.0
        mean_diff = (sum(prob_diffs) / len(prob_diffs)) if prob_diffs else 0.0

        summary = ParitySummary(
            total_evaluated=total,
            reconciled_count=reconciled_count,
            missing_count=missing_count,
            failed_count=failed_count,
            reconciliation_rate=round(reconciled_count / total, 4) if total > 0 else 0.0,
            probability_parity_percentage=round((prob_matches / total) * 100.0, 2) if total > 0 else 0.0,
            decision_agreement_percentage=round((dec_matches / total) * 100.0, 2) if total > 0 else 0.0,
            risk_level_agreement_percentage=round((risk_matches / total) * 100.0, 2) if total > 0 else 0.0,
            reason_code_agreement_percentage=round((reason_matches / total) * 100.0, 2) if total > 0 else 0.0,
            tolerance_used=self.tolerance,
            max_absolute_probability_difference=round(max_diff, 10),
            mean_absolute_probability_difference=round(mean_diff, 10),
            mismatched_transaction_ids=mismatched_ids,
        )

        logger.info(
            "Parity Check Complete: %d/%d reconciled (%.2f%%). Max diff: %e. Decision agreement: %.2f%%",
            reconciled_count,
            total,
            summary.reconciliation_rate * 100.0,
            max_diff,
            summary.decision_agreement_percentage,
        )

        return summary, detailed_rows
