"""
streaming/stream_serving_bridge.py
Phase 13: Minimal Stream-to-Serving Integration Bridge.

Bridges Member 1 Kafka event ingestion and Member 2 Flink stream features
to Member 3 FastAPI serving and the Online Feature Hydration Gate for frozen E1 LightGBM inference.

Strict Invariants Enforced:
1. Deterministic correlation by entity key and temporal window (no approximate cross-joins).
2. Zero future-event leakage: velocity records must satisfy feat.timestamp <= event.timestamp.
3. Target isolation: isFraud and is_fraud are strictly stripped prior to inference.
4. Hard hydration gate: incomplete events return scoreable=False; model is unreachable.
5. Canonical E1 preservation: loads canonical model.txt and preprocessing.joblib at threshold 0.616521.
"""
from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root and serving directory to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SERVING_DIR = REPO_ROOT / "serving"
for p in [str(REPO_ROOT), str(SERVING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from serving.hydration.adapter import OnlineFeatureHydrationAdapter
from serving.hydration.entity_store import EntityProfileStore
from serving.hydration.schemas import (
    HydrationResult,
    StreamingTransactionPayload,
    VelocityMetrics,
)
from serving.inference.offline_inference import (
    E1_DECISION_THRESHOLD,
    OfflineInferenceEngine,
)

logger = logging.getLogger("stream_serving_bridge")


class CorrelationBuffer:
    """
    Watermarked in-memory buffer that correlates Flink streaming window features
    with transaction events strictly adhering to temporal causality.
    """

    def __init__(self, retention_seconds: float = 600.0):
        self.retention_seconds = retention_seconds
        # State: card_id -> deque of (timestamp_seconds, metrics_dict)
        self._buffer: Dict[str, deque] = defaultdict(deque)

    def add_feature_event(self, feature_record: Dict[str, Any]) -> None:
        """
        Ingest a streaming feature record emitted by Flink / Spark into the correlation buffer.
        """
        # Harmonize user_id / card_id
        user_key = str(feature_record.get("user_id") or feature_record.get("card_id") or "UNKNOWN")
        if not user_key.startswith("CARD-") and user_key.isdigit():
            user_key = f"CARD-{user_key}"

        # Harmonize timestamp (convert epoch ms to seconds if necessary)
        raw_ts = feature_record.get("timestamp") or feature_record.get("event_time", 0.0)
        try:
            ts_sec = float(raw_ts)
            if ts_sec > 1e8:  # Milliseconds detected (Unix epoch ms or scaled ms)
                ts_sec = ts_sec / 1000.0
        except (ValueError, TypeError):
            ts_sec = time.time()

        # Prune old records for this card
        q = self._buffer[user_key]
        cutoff = ts_sec - self.retention_seconds
        while q and q[0][0] < cutoff:
            q.popleft()

        q.append((ts_sec, feature_record))

    def correlate(
        self,
        card_id: str,
        event_timestamp: float,
        window_tolerance_sec: float = 300.0,
    ) -> Optional[VelocityMetrics]:
        """
        Find the most recent causal velocity record for `card_id`.
        
        INVARIANT: feat_timestamp <= event_timestamp.
        Future velocity records are NEVER correlated with a past transaction.
        """
        key = str(card_id)
        if not key.startswith("CARD-") and key.isdigit():
            key = f"CARD-{key}"

        q = self._buffer.get(key)
        if not q:
            return None

        # Search backward from the latest record for the most recent causal entry
        best_match: Optional[Dict[str, Any]] = None
        for ts_sec, record in reversed(q):
            if ts_sec <= event_timestamp:
                if (event_timestamp - ts_sec) <= window_tolerance_sec:
                    best_match = record
                break  # Found latest record with ts <= event_timestamp

        if not best_match:
            return None

        try:
            return VelocityMetrics(
                count_5m=best_match.get("transaction_count_5m"),
                amount_5m=best_match.get("total_amount_5m"),
                avg_amount_5m=best_match.get("average_amount_5m"),
                count_10m=best_match.get("transaction_count_10m"),
                amount_10m=best_match.get("total_amount_10m"),
                velocity_ratio=best_match.get("transaction_velocity_ratio"),
            )
        except Exception:
            # Corrupted velocity record: do not let corrupted data crash serving
            return None


class StreamServingBridge:
    """
    Cross-Member Integration Bridge connecting Kafka/Flink streaming to E1 serving.
    """

    def __init__(
        self,
        adapter: Optional[OnlineFeatureHydrationAdapter] = None,
        engine: Optional[OfflineInferenceEngine] = None,
        correlation_buffer: Optional[CorrelationBuffer] = None,
        correlation_window_sec: float = 300.0,
    ):
        store = EntityProfileStore()
        store.seed_mock_profiles()
        self.adapter = adapter or OnlineFeatureHydrationAdapter(entity_store=store)
        self.engine = engine or OfflineInferenceEngine(threshold=E1_DECISION_THRESHOLD)
        self.correlation_buffer = correlation_buffer or CorrelationBuffer()
        self.correlation_window_sec = correlation_window_sec

    def normalize_stream_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Harmonize Member 1 / Kafka fields into normalized transaction format.
        Strictly excludes any target labels (isFraud / is_fraud).
        """
        event_dict = raw_event.copy()

        # ── Target Leakage Elimination ────────────────────────────────────────
        event_dict.pop("isFraud", None)
        event_dict.pop("is_fraud", None)
        event_dict.pop("fraud_bool", None)

        # Harmonize Transaction ID
        tx_id = (
            event_dict.get("TransactionID")
            or event_dict.get("transaction_id")
            or event_dict.get("id")
        )

        # Harmonize Card ID / Key
        card_val = event_dict.get("card_id") or event_dict.get("card1") or event_dict.get("user_id")
        if card_val is not None:
            card_str = str(card_val)
            if not card_str.startswith("CARD-"):
                card_id = f"CARD-{card_str}"
            else:
                card_id = card_str
        else:
            card_id = "CARD-UNKNOWN"

        # Harmonize Amount
        amt = event_dict.get("TransactionAmt") or event_dict.get("amount")

        # Harmonize Event Time (TransactionDT)
        dt = event_dict.get("TransactionDT") or event_dict.get("epoch_timestamp") or event_dict.get("timestamp")
        if dt is not None:
            try:
                dt_val = float(dt)
                if dt_val > 1e11:  # Epoch milliseconds
                    dt_val = dt_val / 1000.0
            except (ValueError, TypeError):
                dt_val = 0.0
        else:
            dt_val = None

        # Harmonize Product Code
        prod_cd = event_dict.get("ProductCD") or event_dict.get("merchant_id", "W")
        if isinstance(prod_cd, str) and prod_cd.startswith("M-"):
            prod_cd = prod_cd.replace("M-", "")

        return {
            "TransactionID": tx_id,
            "card_id": card_id,
            "TransactionAmt": float(amt) if amt is not None else None,
            "TransactionDT": dt_val,
            "ProductCD": str(prod_cd) if prod_cd else "W",
            "device_type": str(event_dict.get("device_type") or event_dict.get("DeviceType")) if (event_dict.get("device_type") or event_dict.get("DeviceType")) else None,
            "country": str(event_dict.get("country") or event_dict.get("addr2")) if (event_dict.get("country") or event_dict.get("addr2")) else None,
        }

    def assemble_payload(
        self,
        raw_event: Dict[str, Any],
        velocity_metrics: Optional[VelocityMetrics] = None,
    ) -> StreamingTransactionPayload:
        """
        Assemble normalized event and correlated velocity metrics into StreamingTransactionPayload.
        """
        norm = self.normalize_stream_event(raw_event)
        payload_dict = {
            "TransactionID": norm["TransactionID"],
            "card_id": norm["card_id"],
            "TransactionAmt": norm["TransactionAmt"],
            "TransactionDT": norm["TransactionDT"],
            "ProductCD": norm["ProductCD"],
            "device_type": norm.get("device_type"),
            "country": norm.get("country"),
            "velocity": velocity_metrics,
        }
        return StreamingTransactionPayload.model_validate(payload_dict)

    def process_transaction(
        self,
        raw_event: Dict[str, Any],
        feature_record: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest transaction event, correlate with streaming velocity, and execute inference.
        Enforces Hard Hydration Gate: incomplete events return scoreable=False without model scoring.
        """
        # Register streaming feature record if provided
        if feature_record:
            self.correlation_buffer.add_feature_event(feature_record)

        # Normalize raw transaction
        norm = self.normalize_stream_event(raw_event)
        card_id = norm["card_id"]
        event_dt = norm["TransactionDT"]

        # Validate essential field existence before payload validation
        if norm["TransactionAmt"] is None or norm["TransactionDT"] is None or norm["TransactionID"] is None:
            return {
                "scoreable": False,
                "scored": False,
                "transaction_id": norm.get("TransactionID"),
                "card_id": card_id,
                "error": "REJECTED_BY_HYDRATION_GATE",
                "rejection_reason": "Missing required event fields (TransactionID, TransactionAmt, or TransactionDT)",
                "point_in_time_valid": False,
            }

        # Correlate causal velocity metrics
        velocity_metrics = None
        if event_dt is not None:
            velocity_metrics = self.correlation_buffer.correlate(
                card_id=card_id,
                event_timestamp=event_dt,
                window_tolerance_sec=self.correlation_window_sec,
            )

        payload = self.assemble_payload(raw_event, velocity_metrics)

        # Execute scoring via Hard Hydration Gate
        score_res = self.adapter.score_or_reject(payload, self.engine)
        score_res["correlated_velocity"] = velocity_metrics is not None
        score_res["decision_threshold"] = E1_DECISION_THRESHOLD
        return score_res
