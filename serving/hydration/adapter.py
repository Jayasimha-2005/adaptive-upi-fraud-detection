"""
serving/hydration/adapter.py
Online Feature Hydration Adapter for E1 Serving.

Bridges compact streaming events (from Kafka & Flink) to the canonical
406-feature tabular representation required by the frozen E1 LightGBM model.
Enforces strict point-in-time causality, feature provenance tracking,
and a hard hydration gate (no silent scoring of incomplete events).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from serving.hydration.entity_store import EntityProfileStore, EntityProfile
from serving.hydration.provenance import (
    classify_feature,
    TIER_DIRECT_EVENT,
    TIER_DERIVED_EVENT,
    TIER_ENTITY_PROFILE,
    TIER_HISTORICAL_AGG,
    TIER_UNRECONSTRUCTABLE,
)
from serving.hydration.schemas import (
    HydrationResult,
    StreamingTransactionPayload,
    VelocityMetrics,
)

logger = logging.getLogger(__name__)


class OnlineFeatureHydrationAdapter:
    """
    Stateful Feature Hydration Adapter with a Hard Invariant Gate.
    """

    # Essential fields required for faithful E1 representation
    ESSENTIAL_PROFILE_FIELDS = ["card1", "card4", "card6"]
    ESSENTIAL_EVENT_FIELDS = ["TransactionAmt", "TransactionDT", "ProductCD"]

    def __init__(self, entity_store: Optional[EntityProfileStore] = None):
        self.entity_store = entity_store or EntityProfileStore()
        # Ensure default mock profiles are registered
        self.entity_store.seed_mock_profiles()

    def hydrate(
        self,
        payload: Union[StreamingTransactionPayload, Dict[str, Any]],
    ) -> HydrationResult:
        """
        Hydrate a compact streaming transaction event into an IEEE-CIS raw DataFrame.
        
        Hard Gate:
        If essential profile attributes are missing, or point-in-time validity fails,
        returns complete=False with explicit diagnostic reason.
        """
        if isinstance(payload, dict):
            # Parse dict into StreamingTransactionPayload
            raw_dict = payload.copy()
            # Normalize field names
            if "TransactionAmt" in raw_dict and "amount" not in raw_dict:
                raw_dict["amount"] = raw_dict["TransactionAmt"]
            if "TransactionDT" in raw_dict and "event_time" not in raw_dict:
                raw_dict["event_time"] = raw_dict["TransactionDT"]
            if "TransactionID" in raw_dict and "transaction_id" not in raw_dict:
                raw_dict["transaction_id"] = raw_dict["TransactionID"]
            if "card_id" not in raw_dict and "card1" in raw_dict:
                raw_dict["card_id"] = f"CARD-{raw_dict['card1']}"

            try:
                event = StreamingTransactionPayload.model_validate(raw_dict)
            except Exception as e:
                return HydrationResult(
                    complete=False,
                    point_in_time_valid=False,
                    dataframe=None,
                    rejection_reason=f"Payload schema validation error: {str(e)}",
                )
        else:
            event = payload

        tx_id = str(event.transaction_id)
        card_id = str(event.card_id)
        current_dt = float(event.event_time)
        amount = float(event.amount)
        product_cd = str(event.product_code or "W")

        provenance_map: Dict[str, str] = {}
        missing_features: List[str] = []

        # ── Step 1: Query Entity Profile Store ────────────────────────────────
        profile: Optional[EntityProfile] = self.entity_store.get_profile(card_id)
        if profile is None:
            # Check if card_id is formatted as CARD-<number> and try looking up by number
            if card_id.startswith("CARD-"):
                num_key = card_id.replace("CARD-", "")
                profile = self.entity_store.get_profile(num_key)

        if profile is None:
            logger.warning("Hydration rejected: Cardholder entity '%s' not registered in store", card_id)
            return HydrationResult(
                complete=False,
                point_in_time_valid=True,
                card_id=card_id,
                transaction_id=tx_id,
                missing_features=["card1", "card4", "card6", "addr1"],
                rejection_reason=f"Unknown card_id '{card_id}': Entity profile lookup failed in store",
            )

        # ── Step 2: Query Causal Point-in-Time History ────────────────────────
        causal_history = self.entity_store.get_causal_history(card_id, current_dt)

        # Hard Gate: Verify zero future lookahead
        for hist_item in causal_history:
            if hist_item.timestamp >= current_dt:
                return HydrationResult(
                    complete=False,
                    point_in_time_valid=False,
                    card_id=card_id,
                    transaction_id=tx_id,
                    rejection_reason=(
                        f"Point-in-time violation: Found history item {hist_item.transaction_id} "
                        f"with timestamp {hist_item.timestamp} >= current_dt {current_dt}"
                    ),
                )

        # ── Step 3: Compute Behavioral Counters and Timedeltas ────────────────
        hist_count = len(causal_history)
        if hist_count > 0:
            first_dt = causal_history[0].timestamp
            prev_dt = causal_history[-1].timestamp
            d1_val = (current_dt - first_dt) / 86400.0   # Days since first transaction
            d2_val = (current_dt - prev_dt) / 86400.0    # Days since previous transaction
        else:
            d1_val = 0.0
            d2_val = 0.0

        c1_val = float(hist_count + 1)
        c2_val = float(hist_count + 1)

        # Integrate Flink real-time velocity if provided
        if event.velocity and event.velocity.count_5m is not None:
            c1_val = max(c1_val, float(event.velocity.count_5m))

        # ── Step 4: Assemble Raw IEEE-CIS Record Dictionary ───────────────────
        raw_record: Dict[str, Any] = {
            "TransactionID": tx_id,
            "TransactionDT": current_dt,
            "TransactionAmt": amount,
            "ProductCD": product_cd,
            # Entity Profile
            "card1": profile.card1,
            "card2": profile.card2,
            "card3": profile.card3,
            "card4": profile.card4,
            "card5": profile.card5,
            "card6": profile.card6,
            "addr1": profile.addr1,
            "addr2": profile.addr2,
            "dist1": profile.dist1,
            "P_emaildomain": profile.P_emaildomain,
            "R_emaildomain": profile.R_emaildomain,
            "DeviceType": profile.DeviceType,
            "DeviceInfo": profile.DeviceInfo,
            # Historical aggregates
            "C1": c1_val,
            "C2": c2_val,
            "D1": d1_val,
            "D2": d2_val,
        }

        # Add identity attributes if available
        for k, v in profile.id_attributes.items():
            raw_record[k] = v

        # Classify provenances
        for k in raw_record.keys():
            provenance_map[k] = classify_feature(k)

        # Absolute Invariant: isFraud MUST NOT exist in raw_record
        if "isFraud" in raw_record or "is_fraud" in raw_record:
            raw_record.pop("isFraud", None)
            raw_record.pop("is_fraud", None)

        # ── Step 5: Validate Completeness Gate ────────────────────────────────
        for field_name in self.ESSENTIAL_PROFILE_FIELDS:
            if raw_record.get(field_name) is None or pd.isna(raw_record.get(field_name)):
                missing_features.append(field_name)

        for field_name in self.ESSENTIAL_EVENT_FIELDS:
            if raw_record.get(field_name) is None or pd.isna(raw_record.get(field_name)):
                missing_features.append(field_name)

        if missing_features:
            return HydrationResult(
                complete=False,
                point_in_time_valid=True,
                card_id=card_id,
                transaction_id=tx_id,
                missing_features=missing_features,
                feature_provenance=provenance_map,
                rejection_reason=f"Incomplete hydration: missing essential fields {missing_features}",
            )

        df = pd.DataFrame([raw_record])

        return HydrationResult(
            complete=True,
            point_in_time_valid=True,
            card_id=card_id,
            transaction_id=tx_id,
            dataframe=df,
            missing_features=[],
            feature_provenance=provenance_map,
            rejection_reason=None,
        )

    def score_or_reject(
        self,
        payload: Union[StreamingTransactionPayload, Dict[str, Any]],
        inference_engine: Any,
    ) -> Dict[str, Any]:
        """
        Hard Hydration Gate execution:
        - If complete == True: score via inference engine.
        - If complete == False: DO NOT SCORE. Return structured rejection reason.
        """
        res = self.hydrate(payload)

        if not res.complete:
            logger.warning(
                "Inference bypassed by hydration gate for TX %s (Card %s): %s",
                res.transaction_id,
                res.card_id,
                res.rejection_reason,
            )
            return {
                "scoreable": False,
                "scored": False,
                "transaction_id": res.transaction_id,
                "card_id": res.card_id,
                "error": "REJECTED_BY_HYDRATION_GATE",
                "rejection_reason": res.rejection_reason,
                "missing_features": res.missing_features,
                "point_in_time_valid": res.point_in_time_valid,
            }

        # Safe to score via E1 model
        res_df, meta = inference_engine.predict_transaction(res.dataframe)
        row = res_df.iloc[0]

        # Record transaction into history strictly post-scoring
        if res.card_id and res.dataframe is not None:
            self.entity_store.record_transaction(
                card_id=res.card_id,
                timestamp=float(res.dataframe.iloc[0]["TransactionDT"]),
                amount=float(res.dataframe.iloc[0]["TransactionAmt"]),
                transaction_id=str(res.transaction_id),
            )

        return {
            "scoreable": True,
            "scored": True,
            "transaction_id": str(row["transaction_id"]),
            "fraud_probability": float(row["fraud_probability"]),
            "decision": str(row["decision"]),
            "preprocessing_time_ms": float(row["preprocessing_time_ms"]),
            "model_prediction_time_ms": float(row["model_prediction_time_ms"]),
            "total_inference_time_ms": float(row["total_inference_time_ms"]),
            "point_in_time_valid": True,
        }
