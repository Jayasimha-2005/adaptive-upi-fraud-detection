"""
streaming/stream_pipeline.py
Phase 14: End-to-End Cross-Member Streaming Pipeline Orchestrator.

Wires together:
1. Member 1: Kafka Ingestion Producer (`ieee_cis_transactions`)
2. Member 2: Flink Stateful CEP Velocity Stream Processor (`fraud-features`)
3. Member 3: StreamServingBridge -> OnlineFeatureHydrationAdapter -> Canonical E1 LightGBM Serving

Enforces:
- Strict zero-target-leakage at ingress
- Temporal causality and point-in-time validity
- Complete transaction-level traceability from raw Kafka offset to model prediction
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from serving.hydration.adapter import OnlineFeatureHydrationAdapter
from serving.hydration.entity_store import EntityProfileStore
from serving.inference.offline_inference import (
    E1_DECISION_THRESHOLD,
    OfflineInferenceEngine,
)
from streaming.flink_processor import FlinkStreamProcessor
from streaming.kafka_broker import (
    ConsumerRecord,
    LocalKafkaBroker,
    LocalKafkaConsumer,
    LocalKafkaProducer,
    TopicPartition,
)
from streaming.stream_serving_bridge import StreamServingBridge

logger = logging.getLogger("stream_pipeline")

RAW_TRANSACTIONS_TOPIC = "ieee_cis_transactions"
FRAUD_FEATURES_TOPIC = "fraud-features"


@dataclass
class TransactionTrace:
    transaction_id: str
    card_id: str
    amount: float
    timestamp: float
    raw_topic: str
    raw_partition: int
    raw_offset: int
    flink_features_generated: bool
    features_topic: Optional[str] = None
    features_partition: Optional[int] = None
    features_offset: Optional[int] = None
    velocity_count_5m: Optional[int] = None
    velocity_amount_5m: Optional[float] = None
    correlated_velocity: bool = False
    hydration_scoreable: bool = False
    model_invoked: bool = False
    fraud_probability: Optional[float] = None
    decision: Optional[str] = None
    threshold: float = E1_DECISION_THRESHOLD
    latency_ms: float = 0.0
    error: Optional[str] = None


class StreamingPipelineOrchestrator:
    """
    Coordinates real cross-member streaming execution across Kafka, Flink, Bridge, and Serving.
    """

    def __init__(
        self,
        broker: Optional[LocalKafkaBroker] = None,
        entity_store: Optional[EntityProfileStore] = None,
        raw_topic: str = RAW_TRANSACTIONS_TOPIC,
        features_topic: str = FRAUD_FEATURES_TOPIC,
        partitions: int = 6,
    ) -> None:
        self.raw_topic = raw_topic
        self.features_topic = features_topic
        self.partitions = partitions

        # 1. Kafka Broker Infrastructure
        self.broker = broker or LocalKafkaBroker()
        self.broker.create_topic(self.raw_topic, partitions=self.partitions)
        self.broker.create_topic(self.features_topic, partitions=self.partitions)

        # 2. Producer clients
        self.producer = self.broker.create_producer()

        # 3. Member 2 Flink Streaming Processor
        flink_consumer = self.broker.create_consumer(
            self.raw_topic,
            group_id="flink-streaming-velocity-group",
        )
        flink_producer = self.broker.create_producer()
        self.flink_processor = FlinkStreamProcessor(
            consumer=flink_consumer,
            producer=flink_producer,
            in_topic=self.raw_topic,
            out_topic=self.features_topic,
        )

        # 4. Member 3 Serving Layer & Bridge
        self.entity_store = entity_store or EntityProfileStore()
        self.entity_store.seed_mock_profiles()
        self.adapter = OnlineFeatureHydrationAdapter(entity_store=self.entity_store)
        self.engine = OfflineInferenceEngine(threshold=E1_DECISION_THRESHOLD)
        self.bridge = StreamServingBridge(
            adapter=self.adapter,
            engine=self.engine,
        )

        # Lineage log
        self.traces: List[TransactionTrace] = []

    def ingest_transaction(
        self,
        raw_event: Dict[str, Any],
        partition: Optional[int] = None,
    ) -> TransactionTrace:
        """
        Ingest a raw transaction event into the streaming pipeline:
        Member 1 Produce -> Member 2 Flink Process -> Member 3 Bridge & Serve.
        """
        start_t = time.perf_counter()

        # ── Step 1: Member 1 Produce to Kafka ─────────────────────────────────
        card_val = raw_event.get("card_id") or raw_event.get("card1") or raw_event.get("user_id") or "0"
        card_key = str(card_val)
        if not card_key.startswith("CARD-") and card_key.isdigit():
            card_key = f"CARD-{card_key}"

        tx_id = str(raw_event.get("TransactionID") or raw_event.get("transaction_id") or "UNKNOWN")
        amt = float(raw_event.get("TransactionAmt") or raw_event.get("amount") or 0.0)
        dt = float(raw_event.get("TransactionDT") or raw_event.get("timestamp") or time.time())

        meta_raw = self.producer.send(
            topic=self.raw_topic,
            key=card_key,
            value=raw_event,
            partition=partition,
            timestamp=dt,
        )
        self.producer.flush()

        trace = TransactionTrace(
            transaction_id=tx_id,
            card_id=card_key,
            amount=amt,
            timestamp=dt,
            raw_topic=meta_raw.topic,
            raw_partition=meta_raw.partition,
            raw_offset=meta_raw.offset,
            flink_features_generated=False,
        )

        # ── Step 2: Member 2 Flink Consumption & Window Aggregation ───────────
        features = self.flink_processor.process_records(max_records=10)
        matching_feat: Optional[Dict[str, Any]] = None
        for f in features:
            if str(f.get("transaction_id")) == tx_id:
                matching_feat = f
                break

        if matching_feat:
            trace.flink_features_generated = True
            trace.features_topic = self.features_topic
            trace.velocity_count_5m = matching_feat.get("transaction_count_5m")
            trace.velocity_amount_5m = matching_feat.get("total_amount_5m")

            # Look up offset in fraud-features partition
            feat_part = self.broker.partition_for_key(self.features_topic, card_key)
            part_log = self.broker._logs.get(self.features_topic, {}).get(feat_part, [])
            trace.features_partition = feat_part
            trace.features_offset = len(part_log) - 1 if part_log else 0

        # ── Step 3: Member 3 Bridge Correlation & Serving ─────────────────────
        score_res = self.bridge.process_transaction(
            raw_event=raw_event,
            feature_record=matching_feat,
        )

        trace.correlated_velocity = score_res.get("correlated_velocity", False)
        trace.hydration_scoreable = score_res.get("scoreable", False)
        trace.model_invoked = score_res.get("scored", False)

        if score_res.get("scored"):
            trace.fraud_probability = score_res.get("fraud_probability")
            trace.decision = score_res.get("decision")
        else:
            trace.error = score_res.get("error") or score_res.get("rejection_reason")

        trace.latency_ms = (time.perf_counter() - start_t) * 1000.0
        self.traces.append(trace)
        return trace
