"""
streaming/flink_processor.py
Phase 14: Member 2 Apache Flink Stream Processing & Stateful Velocity Engine.

Implements real-time stateful sliding window aggregations and CEP velocity scoring:
- Input: Ingests raw transactions from Kafka topic `ieee_cis_transactions`
- KeyedStream: Partitioned and evaluated per card_id / user_id
- State: Dual sliding windows (5-minute and 10-minute) with incremental O(1) deque eviction
- Output: Emits enriched velocity records to Kafka topic `fraud-features`
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union

from streaming.kafka_broker import ConsumerRecord, LocalKafkaConsumer, LocalKafkaProducer

logger = logging.getLogger("flink_processor")


class FlinkStatefulVelocityEngine:
    """
    Implements Flink's KeyedStream stateful sliding window operator:
    Keeps sliding window state per entity key to evaluate 5m and 10m velocity metrics.
    """

    def __init__(self, window_5m_sec: float = 300.0, window_10m_sec: float = 600.0) -> None:
        self.window_5m_sec = window_5m_sec
        self.window_10m_sec = window_10m_sec
        # State: card_key -> { 'q5m': deque of (ts, amt), 'sum5m': float, 'q10m': deque of (ts, amt), 'sum10m': float }
        self._state: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "q5m": deque(),
            "sum5m": 0.0,
            "q10m": deque(),
            "sum10m": 0.0,
        })

    def process_event(
        self,
        card_key: str,
        tx_id: str,
        event_timestamp: float,
        amount: float,
    ) -> Dict[str, Any]:
        """
        Process a transaction event incrementally in state and return the enriched velocity feature record.
        """
        st = self._state[card_key]
        q5m: deque = st["q5m"]
        q10m: deque = st["q10m"]

        # Incremental O(1) eviction for 5-minute sliding window
        cutoff_5m = event_timestamp - self.window_5m_sec
        while q5m and q5m[0][0] < cutoff_5m:
            old_ts, old_amt = q5m.popleft()
            st["sum5m"] -= old_amt

        # Incremental O(1) eviction for 10-minute sliding window
        cutoff_10m = event_timestamp - self.window_10m_sec
        while q10m and q10m[0][0] < cutoff_10m:
            old_ts, old_amt = q10m.popleft()
            st["sum10m"] -= old_amt

        # Append current event into sliding windows
        q5m.append((event_timestamp, amount))
        st["sum5m"] += amount

        q10m.append((event_timestamp, amount))
        st["sum10m"] += amount

        count_5m = len(q5m)
        total_5m = max(0.0, st["sum5m"])
        avg_5m = total_5m / max(1, count_5m)

        count_10m = len(q10m)
        total_10m = max(0.0, st["sum10m"])
        avg_10m = total_10m / max(1, count_10m)

        velocity_ratio = count_5m / max(1, count_10m)
        amount_ratio = total_5m / max(1.0, total_10m)

        if total_10m > 5000:
            amount_bucket = "VERY_HIGH"
        elif total_10m > 2000:
            amount_bucket = "HIGH"
        elif total_10m > 500:
            amount_bucket = "MEDIUM"
        else:
            amount_bucket = "LOW"

        return {
            "card_id": card_key,
            "user_id": card_key,
            "transaction_id": tx_id,
            "timestamp": event_timestamp,
            "event_time": event_timestamp,
            "transaction_count_5m": count_5m,
            "total_amount_5m": round(total_5m, 2),
            "average_amount_5m": round(avg_5m, 2),
            "transaction_count_10m": count_10m,
            "total_amount_10m": round(total_10m, 2),
            "average_amount_10m": round(avg_10m, 2),
            "transaction_velocity_ratio": round(velocity_ratio, 3),
            "amount_velocity_ratio": round(amount_ratio, 3),
            "amount_bucket_10m": amount_bucket,
        }


class FlinkStreamProcessor:
    """
    Subscribes to raw transactions from Kafka topic `ieee_cis_transactions`,
    runs Flink stateful window processing, and publishes to Kafka topic `fraud-features`.
    """

    def __init__(
        self,
        consumer: LocalKafkaConsumer,
        producer: LocalKafkaProducer,
        in_topic: str = "ieee_cis_transactions",
        out_topic: str = "fraud-features",
        window_5m_sec: float = 300.0,
        window_10m_sec: float = 600.0,
    ) -> None:
        self.consumer = consumer
        self.producer = producer
        self.in_topic = in_topic
        self.out_topic = out_topic
        self.engine = FlinkStatefulVelocityEngine(
            window_5m_sec=window_5m_sec,
            window_10m_sec=window_10m_sec,
        )
        self.processed_count = 0

    def process_records(self, max_records: int = 100) -> List[Dict[str, Any]]:
        """
        Poll input topic, process through stateful windows, and emit to output topic.
        """
        poll_res = self.consumer.poll(timeout_ms=100, max_records=max_records)
        generated_features: List[Dict[str, Any]] = []

        for tp, messages in poll_res.items():
            for msg in messages:
                tx = msg.value
                if not isinstance(tx, dict):
                    continue

                card_val = tx.get("card_id") or tx.get("card1") or tx.get("user_id") or "UNKNOWN"
                card_key = str(card_val)
                if not card_key.startswith("CARD-") and card_key.isdigit():
                    card_key = f"CARD-{card_key}"

                tx_id = str(tx.get("TransactionID") or tx.get("transaction_id") or f"TX-{self.processed_count}")
                amt = float(tx.get("TransactionAmt") or tx.get("amount", 0.0))
                raw_ts = tx.get("TransactionDT") or tx.get("timestamp") or tx.get("event_time", 0.0)
                try:
                    ts = float(raw_ts)
                    if ts > 1e11:
                        ts = ts / 1000.0
                except (ValueError, TypeError):
                    ts = time.time()

                feature_record = self.engine.process_event(
                    card_key=card_key,
                    tx_id=tx_id,
                    event_timestamp=ts,
                    amount=amt,
                )

                # Publish to fraud-features topic
                self.producer.send(
                    topic=self.out_topic,
                    key=card_key,
                    value=feature_record,
                    timestamp=ts,
                )
                generated_features.append(feature_record)
                self.processed_count += 1

        self.producer.flush()
        self.consumer.commit()
        return generated_features
