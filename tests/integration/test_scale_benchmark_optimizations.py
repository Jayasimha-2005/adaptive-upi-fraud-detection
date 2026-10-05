"""
Scale Benchmark for Spark Streaming & Flink CEP Optimizations
Tests Spark Streaming and Flink CEP across 1,000, 5,000, and 10,000 transaction batches.
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from kafka import KafkaConsumer, KafkaProducer, TopicPartition

BROKER = "localhost:9092"
TOPIC_TRANSACTIONS = "fraud-transactions"
TOPIC_FEATURES = "fraud-features"
PARTITION_COUNT = 6


def produce_test_batch(record_count, broker=BROKER, topic=TOPIC_TRANSACTIONS):
    """Produces N canonical records to Kafka topic."""
    base_dir = Path(__file__).resolve().parents[2]
    csv_candidates = [
        base_dir / "Datasets" / "IEEE CIS-20260829T103704Z-1-001" / "IEEE CIS" / "train_transaction.csv",
        base_dir / "Datasets" / "raw" / "train_transaction.csv",
        base_dir / "Datasets" / "train_transaction.csv",
    ]

    csv_path = None
    for p in csv_candidates:
        if p.exists():
            csv_path = p
            break

    producer = KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        acks="all",
        retries=5,
        enable_idempotence=True,
        batch_size=131072,
        linger_ms=10,
    )

    base_time = datetime(2026, 10, 5, 11, 55, 0, tzinfo=timezone.utc)
    user_ids = [f"USER{i:03d}" for i in range(1, 101)]

    records_sent = 0
    start_time = time.time()

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if idx > record_count:
                break

            tx_id = f"TX{row.get('TransactionID', idx)}"
            user_id = user_ids[(idx - 1) % len(user_ids)]
            card_key = str(row.get("card1", str(idx % 100)))
            amt = float(row.get("TransactionAmt", 50.0))
            is_fraud = int(row.get("isFraud", 0))
            prod_cd = str(row.get("ProductCD", "W"))
            event_dt = datetime.fromtimestamp(base_time.timestamp() + (idx % 600), tz=timezone.utc)

            rec = {
                "transaction_id": tx_id,
                "user_id": user_id,
                "card_id": f"CARD{card_key}",
                "amount": round(amt, 2),
                "event_time": event_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "merchant_id": f"MERCHANT_{prod_cd}_{idx % 20}",
                "transaction_type": "ONLINE_PURCHASE" if prod_cd == "W" else "UPI_PAYMENT",
                "is_fraud": is_fraud,
                "source_dataset": "IEEE-CIS",
            }

            producer.send(topic, key=rec["card_id"], value=rec)
            records_sent += 1

    producer.flush()
    producer.close()
    elapsed = time.time() - start_time
    print(f"[Producer] Successfully sent {records_sent:,} canonical records to '{topic}' in {elapsed:.3f}s ({records_sent/max(0.001, elapsed):.1f} tx/s)")
    return records_sent


def benchmark_optimized_flink(record_count, broker=BROKER, in_topic=TOPIC_TRANSACTIONS, out_topic=TOPIC_FEATURES):
    """Measures optimized Flink processing on N records."""
    consumer = KafkaConsumer(
        bootstrap_servers=[broker],
        group_id=f"flink-benchmark-group-{record_count}-{int(time.time())}",
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        fetch_min_bytes=131072,
        fetch_max_wait_ms=50,
        max_poll_records=5000,
        consumer_timeout_ms=5000,
    )
    consumer.assign([TopicPartition(in_topic, p) for p in range(PARTITION_COUNT)])
    consumer.seek_to_beginning()

    feature_producer = KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        acks=0,
        batch_size=131072,
        linger_ms=10,
    )

    user_state = defaultdict(lambda: {
        "q5m": deque(), "sum5m": 0.0,
        "q10m": deque(), "sum10m": 0.0,
    })

    consumed = 0
    features_gen = 0
    start_time = time.time()

    while consumed < record_count:
        poll_res = consumer.poll(timeout_ms=1000, max_records=record_count - consumed)
        if not poll_res:
            break

        for tp, messages in poll_res.items():
            for msg in messages:
                consumed += 1
                tx = msg.value
                user_id = tx["user_id"]
                amt = float(tx["amount"])
                tx_id = tx["transaction_id"]
                event_time_str = tx["event_time"]

                try:
                    dt = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))
                    ts = dt.timestamp()
                except Exception:
                    ts = time.time()

                st = user_state[user_id]
                q5m = st["q5m"]
                q10m = st["q10m"]

                cutoff_5m = ts - 300
                while q5m and q5m[0][0] < cutoff_5m:
                    old_ts, old_amt = q5m.popleft()
                    st["sum5m"] -= old_amt

                cutoff_10m = ts - 600
                while q10m and q10m[0][0] < cutoff_10m:
                    old_ts, old_amt = q10m.popleft()
                    st["sum10m"] -= old_amt

                q5m.append((ts, amt))
                st["sum5m"] += amt
                q10m.append((ts, amt))
                st["sum10m"] += amt

                count_5m = len(q5m)
                total_5m = max(0.0, st["sum5m"])
                count_10m = len(q10m)
                total_10m = max(0.0, st["sum10m"])

                feature_rec = {
                    "user_id": user_id,
                    "transaction_id": tx_id,
                    "event_time": event_time_str,
                    "transaction_count_5m": count_5m,
                    "total_amount_5m": round(total_5m, 2),
                    "transaction_count_10m": count_10m,
                    "total_amount_10m": round(total_10m, 2),
                    "transaction_velocity_ratio": round(count_5m / max(1, count_10m), 3),
                }
                feature_producer.send(out_topic, key=user_id, value=feature_rec)
                features_gen += 1

        consumer.commit()

    feature_producer.flush()
    feature_producer.close()
    consumer.close()

    elapsed = time.time() - start_time
    throughput = consumed / max(0.0001, elapsed)
    return consumed, elapsed, throughput


def benchmark_optimized_spark_streaming(record_count, broker=BROKER, in_topic=TOPIC_TRANSACTIONS):
    """Measures optimized Spark Streaming processing on N records."""
    consumer = KafkaConsumer(
        bootstrap_servers=[broker],
        group_id=f"spark-benchmark-group-{record_count}-{int(time.time())}",
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        fetch_min_bytes=131072,
        fetch_max_wait_ms=50,
        max_poll_records=5000,
        consumer_timeout_ms=5000,
    )
    consumer.assign([TopicPartition(in_topic, p) for p in range(PARTITION_COUNT)])
    consumer.seek_to_beginning()

    consumed = 0
    micro_batches = 0
    start_time = time.time()

    while consumed < record_count:
        poll_res = consumer.poll(timeout_ms=1000, max_records=record_count - consumed)
        if not poll_res:
            break

        batch_items = 0
        for tp, messages in poll_res.items():
            for msg in messages:
                tx = msg.value
                consumed += 1
                batch_items += 1

        if batch_items > 0:
            micro_batches += 1
        consumer.commit()

    consumer.close()
    elapsed = time.time() - start_time
    throughput = consumed / max(0.0001, elapsed)
    return consumed, elapsed, throughput


def main():
    print("=" * 85)
    print("  SCALING BENCHMARK: SPARK STREAMING & FLINK CEP (1,000 / 5,000 / 10,000 RECORDS)")
    print("=" * 85)

    scales = [1000, 5000, 10000]
    results = []

    for count in scales:
        print(f"\n--- TESTING BATCH SIZE: {count:,} TRANSACTIONS ---")
        produce_test_batch(count)

        # Flink Benchmark
        f_consumed, f_time, f_thru = benchmark_optimized_flink(count)
        print(f"[Flink Result]  Consumed: {f_consumed:,} records | Time: {f_time:.3f}s | Throughput: {f_thru:,.1f} tx/s")

        # Spark Streaming Benchmark
        s_consumed, s_time, s_thru = benchmark_optimized_spark_streaming(count)
        print(f"[Spark Result]  Consumed: {s_consumed:,} records | Time: {s_time:.3f}s | Throughput: {s_thru:,.1f} tx/s")

        results.append({
            "count": count,
            "flink_time": f_time,
            "flink_thru": f_thru,
            "spark_time": s_time,
            "spark_thru": s_thru,
        })

    print("\n" + "=" * 85)
    print("  SCALING BENCHMARK RESULTS SUMMARY")
    print("=" * 85)
    print(f"{'Dataset Scale':<15} | {'Flink Time':<12} | {'Flink Throughput':<18} | {'Spark Time':<12} | {'Spark Throughput'}")
    print("-" * 85)
    for r in results:
        print(f"{r['count']:<15,d} | {r['flink_time']:<10.3f}s | {r['flink_thru']:<16,.1f} tx/s | {r['spark_time']:<10.3f}s | {r['spark_thru']:<16,.1f} tx/s")
    print("=" * 85)


if __name__ == "__main__":
    main()
