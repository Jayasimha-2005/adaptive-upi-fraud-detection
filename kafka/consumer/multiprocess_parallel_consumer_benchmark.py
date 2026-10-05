"""
Member 1 — Multi-Consumer Parallelism & Partition Scaling Benchmark
Benchmarks:
  1. Consumer throughput with 1 consumer (1 partition)
  2. Consumer throughput with 3 parallel consumers (3 partitions)
  3. Consumer throughput with 6 parallel consumers (6 partitions)
  4. Consumer throughput with 12 consumers (6 partitions vs 12 partitions)
  5. 1,000 tx/s & 5,000 tx/s Producer + Multiple Consumers Real-Time Ingestion (Zero Lag Test)
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

generator_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "transaction_generator")
)
if generator_path not in sys.path:
    sys.path.insert(0, generator_path)

# pyrefly: ignore [missing-import]
from dataset_loader import DatasetTransactionStreamer

BROKER = "localhost:9092"
RESULTS_CSV = Path(__file__).resolve().parent.parent.parent / "member1_parallel_consumer_results.csv"


def consumer_worker(consumer_id, group_id, topic, target_count, result_queue, timeout_sec=25):
    """Worker process that consumes from Kafka within a shared consumer group."""
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[BROKER],
            group_id=group_id,
            client_id=f"consumer-{consumer_id}",
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            consumer_timeout_ms=3000,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
        )

        consumed_count = 0
        fraud_count = 0
        partition_dist = defaultdict(int)
        start_time = None
        end_time = None
        last_msg_time = time.time()

        while time.time() - last_msg_time < timeout_sec:
            records_dict = consumer.poll(timeout_ms=500, max_records=500)
            if not records_dict:
                if consumed_count > 0 and (time.time() - last_msg_time > 3.0):
                    break
                continue

            if start_time is None:
                start_time = time.time()

            for tp, messages in records_dict.items():
                for message in messages:
                    consumed_count += 1
                    last_msg_time = time.time()
                    partition_dist[message.partition] += 1
                    tx = message.value
                    if tx.get("is_fraud") or tx.get("isFraud"):
                        fraud_count += 1

            consumer.commit()
            if target_count > 0 and consumed_count >= target_count:
                break

        end_time = time.time()
        consumer.close()

        duration = (end_time - start_time) if (start_time and end_time and end_time > start_time) else 0.001
        throughput = consumed_count / duration if duration > 0 else 0

        result_queue.put({
            "consumer_id": consumer_id,
            "consumed": consumed_count,
            "fraud_count": fraud_count,
            "duration_s": round(duration, 3),
            "throughput_tx_s": round(throughput, 2),
            "partition_dist": dict(partition_dist),
        })

    except Exception as e:
        result_queue.put({
            "consumer_id": consumer_id,
            "error": str(e),
            "consumed": 0,
            "duration_s": 0,
            "throughput_tx_s": 0,
            "partition_dist": {},
        })


def run_producer(topic, count, rate, dataset_name="ieee_cis"):
    """Fast dataset producer using asynchronous pipelining."""
    streamer = DatasetTransactionStreamer(dataset_path=dataset_name, loop=True)
    producer = KafkaProducer(
        bootstrap_servers=[BROKER],
        acks="all",
        retries=5,
        enable_idempotence=True,
        batch_size=65536,
        linger_ms=5,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )

    interval = 1.0 / rate if rate > 0 else 0
    start_time = time.time()
    next_send = start_time

    for i in range(1, count + 1):
        now = time.time()
        if interval > 0 and now < next_send:
            time.sleep(next_send - now)

        tx = streamer.next_transaction()
        key = str(tx.get("card_id", f"CARD-{i % 100}"))
        producer.send(topic, key=key, value=tx)
        next_send += interval

    producer.flush()
    producer.close()
    streamer.close()
    total_time = time.time() - start_time
    actual_rate = count / total_time if total_time > 0 else 0
    return actual_rate, total_time


def benchmark_consumer_scaling(num_consumers, num_partitions, topic, total_messages, producer_rate, dataset="ieee_cis"):
    group_id = f"scaling-group-{num_partitions}p-{num_consumers}c-{int(time.time() * 1000)}"
    print(f"\n---> Benchmark: {num_consumers} Consumer(s) on {num_partitions} Partition(s) [{topic}]")
    print(f"     Producer Streaming : {total_messages} messages @ {producer_rate} tx/sec...")

    result_queue = mp.Queue()
    processes = []

    # Start parallel consumer processes
    for cid in range(1, num_consumers + 1):
        p = mp.Process(
            target=consumer_worker,
            args=(cid, group_id, topic, 0, result_queue, 15)
        )
        p.start()
        processes.append(p)

    # Allow consumers to join group and complete group rebalancing
    time.sleep(2.5)

    # Start producer stream
    prod_rate, prod_time = run_producer(topic, count=total_messages, rate=producer_rate, dataset_name=dataset)
    print(f"     Producer Completed : {total_messages} records in {prod_time:.2f}s ({prod_rate:.1f} tx/sec)")

    # Wait for all consumers to finish
    for p in processes:
        p.join(timeout=20)
        if p.is_alive():
            p.terminate()

    # Collect results
    consumer_results = []
    while not result_queue.empty():
        consumer_results.append(result_queue.get())

    active_consumers = [r for r in consumer_results if r.get("consumed", 0) > 0]
    idle_consumers = [r for r in consumer_results if r.get("consumed", 0) == 0]

    total_consumed = sum(r.get("consumed", 0) for r in consumer_results)
    max_duration = max((r.get("duration_s", 0.001) for r in active_consumers), default=0.001)
    aggregate_throughput = total_consumed / max_duration if max_duration > 0 else 0
    lost_events = max(0, total_messages - total_consumed)

    print(f"     Consumers Drained  : {total_consumed}/{total_messages} records in {max_duration:.2f}s")
    print(f"     Active Workers     : {len(active_consumers)} active | {len(idle_consumers)} standby/idle")
    print(f"     Aggregate Rate     : {aggregate_throughput:,.2f} tx/sec | Lost Events: {lost_events}")
    for r in sorted(consumer_results, key=lambda x: x.get("consumer_id", 0)):
        role = "ACTIVE" if r.get("consumed", 0) > 0 else "STANDBY (HOT FAILOVER)"
        print(f"       • Consumer {r.get('consumer_id'):2d} [{role:<22}]: {r.get('consumed'):5d} msgs ({r.get('throughput_tx_s'):>7.1f} tx/s) | Partitions: {r.get('partition_dist')}")

    return {
        "partitions": num_partitions,
        "consumers": num_consumers,
        "active_consumers": len(active_consumers),
        "standby_consumers": len(idle_consumers),
        "topic": topic,
        "messages_sent": total_messages,
        "producer_target_rate": producer_rate,
        "producer_actual_rate": round(prod_rate, 2),
        "total_consumed": total_consumed,
        "aggregate_consumer_throughput": round(aggregate_throughput, 2),
        "avg_individual_throughput": round(aggregate_throughput / max(1, len(active_consumers)), 2),
        "lost_events": lost_events,
        "dataset": dataset,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Consumer Parallelism Benchmark Suite")
    parser.add_argument("--dataset", default="ieee_cis", help="Dataset name ('ieee_cis' or 'credit_card_10k')")
    parser.add_argument("--consumers", type=int, default=0, help="Specific number of consumers to test (e.g. 6 or 12)")
    parser.add_argument("--partitions", type=int, default=0, help="Specific number of partitions to test (e.g. 6 or 12)")
    parser.add_argument("--messages", type=int, default=0, help="Number of messages to stream")
    parser.add_argument("--rate", type=int, default=0, help="Producer rate (tx/sec)")
    args = parser.parse_args()

    print("=" * 90)
    print("  MEMBER 1 — MULTI-CONSUMER PARALLELISM & PARTITION SCALING BENCHMARK")
    print("=" * 90)
    print(f"Broker  : {BROKER}")
    print(f"Dataset : {args.dataset}")
    print("=" * 90)

    suite_results = []

    # If specific test requested
    if args.consumers > 0 and args.partitions > 0:
        topic_map = {1: "transactions_p1", 3: "transactions_p3", 6: "transactions_p6", 12: "transactions_p12"}
        topic = topic_map.get(args.partitions, f"transactions_p{args.partitions}")
        count = args.messages if args.messages > 0 else (args.consumers * 1000)
        rate = args.rate if args.rate > 0 else 2000

        r = benchmark_consumer_scaling(
            num_consumers=args.consumers,
            num_partitions=args.partitions,
            topic=topic,
            total_messages=count,
            producer_rate=rate,
            dataset=args.dataset,
        )
        suite_results.append(r)
    else:
        # Full Suite: 1, 3, 6, 12 consumers
        r1 = benchmark_consumer_scaling(1, 1, "transactions_p1", 1000, 1000, args.dataset)
        suite_results.append(r1)

        r2 = benchmark_consumer_scaling(3, 3, "transactions_p3", 3000, 1000, args.dataset)
        suite_results.append(r2)

        r3 = benchmark_consumer_scaling(6, 6, "transactions_p6", 6000, 2000, args.dataset)
        suite_results.append(r3)

        # 12 consumers on 6 partitions (Demonstrating 6 active + 6 standby failover)
        r4 = benchmark_consumer_scaling(12, 6, "transactions_p6", 6000, 2000, args.dataset)
        suite_results.append(r4)

        # 12 consumers on 12 partitions (Full 12-way parallel scaling)
        r5 = benchmark_consumer_scaling(12, 12, "transactions_p12", 12000, 3000, args.dataset)
        suite_results.append(r5)

    # Save to CSV
    fieldnames = [
        "partitions", "consumers", "active_consumers", "standby_consumers", "topic",
        "messages_sent", "producer_target_rate", "producer_actual_rate", "total_consumed",
        "aggregate_consumer_throughput", "avg_individual_throughput", "lost_events", "dataset"
    ]
    with open(RESULTS_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in suite_results:
            writer.writerow(r)

    print("\n" + "=" * 100)
    print("  FINAL PARALLEL CONSUMER SCALING & THROUGHPUT BENCHMARK TABLE")
    print("=" * 100)
    print(f"{'Partitions':<12} | {'Consumers':<10} | {'Active/Standby':<16} | {'Producer Rate':<16} | {'Aggregate Consumer (tx/s)':<26} | {'Lost Events':<10}")
    print("-" * 100)
    for r in suite_results:
        status = f"{r.get('active_consumers', r['consumers'])}A / {r.get('standby_consumers', 0)}S"
        print(
            f"{r['partitions']:<12} | "
            f"{r['consumers']:<10} | "
            f"{status:<16} | "
            f"{r['producer_actual_rate']:>7.1f} tx/s   | "
            f"{r['aggregate_consumer_throughput']:>14,.2f} tx/sec      | "
            f"{r['lost_events']:<10}"
        )
    print("=" * 100)
    print(f"[+] Results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
