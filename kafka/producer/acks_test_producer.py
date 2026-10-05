import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from kafka import KafkaProducer

generator_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "transaction_generator")
)
sys.path.append(generator_path)
from dataset_loader import DatasetTransactionStreamer

BROKER = "localhost:9092"
DEFAULT_TOPIC = "acks_test"
DEFAULT_TOTAL_MESSAGES = 2000
DEFAULT_TARGET_RATE = 200


def calculate_percentile(values, percentile):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(percentile * len(sorted_values)) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def main():
    parser = argparse.ArgumentParser(description="Kafka ACKS Durability & Latency Benchmark")
    parser.add_argument("--acks", required=True, choices=["0", "1", "all"])
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--count", type=int, default=DEFAULT_TOTAL_MESSAGES)
    parser.add_argument("--rate", type=int, default=DEFAULT_TARGET_RATE)
    parser.add_argument("--dataset", default="default")

    args = parser.parse_args()

    ack_setting = 0 if args.acks == "0" else (1 if args.acks == "1" else "all")
    streamer = DatasetTransactionStreamer(dataset_path=args.dataset, loop=True)

    producer = KafkaProducer(
        bootstrap_servers=[BROKER],
        acks=ack_setting,
        retries=0,
        linger_ms=0,
        batch_size=16384,
        request_timeout_ms=5000,
        delivery_timeout_ms=7000,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
    )

    print("=" * 80)
    print("KAFKA ACKS EXPERIMENT (DATASET-POWERED)")
    print("=" * 80)
    print(f"Broker          : {BROKER}")
    print(f"Topic           : {args.topic}")
    print(f"Dataset         : {streamer.dataset_file or 'Synthetic'}")
    print(f"ACKS            : {args.acks}")
    print(f"Total messages  : {args.count}")
    print(f"Target rate     : {args.rate} tx/sec")
    print("=" * 80)
    print()

    latencies = []
    successful = 0
    failed = 0

    experiment_start = time.perf_counter()

    for i in range(1, args.count + 1):
        target_send_time = experiment_start + ((i - 1) / args.rate)
        sleep_time = target_send_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

        transaction = streamer.next_transaction()
        key = str(transaction.get("card_id", "CARD-0"))

        start_time = time.perf_counter()
        try:
            future = producer.send(args.topic, key=key, value=transaction)
            if args.acks != "0":
                future.get(timeout=5)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            successful += 1

            if i % 500 == 0 or i == args.count:
                print(f"[{i}/{args.count}] TX={transaction['transaction_id']} | ACKS={args.acks} | Latency={latency_ms:.2f} ms")

        except Exception as error:
            failed += 1
            print(f"FAILED | TX={transaction['transaction_id']} | Error={error}")

    producer.flush()
    producer.close()
    streamer.close()

    total_duration = time.perf_counter() - experiment_start
    actual_throughput = successful / total_duration if total_duration > 0 else 0

    print()
    print("=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"ACKS mode              : {args.acks}")
    print(f"Successful messages    : {successful}")
    print(f"Failed messages        : {failed}")
    print(f"Total duration (s)     : {total_duration:.3f}")
    print(f"Actual throughput      : {actual_throughput:.2f} msg/sec")

    if latencies:
        print()
        print("Latency distribution (ms):")
        print(f"  Min latency          : {min(latencies):.3f} ms")
        print(f"  Avg latency          : {statistics.mean(latencies):.3f} ms")
        print(f"  Median latency       : {statistics.median(latencies):.3f} ms")
        print(f"  P95 latency          : {calculate_percentile(latencies, 0.95):.3f} ms")
        print(f"  P99 latency          : {calculate_percentile(latencies, 0.99):.3f} ms")
        print(f"  Max latency          : {max(latencies):.3f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
