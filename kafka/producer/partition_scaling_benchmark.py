import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from kafka import KafkaProducer

# Import dataset loader
generator_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "transaction_generator")
)
sys.path.append(generator_path)
from dataset_loader import DatasetTransactionStreamer

BROKER = "localhost:9092"
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
    parser = argparse.ArgumentParser(
        description="Kafka partition scaling benchmark with Real Datasets"
    )
    parser.add_argument("--topic", required=True, help="Kafka topic to benchmark (e.g. transactions_p1, transactions_p3, transactions_p6)")
    parser.add_argument("--count", type=int, default=DEFAULT_TOTAL_MESSAGES, help="Total transactions to send")
    parser.add_argument("--rate", type=int, default=DEFAULT_TARGET_RATE, help="Target ingestion rate (tx/sec)")
    parser.add_argument("--dataset", default="default", help="Dataset name or path")

    args = parser.parse_args()
    streamer = DatasetTransactionStreamer(dataset_path=args.dataset, loop=True)

    producer = KafkaProducer(
        bootstrap_servers=[BROKER],
        acks="all",
        retries=5,
        enable_idempotence=True,
        batch_size=16384,
        linger_ms=5,
        request_timeout_ms=10000,
        delivery_timeout_ms=15000,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),
    )

    print("=" * 80)
    print("KAFKA PARTITION SCALING BENCHMARK (DATASET-POWERED)")
    print("=" * 80)
    print(f"Broker             : {BROKER}")
    print(f"Topic              : {args.topic}")
    print(f"Dataset            : {streamer.dataset_file or 'Synthetic'}")
    print(f"Total messages     : {args.count}")
    print(f"Target rate        : {args.rate} tx/sec")
    print(f"ACKS               : all (idempotent)")
    print("=" * 80)
    print()

    latencies = []
    successful = 0
    failed = 0
    partition_counts = {}

    experiment_start = time.perf_counter()

    for i in range(1, args.count + 1):
        target_send_time = experiment_start + ((i - 1) / args.rate)
        sleep_time = target_send_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

        transaction = streamer.next_transaction()
        # Card ID serves as partition key ensuring card-level per-partition ordering
        key = str(transaction.get("card_id", "CARD-0"))

        start_time = time.perf_counter()
        try:
            future = producer.send(args.topic, key=key, value=transaction)
            metadata = future.get(timeout=10)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            partition = metadata.partition
            partition_counts[partition] = partition_counts.get(partition, 0) + 1
            successful += 1

            if i % 250 == 0 or i == args.count:
                print(
                    f"[{i}/{args.count}] TX={transaction['transaction_id']} | "
                    f"Card={key} | Part={partition} | "
                    f"Offset={metadata.offset} | Latency={latency_ms:.2f} ms"
                )

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
    print("PARTITION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Topic                  : {args.topic}")
    print(f"Successful messages    : {successful}")
    print(f"Failed messages        : {failed}")
    print(f"Total time (s)         : {total_duration:.3f}")
    print(f"Actual throughput      : {actual_throughput:.2f} msg/sec")
    print()
    print("Partition Distribution:")
    for partition in sorted(partition_counts.keys()):
        count = partition_counts[partition]
        pct = (count / successful) * 100 if successful > 0 else 0
        print(f"  Partition {partition:2d}           : {count} messages ({pct:.1f}%)")

    if latencies:
        print()
        print("Latency (ms):")
        print(f"  Min latency          : {min(latencies):.3f} ms")
        print(f"  Avg latency          : {statistics.mean(latencies):.3f} ms")
        print(f"  Median latency       : {statistics.median(latencies):.3f} ms")
        print(f"  P95 latency          : {calculate_percentile(latencies, 0.95):.3f} ms")
        print(f"  P99 latency          : {calculate_percentile(latencies, 0.99):.3f} ms")
        print(f"  Max latency          : {max(latencies):.3f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
