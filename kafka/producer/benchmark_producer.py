import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Import dataset loader
generator_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "transaction_generator")
)
sys.path.append(generator_path)
from dataset_loader import DatasetTransactionStreamer


BROKER = "localhost:9092"


def percentile(values, percentage):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int((percentage / 100) * len(sorted_values))
    if index >= len(sorted_values):
        index = len(sorted_values) - 1
    return sorted_values[index]


def main():
    parser = argparse.ArgumentParser(
        description="Kafka throughput, latency and reliability benchmark using Real Datasets"
    )
    parser.add_argument("--topic", required=True, help="Kafka topic")
    parser.add_argument("--rate", type=int, required=True, help="Target transactions per second")
    parser.add_argument("--count", type=int, default=2000, help="Number of transactions to send")
    parser.add_argument("--dataset", default="default", help="Dataset name ('ieee_cis', 'credit_card_10k', 'paysim') or CSV path")
    parser.add_argument("--acks", default="all", choices=["0", "1", "all"], help="Kafka ACK durability mode")
    parser.add_argument("--batch-size", type=int, default=16384, help="Producer batch.size in bytes")
    parser.add_argument("--linger-ms", type=int, default=5, help="Producer linger.ms")
    parser.add_argument("--compression", default="none", choices=["none", "gzip", "snappy", "lz4"], help="Compression type")

    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("Rate must be greater than 0")
    if args.count <= 0:
        raise ValueError("Count must be greater than 0")

    streamer = DatasetTransactionStreamer(dataset_path=args.dataset, loop=True)
    ack_setting = 0 if args.acks == "0" else (1 if args.acks == "1" else "all")
    enable_idempotence = (args.acks == "all")

    producer_kwargs = {
        "bootstrap_servers": [BROKER],
        "value_serializer": lambda value: json.dumps(value).encode("utf-8"),
        "key_serializer": lambda key: key.encode("utf-8"),
        "acks": ack_setting,
        "retries": 5 if enable_idempotence else 0,
        "enable_idempotence": enable_idempotence,
        "batch_size": args.batch_size,
        "linger_ms": args.linger_ms,
    }
    if args.compression != "none":
        producer_kwargs["compression_type"] = args.compression

    producer = KafkaProducer(**producer_kwargs)

    print("=" * 80)
    print("KAFKA THROUGHPUT, LATENCY & RELIABILITY BENCHMARK (DATASET-DRIVEN)")
    print("=" * 80)
    print(f"Broker              : {BROKER}")
    print(f"Topic               : {args.topic}")
    print(f"Dataset Source      : {streamer.dataset_file or 'Synthetic'}")
    print(f"Target rate         : {args.rate} tx/sec")
    print(f"Message count       : {args.count}")
    print(f"ACK mode            : {args.acks}")
    print(f"Batch Size          : {args.batch_size} bytes | Linger: {args.linger_ms} ms")
    print(f"Compression         : {args.compression}")
    print("=" * 80)
    print()

    interval = 1.0 / args.rate if args.rate > 0 else 0
    successful = 0
    failed = 0
    fraud_count = 0
    latencies = []

    start_time = time.perf_counter()
    next_send_time = start_time

    for i in range(1, args.count + 1):
        now = time.perf_counter()
        if interval > 0 and now < next_send_time:
            time.sleep(next_send_time - now)

        transaction = streamer.next_transaction()
        if transaction.get("is_fraud"):
            fraud_count += 1
        key = str(transaction.get("card_id", "CARD-DEFAULT"))

        send_start = time.perf_counter()
        try:
            future = producer.send(args.topic, key=key, value=transaction)
            if args.acks != "0":
                metadata = future.get(timeout=10)
                part = metadata.partition
                off = metadata.offset
            else:
                part = "async"
                off = "async"

            send_end = time.perf_counter()
            latency_ms = (send_end - send_start) * 1000
            latencies.append(latency_ms)
            successful += 1

            if i % 250 == 0 or i == args.count:
                print(
                    f"[{i}/{args.count}] SUCCESS | "
                    f"TX={transaction['transaction_id']} | "
                    f"Card={key} | "
                    f"Amt=${transaction.get('amount', 0):.2f} | "
                    f"Part={part} | Off={off} | "
                    f"Lat={latency_ms:.2f} ms"
                )

        except KafkaError as error:
            failed += 1
            print(f"FAILED | TX={transaction['transaction_id']} | Error={type(error).__name__}: {error}")

        next_send_time += interval

    producer.flush()
    producer.close()
    streamer.close()

    total_time = time.perf_counter() - start_time
    actual_rate = successful / total_time if total_time > 0 else 0
    loss_rate = (failed / (successful + failed)) * 100 if (successful + failed) > 0 else 0

    print()
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Dataset             : {Path(str(streamer.dataset_file)).name if streamer.dataset_file else 'Synthetic'}")
    print(f"Target rate         : {args.rate} tx/sec")
    print(f"Actual throughput   : {actual_rate:.2f} tx/sec")
    print(f"Messages sent       : {successful}")
    print(f"Messages lost/failed: {failed}")
    print(f"Loss rate           : {loss_rate:.2f}%")
    print(f"Fraudulent replayed : {fraud_count}")
    print(f"Total time          : {total_time:.3f} s")

    if latencies:
        print()
        print("Latency distribution (ms):")
        print(f"  Min latency       : {min(latencies):.3f} ms")
        print(f"  Avg latency       : {statistics.mean(latencies):.3f} ms")
        print(f"  Median latency    : {statistics.median(latencies):.3f} ms")
        print(f"  P95 latency       : {percentile(latencies, 95):.3f} ms")
        print(f"  P99 latency       : {percentile(latencies, 99):.3f} ms")
        print(f"  Max latency       : {max(latencies):.3f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
