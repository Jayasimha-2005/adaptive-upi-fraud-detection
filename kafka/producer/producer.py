import argparse
import json
import os
import sys
import time
from pathlib import Path
from kafka import KafkaProducer

# Import transaction generator & dataset loader
generator_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "transaction_generator")
)
if generator_path not in sys.path:
    sys.path.insert(0, generator_path)

# pyrefly: ignore [missing-import]
from dataset_loader import DatasetTransactionStreamer


DEFAULT_BROKER = "localhost:9092"
DEFAULT_TOPIC = "transactions"
DEFAULT_COUNT = 50
DEFAULT_RATE = 20  # tx/sec


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Dataset-Driven Financial Transaction Kafka Producer"
    )
    parser.add_argument(
        "--broker",
        default=DEFAULT_BROKER,
        help=f"Kafka bootstrap broker (default: {DEFAULT_BROKER})",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"Target Kafka topic (default: {DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--dataset",
        default="default",
        help="Dataset name ('ieee_cis', 'credit_card_10k', 'paysim', 'synthetic') or path to CSV",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of transactions to send (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE,
        help=f"Target ingestion rate in tx/sec (default: {DEFAULT_RATE})",
    )
    parser.add_argument(
        "--acks",
        default="all",
        choices=["0", "1", "all"],
        help="Kafka ACK durability mode (default: all)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of producer retries on transient errors (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    ack_setting = 0 if args.acks == "0" else (1 if args.acks == "1" else "all")
    enable_idempotence = (args.acks == "all")

    streamer = DatasetTransactionStreamer(dataset_path=args.dataset, loop=True)

    print("=" * 80)
    print("KAFKA DATASET TRANSACTION PRODUCER")
    print("=" * 80)
    print(f"Broker               : {args.broker}")
    print(f"Topic                : {args.topic}")
    print(f"Dataset Source       : {streamer.dataset_file or 'Synthetic'}")
    print(f"Transactions to Send : {args.count}")
    print(f"Ingestion Rate       : {args.rate} tx/sec")
    print(f"ACK Configuration    : {args.acks}")
    print(f"Retries              : {args.retries}")
    print(f"Idempotence Enabled  : {enable_idempotence}")
    print("=" * 80)
    print()

    producer = KafkaProducer(
        bootstrap_servers=[args.broker],
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),
        acks=ack_setting,
        retries=args.retries,
        enable_idempotence=enable_idempotence,
    )

    interval = 1.0 / args.rate if args.rate > 0 else 0
    successful = 0
    failed = 0
    fraud_count = 0
    latencies = []

    start_time = time.time()
    next_send_time = start_time

    try:
        for i in range(1, args.count + 1):
            now = time.time()
            if interval > 0 and now < next_send_time:
                time.sleep(next_send_time - now)

            transaction = streamer.next_transaction()
            key = str(transaction.get("card_id", "CARD-DEFAULT"))
            if transaction.get("is_fraud"):
                fraud_count += 1

            send_start = time.time()
            try:
                future = producer.send(args.topic, key=key, value=transaction)
                if args.acks != "0":
                    metadata = future.get(timeout=10)
                    partition = metadata.partition
                    offset = metadata.offset
                else:
                    partition = "async"
                    offset = "async"

                send_end = time.time()
                latencies.append((send_end - send_start) * 1000)
                successful += 1

                print(
                    f"[{i}/{args.count}] "
                    f"Sent: {transaction['transaction_id']} | "
                    f"Card: {key} | "
                    f"Amt: ${transaction.get('amount', 0):.2f} | "
                    f"Fraud: {transaction.get('is_fraud')} | "
                    f"Part: {partition} | "
                    f"Offset: {offset}"
                )

            except Exception as e:
                failed += 1
                print(f"FAILED: {transaction['transaction_id']} | Error: {e}")

            next_send_time += interval

        producer.flush()

    finally:
        producer.close()
        streamer.close()

    total_time = time.time() - start_time
    effective_throughput = successful / total_time if total_time > 0 else 0

    print()
    print("=" * 80)
    print("PRODUCER SUMMARY")
    print("=" * 80)
    print(f"Successful messages  : {successful}")
    print(f"Failed messages      : {failed}")
    print(f"Fraudulent detected  : {fraud_count} ({(fraud_count/max(1, successful))*100:.2f}%)")
    print(f"Total time elapsed   : {total_time:.3f} s")
    print(f"Effective throughput : {effective_throughput:.2f} tx/sec")
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        p95 = sorted_lat[int(0.95 * len(sorted_lat))]
        print(f"Average latency      : {avg_lat:.2f} ms")
        print(f"P95 latency          : {p95:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
