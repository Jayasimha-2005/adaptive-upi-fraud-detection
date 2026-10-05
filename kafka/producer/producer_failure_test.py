import argparse
import json
import os
import sys
import time
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import KafkaError

generator_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "transaction_generator")
)
sys.path.append(generator_path)
from dataset_loader import DatasetTransactionStreamer

BROKER = "localhost:9092"
DEFAULT_TOPIC = "producer_failure_test"
DEFAULT_TOTAL_MESSAGES = 100
DEFAULT_INTERVAL = 0.2


def main():
    parser = argparse.ArgumentParser(description="Kafka Producer Failure & Retry Test (Dataset-Driven)")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--count", type=int, default=DEFAULT_TOTAL_MESSAGES)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--dataset", default="default")

    args = parser.parse_args()
    streamer = DatasetTransactionStreamer(dataset_path=args.dataset, loop=True)

    producer = KafkaProducer(
        bootstrap_servers=[BROKER],
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),
        acks="all",
        retries=5,
        retry_backoff_ms=500,
        request_timeout_ms=3000,
        delivery_timeout_ms=10000,
        enable_idempotence=True,
    )

    successful = 0
    failed = 0

    print("=" * 80)
    print("KAFKA PRODUCER FAILURE & RETRY RECOVERY TEST (DATASET-POWERED)")
    print("=" * 80)
    print(f"Broker           : {BROKER}")
    print(f"Topic            : {args.topic}")
    print(f"Dataset          : {streamer.dataset_file or 'Synthetic'}")
    print(f"Total messages   : {args.count}")
    print(f"Message interval : {args.interval} seconds")
    print("ACK mode         : all")
    print("Retries          : 5")
    print("Idempotence      : enabled")
    print("=" * 80)
    print()

    for i in range(1, args.count + 1):
        transaction = streamer.next_transaction()
        key = str(transaction.get("card_id", "CARD-0"))

        try:
            future = producer.send(args.topic, key=key, value=transaction)
            metadata = future.get(timeout=10)
            successful += 1

            print(
                f"[{i}/{args.count}] SUCCESS | "
                f"TX={transaction['transaction_id']} | "
                f"Card={key} | "
                f"Amt=${transaction.get('amount', 0):.2f} | "
                f"Part={metadata.partition} | "
                f"Offset={metadata.offset}"
            )

        except KafkaError as error:
            failed += 1
            print(f"[{i}/{args.count}] FAILED  | TX={transaction['transaction_id']} | Error={error}")

        time.sleep(args.interval)

    producer.flush()
    producer.close()
    streamer.close()

    print()
    print("=" * 80)
    print("PRODUCER FAILURE TEST RESULTS")
    print("=" * 80)
    print(f"Messages attempted : {args.count}")
    print(f"Successful         : {successful}")
    print(f"Failed             : {failed}")
    print("=" * 80)


if __name__ == "__main__":
    main()
