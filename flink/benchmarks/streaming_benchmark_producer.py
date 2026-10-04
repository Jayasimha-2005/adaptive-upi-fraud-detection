import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from kafka import KafkaProducer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--rate", type=int, default=500)
    parser.add_argument("--topic", default="fraud-transactions")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    args = parser.parse_args()

    if args.count <= 0 or args.rate <= 0:
        parser.error("--count and --rate must be positive")

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
    )

    start = time.perf_counter()
    base_time = datetime.now(timezone.utc)
    pending = []

    try:
        for i in range(args.count):
            event = {
                "transaction_id": f"BENCH-{args.count}-{i:07d}",
                "user_id": f"USER-{i % 1000:04d}",
                "amount": float(100 + (i % 4900)),
                "event_time": (
                    base_time + timedelta(seconds=i / args.rate)
                ).isoformat(),
                "merchant_id": f"M{(i % 100):03d}",
                "transaction_type": "UPI",
            }

            pending.append(producer.send(args.topic, event))

            if len(pending) >= 500:
                for future in pending:
                    future.get(timeout=60)
                pending.clear()

                elapsed = time.perf_counter() - start
                sent = min(i + 1, args.count)
                print(
                    f"Progress: {sent}/{args.count}; "
                    f"elapsed={elapsed:.2f}s"
                )

            # Pace the producer to avoid overwhelming the local machine.
            target_elapsed = (i + 1) / args.rate
            remaining = target_elapsed - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)

        for future in pending:
            future.get(timeout=60)
        producer.flush()

    finally:
        producer.close()

    elapsed = time.perf_counter() - start
    print("\n=== PRODUCER BENCHMARK ===")
    print(f"Topic: {args.topic}")
    print(f"Transactions acknowledged: {args.count}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print(f"Producer throughput: {args.count / elapsed:.2f} records/second")


if __name__ == "__main__":
    main()
