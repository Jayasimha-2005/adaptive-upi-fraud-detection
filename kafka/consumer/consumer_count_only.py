import argparse
import time

from kafka import KafkaConsumer


BROKER = "localhost:9092"


def main():
    parser = argparse.ArgumentParser(
        description="Kafka count-only consumer benchmark"
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Kafka topic",
    )

    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of messages to receive",
    )

    parser.add_argument(
        "--group",
        required=True,
        help="Kafka consumer group",
    )

    args = parser.parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=[BROKER],
        group_id=args.group,

        auto_offset_reset="earliest",
        enable_auto_commit=True,

        # Fetch messages in batches
        max_poll_records=1000,

        # Allow Kafka to return data efficiently
        fetch_min_bytes=1,
        fetch_max_wait_ms=5,
    )

    received = 0
    start_time = None

    print("=" * 70)
    print("COUNT-ONLY CONSUMER BENCHMARK")
    print("=" * 70)
    print(f"Broker          : {BROKER}")
    print(f"Topic           : {args.topic}")
    print(f"Group           : {args.group}")
    print(f"Target messages : {args.count}")
    print("=" * 70)
    print()
    print("Waiting for messages...")

    try:
        while received < args.count:

            records = consumer.poll(
                timeout_ms=100
            )

            if not records:
                continue

            if start_time is None:
                start_time = time.perf_counter()

            for topic_partition, messages in records.items():

                remaining = args.count - received
                received += min(
                    len(messages),
                    remaining
                )

                if received >= args.count:
                    break

            elapsed = (
                time.perf_counter() - start_time
                if start_time
                else 0
            )

            if elapsed > 0:
                rate = received / elapsed

                print(
                    f"\rReceived: {received}/{args.count} | "
                    f"Rate: {rate:.2f} tx/sec",
                    end="",
                    flush=True
                )

        end_time = time.perf_counter()

        elapsed = end_time - start_time

        throughput = (
            received / elapsed
            if elapsed > 0
            else 0
        )

        print()
        print()
        print("=" * 70)
        print("RESULT")
        print("=" * 70)
        print(f"Messages received : {received}")
        print(f"Elapsed time      : {elapsed:.3f} sec")
        print(f"Consumer throughput: {throughput:.2f} tx/sec")
        print("=" * 70)

    except KeyboardInterrupt:
        print()
        print("Consumer stopped manually.")

    finally:
        consumer.close()


if __name__ == "__main__":
    main()
