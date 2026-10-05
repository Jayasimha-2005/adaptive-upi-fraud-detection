import json
import time
import argparse
import statistics
from kafka import KafkaConsumer


def percentile(values, percentile_value):
    if not values:
        return 0

    values = sorted(values)

    index = (percentile_value / 100) * (len(values) - 1)

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return values[lower] + (values[upper] - values[lower]) * weight


def main():

    parser = argparse.ArgumentParser(
        description="Kafka throughput and latency benchmark consumer"
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Kafka topic"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=2000,
        help="Number of transactions to receive"
    )

    parser.add_argument(
        "--group",
        required=True,
        help="Kafka consumer group"
    )

    args = parser.parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers="localhost:9092",
        group_id=args.group,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda x: json.loads(x.decode("utf-8"))
    )

    print("=" * 60)
    print("KAFKA THROUGHPUT & LATENCY BENCHMARK CONSUMER")
    print("=" * 60)
    print(f"Topic         : {args.topic}")
    print(f"Consumer group: {args.group}")
    print(f"Expected msgs : {args.count}")
    print("=" * 60)
    print()
    print("Waiting for transactions...")
    print()

    latencies = []
    received = 0

    start_time = None

    try:

        for message in consumer:

            receive_time = time.time()

            transaction = message.value

            sent_time = transaction.get("timestamp")

            if sent_time is not None:

                latency_ms = (receive_time - sent_time) * 1000

                # Ignore impossible negative values
                # caused by clock/timestamp issues.
                if latency_ms >= 0:
                    latencies.append(latency_ms)

            if start_time is None:
                start_time = time.perf_counter()

            received += 1

            if received % 100 == 0 or received == 1:
                print(
                    f"Received: {received}/{args.count} | "
                    f"Partition: {message.partition} | "
                    f"Offset: {message.offset}"
                )

            consumer.commit()

            if received >= args.count:
                break

    finally:

        end_time = time.perf_counter()

        consumer.close()

    if start_time is None:
        print("No transactions received.")
        return

    elapsed = end_time - start_time

    throughput = received / elapsed if elapsed > 0 else 0

    average_latency = (
        statistics.mean(latencies)
        if latencies
        else 0
    )

    median_latency = (
        statistics.median(latencies)
        if latencies
        else 0
    )

    p95_latency = percentile(latencies, 95)

    p99_latency = percentile(latencies, 99)

    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"Messages received : {received}")
    print(f"Elapsed time      : {elapsed:.3f} seconds")
    print(f"Consumer throughput: {throughput:.2f} tx/sec")

    print()
    print("LATENCY")
    print(f"Average latency   : {average_latency:.3f} ms")
    print(f"Median latency    : {median_latency:.3f} ms")
    print(f"P95 latency       : {p95_latency:.3f} ms")
    print(f"P99 latency       : {p99_latency:.3f} ms")

    print()
    print("=" * 60)

    if received == args.count:
        print("STATUS: SUCCESS - all expected messages received")
    else:
        print(
            f"STATUS: INCOMPLETE - expected {args.count}, "
            f"received {received}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
