import argparse
import json
import statistics
import time
import uuid

from kafka import KafkaProducer
from kafka.errors import KafkaError


BROKER = "localhost:9092"


def create_transaction(transaction_number):
    return {
        "transaction_id": f"BENCH-{transaction_number}-{uuid.uuid4().hex[:8]}",
        "card_id": f"CARD-{transaction_number % 1000:04d}",
        "amount": round(100 + (transaction_number % 10000) * 0.75, 2),
        "merchant_id": f"M{(transaction_number % 100) + 100}",
        "timestamp": time.time(),
        "device_type": ["mobile", "web", "pos"][transaction_number % 3],
        "country": ["IN", "US", "UK", "AE"][transaction_number % 4],
        "is_fraud": False,
    }


def percentile(values, percentage):
    if not values:
        return 0.0

    values = sorted(values)

    index = int((percentage / 100) * len(values))

    if index >= len(values):
        index = len(values) - 1

    return values[index]


def main():
    parser = argparse.ArgumentParser(
        description="Asynchronous Kafka throughput benchmark"
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Kafka topic",
    )

    parser.add_argument(
        "--rate",
        type=int,
        required=True,
        help="Target transactions per second",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10000,
        help="Number of transactions",
    )

    args = parser.parse_args()

    if args.rate <= 0:
        raise ValueError("Rate must be greater than 0")

    if args.count <= 0:
        raise ValueError("Count must be greater than 0")

    producer = KafkaProducer(
        bootstrap_servers=[BROKER],
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),

        # Reliability
        acks="all",
        retries=5,
        enable_idempotence=True,

        # Batching
        batch_size=64 * 1024,
        linger_ms=5,

        # Multiple requests can remain in flight
        max_in_flight_requests_per_connection=5,
    )

    print("=" * 75)
    print("ASYNCHRONOUS KAFKA THROUGHPUT BENCHMARK")
    print("=" * 75)
    print(f"Broker              : {BROKER}")
    print(f"Topic               : {args.topic}")
    print(f"Target rate         : {args.rate} tx/sec")
    print(f"Message count       : {args.count}")
    print("ACK mode            : all")
    print("Retries             : 5")
    print("Idempotence         : enabled")
    print("Batch size          : 64 KB")
    print("Linger               : 5 ms")
    print("=" * 75)
    print()

    interval = 1.0 / args.rate

    # Store futures and their send timestamps.
    futures = []
    send_times = []

    start_time = time.perf_counter()
    next_send_time = start_time

    print("Sending messages...")

    for i in range(1, args.count + 1):

        now = time.perf_counter()

        if now < next_send_time:
            time.sleep(next_send_time - now)

        transaction = create_transaction(i)

        send_start = time.perf_counter()

        try:
            future = producer.send(
                args.topic,
                key=transaction["card_id"],
                value=transaction,
            )

            futures.append(future)
            send_times.append(send_start)

        except KafkaError as error:
            print(
                f"SEND FAILED | "
                f"Transaction={transaction['transaction_id']} | "
                f"Error={type(error).__name__}: {error}"
            )

        except Exception as error:
            print(
                f"SEND FAILED | "
                f"Transaction={transaction['transaction_id']} | "
                f"Error={type(error).__name__}: {error}"
            )

        next_send_time += interval

    send_loop_end = time.perf_counter()

    print()
    print("All messages submitted.")
    print("Waiting for Kafka acknowledgements...")

    successful = 0
    failed = 0
    latencies = []

    # Wait for every previously submitted message.
    # This does NOT slow down the send loop because all sends
    # were already submitted before we start waiting here.
    for future, send_start in zip(futures, send_times):

        try:
            future.get(timeout=60)

            delivery_end = time.perf_counter()

            latency_ms = (
                delivery_end - send_start
            ) * 1000

            latencies.append(latency_ms)
            successful += 1

        except Exception as error:

            failed += 1

            if failed <= 10:
                print(
                    f"DELIVERY FAILED | "
                    f"Error={type(error).__name__}: {error}"
                )

    delivery_end = time.perf_counter()

    producer.close()

    send_elapsed = send_loop_end - start_time
    delivery_elapsed = delivery_end - start_time

    send_rate = (
        len(futures) / send_elapsed
        if send_elapsed > 0
        else 0
    )

    delivery_rate = (
        successful / delivery_elapsed
        if delivery_elapsed > 0
        else 0
    )

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
    print("=" * 75)
    print("BENCHMARK RESULTS")
    print("=" * 75)

    print(f"Target rate         : {args.rate:.2f} tx/sec")
    print(f"Messages attempted  : {args.count}")
    print(f"Submitted           : {len(futures)}")
    print(f"Successful          : {successful}")
    print(f"Failed              : {failed}")

    print()
    print(f"Send-loop time      : {send_elapsed:.3f} sec")
    print(f"Total delivery time : {delivery_elapsed:.3f} sec")

    print()
    print(f"Send-loop rate      : {send_rate:.2f} tx/sec")
    print(f"Delivery throughput : {delivery_rate:.2f} tx/sec")

    print()
    print("LATENCY")
    print(f"Average latency     : {average_latency:.3f} ms")
    print(f"Median latency      : {median_latency:.3f} ms")
    print(f"P95 latency         : {p95_latency:.3f} ms")
    print(f"P99 latency         : {p99_latency:.3f} ms")

    print("=" * 75)

    if successful == args.count and failed == 0:
        print("STATUS: SUCCESS - all messages acknowledged by Kafka")
    else:
        print("STATUS: WARNING - not all messages were acknowledged")


if __name__ == "__main__":
    main()
