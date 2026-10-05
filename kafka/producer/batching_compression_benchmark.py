import json
import time
import argparse
import statistics

from kafka import KafkaProducer


BROKER = "localhost:9092"

TOTAL_MESSAGES = 2000
TARGET_RATE = 100

TOPIC_DEFAULT = "batching_compression_test"


def create_transaction(number):
    return {
        "transaction_id": f"BATCH-{number:05d}",
        "card_id": f"CARD-{number % 20}",
        "amount": round(100 + number * 3.75, 2),
        "merchant_id": f"M{100 + (number % 20)}",
        "timestamp": time.time(),
        "device_type": ["mobile", "web", "pos"][number % 3],
        "country": ["IN", "US", "UK", "AE"][number % 4],
        "sequence": number
    }


def percentile(values, percentage):

    if not values:
        return 0.0

    sorted_values = sorted(values)

    index = int(
        (percentage / 100) * len(sorted_values)
    )

    if index >= len(sorted_values):
        index = len(sorted_values) - 1

    return sorted_values[index]


def main():

    parser = argparse.ArgumentParser(
        description="Kafka batching and compression benchmark"
    )

    parser.add_argument(
        "--topic",
        default=TOPIC_DEFAULT
    )

    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "batching",
            "compression"
        ],
        required=True
    )

    args = parser.parse_args()

    producer_config = {
        "bootstrap_servers": [BROKER],

        "value_serializer":
            lambda value: json.dumps(value).encode("utf-8"),

        "key_serializer":
            lambda key: key.encode("utf-8"),

        "acks": "all",

        "retries": 5,

        "enable_idempotence": True
    }

    if args.mode == "baseline":

        producer_config["linger_ms"] = 0

        producer_config["compression_type"] = None

    elif args.mode == "batching":

        producer_config["linger_ms"] = 10

        producer_config["batch_size"] = 32768

        producer_config["compression_type"] = None

    elif args.mode == "compression":

        producer_config["linger_ms"] = 10

        producer_config["batch_size"] = 32768

        producer_config["compression_type"] = "gzip"

    producer = KafkaProducer(
        **producer_config
    )

    latencies = []

    successful = 0
    failed = 0

    interval = 1 / TARGET_RATE

    print("=" * 70)
    print("KAFKA BATCHING & COMPRESSION BENCHMARK")
    print("=" * 70)

    print(f"Broker          : {BROKER}")
    print(f"Topic           : {args.topic}")
    print(f"Mode            : {args.mode}")
    print(f"Messages        : {TOTAL_MESSAGES}")
    print(f"Target rate     : {TARGET_RATE} tx/sec")
    print(f"ACK mode        : all")
    print(f"Idempotence     : enabled")

    print()

    print(
        f"Linger ms       : "
        f"{producer_config.get('linger_ms')}"
    )

    print(
        f"Batch size      : "
        f"{producer_config.get('batch_size', 'default')}"
    )

    print(
        f"Compression     : "
        f"{producer_config.get('compression_type')}"
    )

    print("=" * 70)
    print()

    start_time = time.time()

    next_send_time = start_time

    for i in range(1, TOTAL_MESSAGES + 1):

        now = time.time()

        if now < next_send_time:

            time.sleep(
                next_send_time - now
            )

        transaction = create_transaction(i)

        send_start = time.perf_counter()

        try:

            future = producer.send(
                args.topic,
                key=transaction["card_id"],
                value=transaction
            )

            future.get(timeout=10)

            send_end = time.perf_counter()

            latency_ms = (
                send_end - send_start
            ) * 1000

            latencies.append(
                latency_ms
            )

            successful += 1

        except Exception as error:

            failed += 1

            print(
                f"FAILED | "
                f"Transaction={transaction['transaction_id']} | "
                f"Error={type(error).__name__}: {error}"
            )

        next_send_time += interval

    producer.flush()
    producer.close()

    end_time = time.time()

    duration = end_time - start_time

    producer_rate = (
        successful / duration
        if duration > 0
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

    p95_latency = percentile(
        latencies,
        95
    )

    p99_latency = percentile(
        latencies,
        99
    )

    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"Mode              : {args.mode}")
    print(f"Messages attempted: {TOTAL_MESSAGES}")
    print(f"Successful        : {successful}")
    print(f"Failed            : {failed}")
    print(f"Duration          : {duration:.3f} sec")
    print(f"Producer rate     : {producer_rate:.2f} tx/sec")

    print(
        f"Average latency   : "
        f"{average_latency:.3f} ms"
    )

    print(
        f"Median latency    : "
        f"{median_latency:.3f} ms"
    )

    print(
        f"P95 latency       : "
        f"{p95_latency:.3f} ms"
    )

    print(
        f"P99 latency       : "
        f"{p99_latency:.3f} ms"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
