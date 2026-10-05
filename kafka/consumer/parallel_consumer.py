import json
import argparse
import time
from kafka import KafkaConsumer


def main():
    parser = argparse.ArgumentParser(
        description="Kafka consumer parallelism experiment"
    )

    parser.add_argument("--topic", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--consumer-id", required=True)

    args = parser.parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers="localhost:9092",
        group_id=args.group,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        consumer_timeout_ms=60000
    )

    print("=" * 60)
    print("KAFKA CONSUMER PARALLELISM TEST")
    print("=" * 60)
    print(f"Consumer ID : {args.consumer_id}")
    print(f"Topic       : {args.topic}")
    print(f"Group       : {args.group}")
    print("=" * 60)
    print()

    received = 0
    partitions_used = set()

    try:
        for message in consumer:

            transaction = message.value

            partitions_used.add(message.partition)

            print(
                f"Consumer {args.consumer_id} | "
                f"Partition: {message.partition} | "
                f"Offset: {message.offset} | "
                f"Transaction: {transaction['transaction_id']}"
            )

            consumer.commit()

            received += 1

    except Exception as e:
        print(f"Consumer {args.consumer_id} stopped: {e}")

    finally:
        consumer.close()

    print()
    print("=" * 60)
    print(f"Consumer {args.consumer_id} received : {received}")
    print(f"Partitions handled                    : {sorted(partitions_used)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
