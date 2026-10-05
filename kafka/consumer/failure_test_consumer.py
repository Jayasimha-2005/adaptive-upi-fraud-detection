import argparse
import json
import time

from kafka import KafkaConsumer

BROKER = "localhost:9092"


def main():
    parser = argparse.ArgumentParser(
        description="Kafka consumer failure and recovery test"
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Kafka topic"
    )

    parser.add_argument(
        "--group",
        required=True,
        help="Kafka consumer group"
    )

    parser.add_argument(
        "--consumer-id",
        required=True,
        help="Consumer ID"
    )

    parser.add_argument(
        "--crash-after",
        type=int,
        default=1,
        help="Number of messages to process before intentional crash"
    )

    args = parser.parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=[BROKER],
        group_id=args.group,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    print("=" * 70)
    print("KAFKA CONSUMER FAILURE TEST")
    print("=" * 70)
    print(f"Broker      : {BROKER}")
    print(f"Topic       : {args.topic}")
    print(f"Group       : {args.group}")
    print(f"Consumer ID : {args.consumer_id}")
    print(f"Crash after : {args.crash_after} message(s)")
    print("=" * 70)
    print()
    print("Waiting for a transaction...")
    print()

    processed = 0

    try:
        for message in consumer:
            transaction = message.value

            transaction_id = transaction.get(
                "transaction_id",
                "UNKNOWN"
            )

            print(
                f"Received: {transaction_id} | "
                f"Partition: {message.partition} | "
                f"Offset: {message.offset}"
            )

            print("Processing transaction...")
            time.sleep(1)

            print("Transaction processed.")

            processed += 1

            if processed >= args.crash_after:
                print()
                print("CRASHING BEFORE OFFSET COMMIT!")
                print(
                    f"Offset {message.offset} was NOT committed."
                )
                print(
                    "The next consumer in the same group "
                    "should replay this message."
                )

                consumer.close()
                return

            consumer.commit()

            print(
                f"Offset {message.offset} committed successfully."
            )
            print()

    except KeyboardInterrupt:
        print()
        print("Consumer stopped manually.")

    finally:
        consumer.close()


if __name__ == "__main__":
    main()
