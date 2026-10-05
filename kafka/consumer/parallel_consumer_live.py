import argparse
import json
import time

from kafka import KafkaConsumer


BROKER = "localhost:9092"


def main():
    parser = argparse.ArgumentParser(
        description="Live Kafka consumer parallelism experiment"
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Kafka topic",
    )

    parser.add_argument(
        "--group",
        required=True,
        help="Kafka consumer group",
    )

    parser.add_argument(
        "--consumer-id",
        required=True,
        help="Unique consumer ID",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="How long the consumer should stay alive",
    )

    args = parser.parse_args()

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=[BROKER],
        group_id=args.group,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )

    print("=" * 70)
    print("LIVE KAFKA CONSUMER PARALLELISM TEST")
    print("=" * 70)
    print(f"Consumer ID : {args.consumer_id}")
    print(f"Topic       : {args.topic}")
    print(f"Group       : {args.group}")
    print(f"Duration    : {args.duration} seconds")
    print("=" * 70)
    print()

    start_time = time.time()

    received = 0
    partitions_used = set()

    last_assignment = None

    print("Waiting for partition assignment...")

    try:

        while time.time() - start_time < args.duration:

            records = consumer.poll(timeout_ms=1000)

            current_assignment = sorted(
                partition.partition
                for partition in consumer.assignment()
            )

            if current_assignment != last_assignment:
                print(
                    f"Consumer {args.consumer_id} assignment: "
                    f"{current_assignment}"
                )

                last_assignment = current_assignment

            for topic_partition, messages in records.items():

                for message in messages:

                    received += 1

                    partitions_used.add(
                        message.partition
                    )

                    transaction_id = message.value.get(
                        "transaction_id",
                        "UNKNOWN",
                    )

                    print(
                        f"Consumer {args.consumer_id} | "
                        f"Partition={message.partition} | "
                        f"Offset={message.offset} | "
                        f"Transaction={transaction_id}"
                    )

                    consumer.commit()

    except KeyboardInterrupt:

        print()
        print("Consumer stopped manually.")

    finally:

        final_assignment = sorted(
            partition.partition
            for partition in consumer.assignment()
        )

        consumer.close(
            autocommit=False
        )

    print()
    print("=" * 70)
    print("CONSUMER RESULTS")
    print("=" * 70)
    print(f"Consumer ID          : {args.consumer_id}")
    print(f"Messages received    : {received}")
    print(f"Partitions handled   : {sorted(partitions_used)}")
    print(f"Final assignment     : {final_assignment}")
    print("=" * 70)


if __name__ == "__main__":
    main()
