import argparse
import time

from kafka import KafkaConsumer


BROKER = "localhost:9092"
TOPIC = "consumer_parallel_3"


def main():

    parser = argparse.ArgumentParser(
        description="Kafka consumer group scaling and partition assignment test"
    )

    parser.add_argument(
        "--group",
        required=True,
        help="Kafka consumer group ID"
    )

    parser.add_argument(
        "--consumer-id",
        required=True,
        help="Unique consumer ID"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="How long the consumer should run in seconds"
    )

    args = parser.parse_args()

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[BROKER],
        group_id=args.group,
        client_id=args.consumer_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=1000,
    )

    print("=" * 70)
    print("KAFKA CONSUMER GROUP SCALING TEST")
    print("=" * 70)
    print(f"Broker          : {BROKER}")
    print(f"Topic           : {TOPIC}")
    print(f"Consumer group  : {args.group}")
    print(f"Consumer ID     : {args.consumer_id}")
    print(f"Duration        : {args.duration} seconds")
    print("=" * 70)
    print()

    start_time = time.time()

    message_count = 0

    partitions_seen = set()

    last_assignment = None

    print("Waiting for partition assignment...")

    while time.time() - start_time < args.duration:

        records = consumer.poll(
            timeout_ms=1000
        )

        current_assignment = sorted(
            partition.partition
            for partition in consumer.assignment()
        )

        if current_assignment != last_assignment:

            print(
                f"[{args.consumer_id}] "
                f"Partition assignment changed: "
                f"{current_assignment}"
            )

            last_assignment = current_assignment

        for topic_partition, messages in records.items():

            for message in messages:

                message_count += 1

                partitions_seen.add(
                    message.partition
                )

                print(
                    f"[{args.consumer_id}] "
                    f"Partition={message.partition} "
                    f"Offset={message.offset} "
                    f"Transaction={message.value}"
                )

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
    print(f"Consumer ID         : {args.consumer_id}")
    print(f"Consumer group      : {args.group}")
    print(f"Messages consumed   : {message_count}")
    print(f"Partitions observed : {sorted(partitions_seen)}")
    print(f"Final assignment    : {final_assignment}")
    print("=" * 70)


if __name__ == "__main__":
    main()
