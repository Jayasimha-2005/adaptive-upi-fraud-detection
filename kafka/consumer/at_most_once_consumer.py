import json
import os
import argparse

from kafka import KafkaConsumer


BROKER = "localhost:9092"
TOPIC = "at_most_once_test"
GROUP_ID = "at_most_once_group"


def create_consumer():
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=[BROKER],
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        key_deserializer=lambda key: key.decode("utf-8") if key else None,
        consumer_timeout_ms=5000
    )


def main():
    parser = argparse.ArgumentParser(
        description="Kafka at-most-once delivery test"
    )

    parser.add_argument(
        "--mode",
        choices=["crash", "recover"],
        required=True,
        help="crash = commit before processing and stop; recover = check for replay"
    )

    args = parser.parse_args()

    consumer = create_consumer()

    print("=" * 75)
    print("KAFKA AT-MOST-ONCE DELIVERY TEST")
    print("=" * 75)
    print(f"Broker          : {BROKER}")
    print(f"Topic           : {TOPIC}")
    print(f"Consumer group  : {GROUP_ID}")
    print(f"Mode            : {args.mode}")
    print("Auto commit     : disabled")
    print("=" * 75)
    print()

    if args.mode == "crash":

        print("CRASH MODE")
        print("The first message will be committed BEFORE processing.")
        print("Then the consumer will intentionally stop.")
        print()

        for message in consumer:

            transaction = message.value
            sequence = transaction["sequence"]

            print(
                f"RECEIVED | "
                f"Transaction={transaction['transaction_id']} | "
                f"Partition={message.partition} | "
                f"Offset={message.offset} | "
                f"Sequence={sequence}"
            )

            if sequence == 1:

                # Commit BEFORE processing.
                consumer.commit()

                print()
                print("=" * 75)
                print("OFFSET COMMITTED BEFORE PROCESSING")
                print("=" * 75)
                print(
                    f"Transaction={transaction['transaction_id']} | "
                    f"Offset={message.offset}"
                )
                print(
                    "The offset has been committed, but the transaction "
                    "has NOT been processed."
                )
                print()
                print("Simulating consumer failure now...")
                print("=" * 75)

                consumer.close()

                # Simulate a crash before business processing.
                os._exit(1)

            else:
                print(
                    f"PROCESSING | "
                    f"Transaction={transaction['transaction_id']}"
                )
                consumer.commit()

        consumer.close()

    elif args.mode == "recover":

        print("RECOVERY MODE")
        print("Restarting with the SAME consumer group.")
        print("The previously committed message should NOT be delivered again.")
        print()

        first_message_seen = False
        replay_detected = False
        messages_processed = 0

        for message in consumer:

            first_message_seen = True

            transaction = message.value
            sequence = transaction["sequence"]

            print(
                f"RECEIVED | "
                f"Transaction={transaction['transaction_id']} | "
                f"Partition={message.partition} | "
                f"Offset={message.offset} | "
                f"Sequence={sequence}"
            )

            if sequence == 1:
                replay_detected = True

                print()
                print("=" * 75)
                print("UNEXPECTED REPLAY")
                print("=" * 75)
                print(
                    "The committed message was delivered again."
                )
                print("=" * 75)

            print(
                f"PROCESSING | "
                f"Transaction={transaction['transaction_id']}"
            )

            consumer.commit()
            messages_processed += 1

            if sequence == 10:
                break

        print()
        print("=" * 75)
        print("AT-MOST-ONCE TEST RESULT")
        print("=" * 75)

        if replay_detected:
            print("RESULT: REPLAY DETECTED")
            print(
                "The previously committed message was delivered again."
            )
        else:
            print("RESULT: NO REPLAY OF THE COMMITTED MESSAGE")
            print(
                "The committed message was skipped after the consumer restart."
            )
            print(
                "This demonstrates the at-most-once trade-off: "
                "a message can be lost if failure occurs after commit "
                "but before processing."
            )

        print(f"Messages received after restart: {messages_processed}")
        print("=" * 75)

    consumer.close()


if __name__ == "__main__":
    main()
