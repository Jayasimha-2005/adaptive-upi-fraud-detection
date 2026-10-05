import json
import os
import time
import argparse

from kafka import KafkaConsumer


BROKER = "localhost:9092"
TOPIC = "delivery_semantics_test"
GROUP_ID = "delivery_semantics_group"


def main():
    parser = argparse.ArgumentParser(
        description="Kafka at-least-once delivery and offset replay test"
    )

    parser.add_argument(
        "--mode",
        choices=["crash", "recover"],
        required=True,
        help="crash = process message 5 without committing; recover = restart and observe replay"
    )

    args = parser.parse_args()

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[BROKER],
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        key_deserializer=lambda key: key.decode("utf-8") if key else None,
        consumer_timeout_ms=5000
    )

    print("=" * 75)
    print("KAFKA AT-LEAST-ONCE DELIVERY TEST")
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
        print("Messages 1-4 will be processed and committed.")
        print("Message 5 will be processed but NOT committed.")
        print("Then the consumer will intentionally stop.")
        print()

        processed_count = 0

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

            processed_count += 1

            if sequence <= 4:
                consumer.commit()

                print(
                    f"COMMITTED | "
                    f"Transaction={transaction['transaction_id']} | "
                    f"Committed offset={message.offset + 1}"
                )

            elif sequence == 5:
                print()
                print("=" * 75)
                print("INTENTIONAL FAILURE")
                print("=" * 75)
                print(
                    f"PROCESSED BUT NOT COMMITTED | "
                    f"Transaction={transaction['transaction_id']} | "
                    f"Offset={message.offset}"
                )
                print()
                print("The consumer will now stop WITHOUT committing this offset.")
                print("When restarted with the same consumer group, Kafka should replay this message.")
                print("=" * 75)

                consumer.close()
                os._exit(1)

            else:
                print(
                    f"STOPPING BEFORE PROCESSING FURTHER RECORDS | "
                    f"Transaction={transaction['transaction_id']}"
                )
                break

    elif args.mode == "recover":
        print("RECOVERY MODE")
        print("The consumer is restarting with the SAME consumer group.")
        print("The uncommitted message should be delivered again.")
        print()

        replay_detected = False

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

            if sequence == 5:
                replay_detected = True

                print()
                print("=" * 75)
                print("REPLAY DETECTED")
                print("=" * 75)
                print(
                    f"Transaction={transaction['transaction_id']} "
                    f"was delivered again."
                )
                print(
                    f"Offset={message.offset} was not committed before the previous consumer stopped."
                )
                print("This demonstrates at-least-once delivery behavior.")
                print("=" * 75)
                print()

            consumer.commit()

            print(
                f"COMMITTED | "
                f"Transaction={transaction['transaction_id']} | "
                f"Committed offset={message.offset + 1}"
            )

            if sequence == 10:
                break

        print()
        print("=" * 75)
        print("AT-LEAST-ONCE TEST RESULT")
        print("=" * 75)

        if replay_detected:
            print("RESULT: PASS")
            print("Evidence: The uncommitted message was replayed after consumer restart.")
        else:
            print("RESULT: REPLAY NOT DETECTED")

        print("=" * 75)

    consumer.close()


if __name__ == "__main__":
    main()
