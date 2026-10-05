import json
import time
import argparse

from kafka import KafkaProducer
from kafka.errors import KafkaError


BROKER = "localhost:9092"

TOTAL_MESSAGES = 200
INTERVAL = 0.5

START_NUMBER = 1


def create_transaction(number):
    return {
        "transaction_id": f"IDEMP-FAIL-{number:04d}",
        "card_id": f"CARD-{number % 10}",
        "amount": round(100 + number * 7.25, 2),
        "merchant_id": f"M{100 + (number % 10)}",
        "timestamp": time.time(),
        "device_type": ["mobile", "web", "pos"][number % 3],
        "country": ["IN", "US", "UK", "AE"][number % 4],
        "sequence": number
    }


def main():

    parser = argparse.ArgumentParser(
        description="Kafka idempotence failure and retry test"
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Kafka topic to test"
    )

    parser.add_argument(
        "--idempotence",
        choices=["on", "off"],
        required=True,
        help="Enable or disable Kafka producer idempotence"
    )

    args = parser.parse_args()

    idempotence_enabled = args.idempotence == "on"

    producer_config = {
        "bootstrap_servers": [BROKER],

        "value_serializer":
            lambda value: json.dumps(value).encode("utf-8"),

        "key_serializer":
            lambda key: key.encode("utf-8"),

        "acks": "all",

        "retries": 5,

        "retry_backoff_ms": 500,

        "request_timeout_ms": 3000,

        "delivery_timeout_ms": 10000
    }

    if idempotence_enabled:

        producer_config["enable_idempotence"] = True

    else:

        producer_config["enable_idempotence"] = False

    producer = KafkaProducer(**producer_config)

    successful = 0
    failed = 0

    successful_transactions = []
    failed_transactions = []

    print("=" * 75)
    print("KAFKA IDEMPOTENCE FAILURE / RETRY TEST")
    print("=" * 75)

    print(f"Broker              : {BROKER}")
    print(f"Topic               : {args.topic}")
    print(f"Idempotence         : {args.idempotence}")
    print(f"ACK mode            : all")
    print(f"Retries             : 5")
    print(f"Total messages      : {TOTAL_MESSAGES}")
    print(f"Message interval    : {INTERVAL} seconds")

    print("=" * 75)
    print()

    print("IMPORTANT:")
    print("Stop and restart the Kafka broker while this program is running.")
    print("The program will continue attempting to send transactions.")
    print()

    for i in range(
        START_NUMBER,
        START_NUMBER + TOTAL_MESSAGES
    ):

        transaction = create_transaction(i)

        transaction_id = transaction["transaction_id"]

        try:

            future = producer.send(
                args.topic,
                key=transaction["card_id"],
                value=transaction
            )

            metadata = future.get(timeout=10)

            successful += 1

            successful_transactions.append(
                transaction_id
            )

            print(
                f"SUCCESS | "
                f"{transaction_id} | "
                f"Partition={metadata.partition} | "
                f"Offset={metadata.offset}"
            )

        except KafkaError as error:

            failed += 1

            failed_transactions.append(
                transaction_id
            )

            print(
                f"FAILED  | "
                f"{transaction_id} | "
                f"Error={type(error).__name__}: {error}"
            )

        except Exception as error:

            failed += 1

            failed_transactions.append(
                transaction_id
            )

            print(
                f"FAILED  | "
                f"{transaction_id} | "
                f"Error={type(error).__name__}: {error}"
            )

        time.sleep(INTERVAL)

    print()
    print("Flushing producer...")

    try:

        producer.flush()

    except Exception as error:

        print(
            f"Flush error: "
            f"{type(error).__name__}: {error}"
        )

    producer.close()

    print()
    print("=" * 75)
    print("IDEMPOTENCE FAILURE TEST RESULTS")
    print("=" * 75)

    print(f"Topic                  : {args.topic}")
    print(f"Idempotence            : {args.idempotence}")
    print(f"Messages attempted     : {TOTAL_MESSAGES}")
    print(f"Successful             : {successful}")
    print(f"Failed                 : {failed}")

    print()

    print(
        f"Successful transaction IDs : "
        f"{len(successful_transactions)}"
    )

    print(
        f"Failed transaction IDs     : "
        f"{len(failed_transactions)}"
    )

    print()

    if failed_transactions:

        print("Failed transactions:")

        for transaction_id in failed_transactions:

            print(
                f"  {transaction_id}"
            )

    print("=" * 75)


if __name__ == "__main__":
    main()
