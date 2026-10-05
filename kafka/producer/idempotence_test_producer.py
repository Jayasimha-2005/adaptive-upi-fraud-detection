import json
import time
import argparse

from kafka import KafkaProducer
from kafka.errors import KafkaError


BROKER = "localhost:9092"
TOTAL_MESSAGES = 100


def create_transaction(number):
    return {
        "transaction_id": f"IDEMP-{number:04d}",
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
        description="Kafka idempotence comparison test"
    )

    parser.add_argument(
        "--topic",
        required=True
    )

    parser.add_argument(
        "--idempotence",
        choices=["on", "off"],
        required=True
    )

    args = parser.parse_args()

    idempotence_enabled = args.idempotence == "on"

    producer_config = {
        "bootstrap_servers": [BROKER],
        "value_serializer": lambda value: json.dumps(value).encode("utf-8"),
        "key_serializer": lambda key: key.encode("utf-8"),
        "acks": "all",
        "retries": 5,
        "retry_backoff_ms": 500
    }

    if idempotence_enabled:

        producer_config["enable_idempotence"] = True

    else:

        producer_config["enable_idempotence"] = False

    producer = KafkaProducer(**producer_config)

    successful = 0
    failed = 0

    print("=" * 70)
    print("KAFKA IDEMPOTENCE TEST")
    print("=" * 70)
    print(f"Broker          : {BROKER}")
    print(f"Topic           : {args.topic}")
    print(f"Idempotence     : {args.idempotence}")
    print(f"ACK mode        : all")
    print(f"Retries         : 5")
    print(f"Messages        : {TOTAL_MESSAGES}")
    print("=" * 70)
    print()

    for i in range(1, TOTAL_MESSAGES + 1):

        transaction = create_transaction(i)

        try:

            future = producer.send(
                args.topic,
                key=transaction["card_id"],
                value=transaction
            )

            metadata = future.get(timeout=10)

            successful += 1

            print(
                f"SUCCESS | "
                f"Transaction={transaction['transaction_id']} | "
                f"Partition={metadata.partition} | "
                f"Offset={metadata.offset}"
            )

        except KafkaError as error:

            failed += 1

            print(
                f"FAILED  | "
                f"Transaction={transaction['transaction_id']} | "
                f"Error={error}"
            )

    producer.flush()
    producer.close()

    print()
    print("=" * 70)
    print("IDEMPOTENCE TEST RESULTS")
    print("=" * 70)
    print(f"Topic              : {args.topic}")
    print(f"Idempotence        : {args.idempotence}")
    print(f"Messages attempted : {TOTAL_MESSAGES}")
    print(f"Successful         : {successful}")
    print(f"Failed             : {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
