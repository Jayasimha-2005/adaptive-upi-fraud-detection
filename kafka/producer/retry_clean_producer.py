import argparse
import json
import time

from kafka import KafkaProducer
from kafka.errors import KafkaError


BROKER = "localhost:9092"
TOPIC = "retry_clean_test"
TOTAL_MESSAGES = 200
INTERVAL = 0.5


def create_transaction(number):
    return {
        "transaction_id": f"RETRY-CLEAN-{number:04d}",
        "card_id": f"CARD-{number % 5}",
        "amount": round(100 + number * 6.25, 2),
        "merchant_id": f"M{300 + (number % 10)}",
        "timestamp": time.time(),
        "sequence": number
    }


def main():
    parser = argparse.ArgumentParser(
        description="Controlled Kafka retry behavior experiment"
    )

    parser.add_argument(
        "--retries",
        type=int,
        choices=[0, 5],
        required=True,
        help="Number of producer retries"
    )

    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=[BROKER],
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),

        # Keep these identical in BOTH experiments
        acks="all",
        enable_idempotence=False,

        # Only this parameter changes
        retries=args.retries,

        retry_backoff_ms=500,
        request_timeout_ms=3000,
        delivery_timeout_ms=10000
    )

    successful = 0
    failed = 0

    print("=" * 75)
    print("CONTROLLED KAFKA RETRY BEHAVIOR TEST")
    print("=" * 75)
    print(f"Broker              : {BROKER}")
    print(f"Topic               : {TOPIC}")
    print(f"Retries             : {args.retries}")
    print(f"Idempotence         : False")
    print(f"ACK mode            : all")
    print(f"Total messages      : {TOTAL_MESSAGES}")
    print(f"Message interval    : {INTERVAL} seconds")
    print("=" * 75)
    print()

    for i in range(1, TOTAL_MESSAGES + 1):

        transaction = create_transaction(i)

        try:
            future = producer.send(
                TOPIC,
                key=transaction["card_id"],
                value=transaction
            )

            metadata = future.get(timeout=10)

            successful += 1

            print(
                f"SUCCESS | {transaction['transaction_id']} "
                f"| Partition={metadata.partition} "
                f"| Offset={metadata.offset}"
            )

        except KafkaError as error:

            failed += 1

            print(
                f"FAILED  | {transaction['transaction_id']} "
                f"| Error={type(error).__name__}: {error}"
            )

        except Exception as error:

            failed += 1

            print(
                f"FAILED  | {transaction['transaction_id']} "
                f"| Error={type(error).__name__}: {error}"
            )

        time.sleep(INTERVAL)

    print()
    print("Flushing producer...")

    try:
        producer.flush()
    except Exception as error:
        print(
            f"Flush error: {type(error).__name__}: {error}"
        )

    producer.close()

    print()
    print("=" * 75)
    print("CONTROLLED RETRY TEST RESULTS")
    print("=" * 75)
    print(f"Retries             : {args.retries}")
    print(f"Idempotence         : False")
    print(f"Messages attempted  : {TOTAL_MESSAGES}")
    print(f"Successful          : {successful}")
    print(f"Failed              : {failed}")
    print("=" * 75)


if __name__ == "__main__":
    main()
