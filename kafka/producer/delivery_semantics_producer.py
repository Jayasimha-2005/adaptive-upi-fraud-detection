import json
import time
from kafka import KafkaProducer

BROKER = "localhost:9092"
TOPIC = "delivery_semantics_test"
TOTAL_MESSAGES = 10


def create_transaction(number):
    return {
        "transaction_id": f"DELIVERY-{number:03d}",
        "card_id": f"CARD-{number % 3}",
        "amount": round(100 + number * 10.50, 2),
        "merchant_id": f"M{100 + number}",
        "timestamp": time.time(),
        "sequence": number
    }


def main():
    producer = KafkaProducer(
        bootstrap_servers=[BROKER],
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),
        acks="all",
        retries=5,
        enable_idempotence=True
    )

    print("=" * 70)
    print("DELIVERY SEMANTICS TEST - PRODUCER")
    print("=" * 70)
    print(f"Broker          : {BROKER}")
    print(f"Topic           : {TOPIC}")
    print(f"Messages        : {TOTAL_MESSAGES}")
    print("ACK mode        : all")
    print("Idempotence     : enabled")
    print("=" * 70)
    print()

    successful = 0
    failed = 0

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
                f"SUCCESS | "
                f"Transaction={transaction['transaction_id']} | "
                f"Partition={metadata.partition} | "
                f"Offset={metadata.offset}"
            )

        except Exception as error:
            failed += 1
            print(
                f"FAILED  | "
                f"Transaction={transaction['transaction_id']} | "
                f"Error={type(error).__name__}: {error}"
            )

    producer.flush()
    producer.close()

    print()
    print("=" * 70)
    print("PRODUCER RESULTS")
    print("=" * 70)
    print(f"Messages attempted : {TOTAL_MESSAGES}")
    print(f"Successful         : {successful}")
    print(f"Failed             : {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
