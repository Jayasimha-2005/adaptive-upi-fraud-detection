import json
import time
import argparse
from kafka import KafkaProducer


def create_transaction(number):
    return {
        "transaction_id": f"PART-{number:04d}",
        "card_id": f"CARD-{number % 6}",
        "amount": round(100 + number * 10.50, 2),
        "merchant_id": f"M{100 + (number % 10)}",
        "timestamp": time.time(),
        "device_type": ["mobile", "web", "pos"][number % 3],
        "country": ["IN", "US", "UK", "AE"][number % 4],
        "sequence": number
    }


def main():

    parser = argparse.ArgumentParser(
        description="Kafka partition distribution test producer"
    )

    parser.add_argument(
        "--topic",
        required=True
    )

    parser.add_argument(
        "--count",
        type=int,
        default=60
    )

    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda value:
            json.dumps(value).encode("utf-8"),
        acks="all"
    )

    print("=" * 60)
    print("KAFKA PARTITION DISTRIBUTION TEST")
    print("=" * 60)
    print(f"Topic    : {args.topic}")
    print(f"Messages : {args.count}")
    print("=" * 60)
    print()

    for i in range(1, args.count + 1):

        transaction = create_transaction(i)

        # card_id is used as the Kafka message key.
        key = transaction["card_id"].encode("utf-8")

        future = producer.send(
            args.topic,
            key=key,
            value=transaction
        )

        metadata = future.get(timeout=10)

        print(
            f"Transaction: {transaction['transaction_id']} | "
            f"Key: {transaction['card_id']} | "
            f"Partition: {metadata.partition} | "
            f"Offset: {metadata.offset}"
        )

    producer.flush()
    producer.close()

    print()
    print("=" * 60)
    print("PARTITION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
