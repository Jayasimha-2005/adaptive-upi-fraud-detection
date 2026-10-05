import json
import time

from kafka import KafkaProducer


BROKER = "localhost:9092"
TOPIC = "ordering_key_test"

TOTAL_MESSAGES = 60


def create_transaction(number, card_id):
    return {
        "transaction_id": f"ORDER-{number:04d}",
        "card_id": card_id,
        "amount": round(100 + number * 5.50, 2),
        "merchant_id": f"M{100 + (number % 10)}",
        "timestamp": time.time(),
        "sequence": number
    }


def main():

    producer = KafkaProducer(
        bootstrap_servers=[BROKER],

        value_serializer=lambda value:
            json.dumps(value).encode("utf-8"),

        key_serializer=lambda key:
            key.encode("utf-8"),

        acks="all",

        retries=5,

        enable_idempotence=True
    )

    print("=" * 75)
    print("KAFKA MESSAGE ORDERING & KEY PARTITIONING TEST")
    print("=" * 75)
    print(f"Broker          : {BROKER}")
    print(f"Topic           : {TOPIC}")
    print(f"Partitions      : 3")
    print(f"Messages        : {TOTAL_MESSAGES}")
    print(f"ACK mode        : all")
    print(f"Idempotence     : enabled")
    print("=" * 75)
    print()

    results = []

    card_ids = [
        "CARD-A",
        "CARD-B",
        "CARD-C",
        "CARD-D"
    ]

    for i in range(1, TOTAL_MESSAGES + 1):

        card_id = card_ids[(i - 1) % len(card_ids)]

        transaction = create_transaction(
            i,
            card_id
        )

        future = producer.send(
            TOPIC,
            key=card_id,
            value=transaction
        )

        metadata = future.get(
            timeout=10
        )

        results.append(
            (
                transaction["transaction_id"],
                card_id,
                metadata.partition,
                metadata.offset,
                transaction["sequence"]
            )
        )

        print(
            f"Transaction={transaction['transaction_id']} | "
            f"Key={card_id} | "
            f"Partition={metadata.partition} | "
            f"Offset={metadata.offset} | "
            f"Sequence={transaction['sequence']}"
        )

    producer.flush()
    producer.close()

    print()
    print("=" * 75)
    print("KEY → PARTITION SUMMARY")
    print("=" * 75)

    key_partitions = {}

    for (
        transaction_id,
        card_id,
        partition,
        offset,
        sequence
    ) in results:

        if card_id not in key_partitions:

            key_partitions[card_id] = set()

        key_partitions[card_id].add(
            partition
        )

    for card_id, partitions in key_partitions.items():

        print(
            f"{card_id} -> "
            f"Partitions {sorted(partitions)}"
        )

    print()
    print("=" * 75)
    print("ORDERING CHECK")
    print("=" * 75)

    ordering_passed = True

    for card_id in card_ids:

        card_records = [
            item
            for item in results
            if item[1] == card_id
        ]

        offsets = [
            item[3]
            for item in card_records
        ]

        sequences = [
            item[4]
            for item in card_records
        ]

        offsets_increasing = all(
            offsets[i] < offsets[i + 1]
            for i in range(len(offsets) - 1)
        )

        sequences_increasing = all(
            sequences[i] < sequences[i + 1]
            for i in range(len(sequences) - 1)
        )

        partitions = sorted(
            set(
                item[2]
                for item in card_records
            )
        )

        print(
            f"{card_id}: "
            f"Partitions={partitions} | "
            f"Offsets increasing={offsets_increasing} | "
            f"Sequences increasing={sequences_increasing}"
        )

        if not offsets_increasing:

            ordering_passed = False

        if not sequences_increasing:

            ordering_passed = False

    print()

    if ordering_passed:

        print(
            "RESULT: ORDERING CHECK PASSED"
        )

    else:

        print(
            "RESULT: ORDERING CHECK FAILED"
        )

    print("=" * 75)


if __name__ == "__main__":
    main()
