import json
import time
from kafka import KafkaProducer

TOPIC = "parallel_clean_3"

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    acks="all"
)

print("=" * 60)
print("CONTROLLED PARALLEL CONSUMER TEST")
print("=" * 60)
print(f"Topic    : {TOPIC}")
print("Messages : 60")
print("Distribution: 20 messages per partition")
print("=" * 60)

for i in range(1, 61):

    partition = (i - 1) % 3

    transaction = {
        "transaction_id": f"PAR-{i:04d}",
        "amount": 100 + i,
        "timestamp": time.time(),
        "partition_test": True
    }

    future = producer.send(
        TOPIC,
        partition=partition,
        value=transaction
    )

    metadata = future.get(timeout=10)

    print(
        f"Transaction: {transaction['transaction_id']} | "
        f"Partition: {metadata.partition} | "
        f"Offset: {metadata.offset}"
    )

producer.flush()
producer.close()

print()
print("=" * 60)
print("PRODUCER COMPLETE")
print("=" * 60)
