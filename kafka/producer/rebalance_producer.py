import json
import time
from kafka import KafkaProducer

TOPIC = "rebalance_test_2"

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    acks="all"
)

print("=" * 60)
print("KAFKA CONSUMER REBALANCING TEST")
print("=" * 60)
print("Topic    : rebalance_test_2")
print("Messages : 120")
print("Rate     : 5 messages/second")
print("=" * 60)

for i in range(1, 121):

    partition = (i - 1) % 2

    transaction = {
        "transaction_id": f"REB-{i:04d}",
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

    time.sleep(0.2)

producer.flush()
producer.close()

print()
print("=" * 60)
print("PRODUCER COMPLETE")
print("=" * 60)
