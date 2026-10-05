import json
import time
from kafka import KafkaProducer

BROKER = "localhost:9092"
TOPIC = "acks_test"

print("Creating Kafka producer...")

producer = KafkaProducer(
    bootstrap_servers=[BROKER],
    acks=1,
    retries=0,
    linger_ms=0,
    request_timeout_ms=10000,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print("Sending one message...")

message = {
    "transaction_id": "TEST-ACKS-PYTHON-1",
    "amount": 500.00,
    "is_fraud": 0
}

start = time.perf_counter()

try:
    future = producer.send(TOPIC, value=message)

    metadata = future.get(timeout=10)

    latency = (time.perf_counter() - start) * 1000

    print()
    print("SUCCESS")
    print(f"Topic     : {metadata.topic}")
    print(f"Partition : {metadata.partition}")
    print(f"Offset    : {metadata.offset}")
    print(f"Latency   : {latency:.3f} ms")

except Exception as e:
    print()
    print("FAILED")
    print(type(e).__name__)
    print(e)

finally:
    producer.close()
