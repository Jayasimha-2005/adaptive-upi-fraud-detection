import json
import time
from kafka import KafkaProducer


BROKER = "localhost:9092"
TOPIC = "acks_test"


producer = KafkaProducer(
    bootstrap_servers=[BROKER],
    acks=1,
    retries=0,
    linger_ms=0,
    batch_size=16384,
    request_timeout_ms=5000,
    delivery_timeout_ms=7000,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)


print("=" * 60)
print("ACKS=1 CONFIGURATION TEST")
print("=" * 60)

for i in range(5):

    message = {
        "transaction_id": f"CONFIG-TEST-{i}",
        "amount": 500 + i,
        "timestamp": time.time(),
        "is_fraud": 0
    }

    start = time.perf_counter()

    try:

        future = producer.send(
            TOPIC,
            value=message
        )

        metadata = future.get(timeout=5)

        latency = (
            time.perf_counter() - start
        ) * 1000

        print(
            f"Message {i + 1}: "
            f"SUCCESS | "
            f"Partition={metadata.partition} | "
            f"Offset={metadata.offset} | "
            f"Latency={latency:.3f} ms"
        )

    except Exception as e:

        latency = (
            time.perf_counter() - start
        ) * 1000

        print(
            f"Message {i + 1}: "
            f"FAILED | "
            f"{type(e).__name__}: {e} | "
            f"Latency={latency:.3f} ms"
        )


producer.flush()
producer.close()

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
