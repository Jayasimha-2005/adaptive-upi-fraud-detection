"""Kafka transaction producer for local demonstrations.

If kafka-python is not installed, use --print-only to inspect payloads without
requiring Kafka.
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone, timedelta


def make_event(i: int, card_id="CARD123", event_time=None) -> dict:
    return {
        "transaction_id": f"TXN{i:06d}",
        "card_id": card_id,
        "merchant_id": f"M{(i % 7) + 1:03d}",
        "amount": round(random.uniform(100, 5000), 2),
        "event_time": (event_time or datetime.now(timezone.utc)).isoformat(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--topic", default="fraud-transactions")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--interval", type=float, default=0.25)
    p.add_argument("--print-only", action="store_true")
    a = p.parse_args()

    producer = None
    if not a.print_only:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=a.bootstrap,
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
        )

    base = datetime.now(timezone.utc)
    for i in range(a.count):
        event = make_event(i, event_time=base + timedelta(seconds=i))
        if producer:
            producer.send(a.topic, event)
        else:
            print(json.dumps(event))
        time.sleep(a.interval)

    if producer:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
