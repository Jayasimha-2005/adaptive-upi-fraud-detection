import json
import time
import argparse
from kafka import KafkaProducer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--acks", required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=args.acks,
        retries=0,
        linger_ms=0
    )

    print("=" * 60)
    print("SIMPLE KAFKA ACK TEST")
    print("=" * 60)
    print(f"Topic    : {args.topic}")
    print(f"ACKS     : {args.acks}")
    print(f"Messages : {args.count}")
    print("=" * 60)

    start = time.perf_counter()

    for i in range(args.count):
        data = {
            "transaction_id": f"ACK-{args.acks}-{i+1:04d}",
            "amount": 100 + i,
            "timestamp": time.time()
        }

        producer.send(args.topic, value=data)

    print("Messages submitted. Waiting for Kafka...")

    producer.flush()

    elapsed = time.perf_counter() - start

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"ACKS       : {args.acks}")
    print(f"Messages   : {args.count}")
    print(f"Time       : {elapsed:.3f} seconds")
    print(f"Throughput : {args.count / elapsed:.2f} messages/sec")
    print("=" * 60)

    producer.close()


if __name__ == "__main__":
    main()
