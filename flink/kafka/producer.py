import json
from pathlib import Path
from kafka import KafkaProducer

def main():
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda value: json.dumps(value).encode()
    )
    events = json.loads(
        Path(__file__).with_name("sample_transactions.json")
        .read_text(encoding="utf-8")
    )
    for event in events:
        producer.send("fraud-transactions", event)
        print("Sent:", event)
    producer.flush()

if __name__ == "__main__":
    main()
