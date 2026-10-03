import json
from kafka import KafkaConsumer

def main():
    consumer = KafkaConsumer(
        "fraud-predictions",
        bootstrap_servers="localhost:9092",
        auto_offset_reset="earliest",
        value_deserializer=lambda x: json.loads(x.decode())
    )
    for message in consumer:
        print("Prediction:", message.value)

if __name__ == "__main__":
    main()
