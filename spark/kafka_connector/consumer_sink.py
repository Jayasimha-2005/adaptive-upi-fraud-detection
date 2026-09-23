"""Small Kafka sink helper for feature payloads."""

import json


def create_producer(bootstrap_servers: str):
    from kafka import KafkaProducer
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda x: json.dumps(x, default=str).encode("utf-8"),
    )
