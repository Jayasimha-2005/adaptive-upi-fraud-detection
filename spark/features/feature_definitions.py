"""Canonical feature names shared by batch and streaming paths."""

REALTIME_FEATURES = [
    "transaction_count_5m",
    "transaction_amount_5m",
    "transaction_count_10m",
    "transaction_amount_10m",
    "unique_merchants_10m",
    "time_since_previous_transaction",
]

BASE_STREAM_FIELDS = [
    "transaction_id", "card_id", "merchant_id", "amount", "event_time"
]
