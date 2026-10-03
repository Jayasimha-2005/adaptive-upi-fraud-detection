"""Pure-Python reference implementation of the real-time feature semantics.

Used for deterministic tests and demonstrations; production streaming uses
Spark Structured Streaming window aggregations.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable


@dataclass(frozen=True)
class Event:
    card_id: str
    merchant_id: str
    amount: float
    event_seconds: float


def calculate_features(events: Iterable[Event], now_seconds: float) -> dict:
    events = list(events)
    five = [e for e in events if now_seconds - 300 <= e.event_seconds <= now_seconds]
    ten = [e for e in events if now_seconds - 600 <= e.event_seconds <= now_seconds]
    previous = sorted((e for e in events if e.event_seconds <= now_seconds),
                      key=lambda x: x.event_seconds)
    delta = None
    if len(previous) >= 2:
        delta = previous[-1].event_seconds - previous[-2].event_seconds
    return {
        "transaction_count_5m": len(five),
        "transaction_amount_5m": sum(e.amount for e in five),
        "transaction_count_10m": len(ten),
        "transaction_amount_10m": sum(e.amount for e in ten),
        "unique_merchants_10m": len({e.merchant_id for e in ten if e.merchant_id}),
        "time_since_previous_transaction": delta,
    }
