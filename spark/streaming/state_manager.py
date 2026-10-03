"""Stateful helper for inter-arrival time.

This file contains the deterministic state logic used by tests/simulations.
Production Spark deployments can use applyInPandasWithState on Spark versions
that support it; keeping the logic isolated makes version differences explicit.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CardState:
    last_event_seconds: Optional[float] = None


class PreviousTransactionState:
    def __init__(self):
        self._state: Dict[str, CardState] = {}

    def update(self, card_id: str, event_seconds: float) -> Optional[float]:
        state = self._state.setdefault(card_id, CardState())
        delta = None
        if state.last_event_seconds is not None:
            delta = max(0.0, event_seconds - state.last_event_seconds)
        state.last_event_seconds = event_seconds
        return delta
