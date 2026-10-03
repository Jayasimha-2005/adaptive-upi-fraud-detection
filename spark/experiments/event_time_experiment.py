"""Deterministic out-of-order event-time and watermark demonstration."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TimedEvent:
    minute: int
    amount: float


EVENTS = [
    TimedEvent(1, 500),    # 10:01
    TimedEvent(4, 2000),   # 10:04
    TimedEvent(2, 800),    # 10:02 arrives late
    TimedEvent(3, 12000),  # 10:03 arrives late
]


def accepted_events(watermark_minutes: int) -> List[TimedEvent]:
    max_seen = -1
    accepted = []
    for e in EVENTS:
        max_seen = max(max_seen, e.minute)
        if e.minute >= max_seen - watermark_minutes:
            accepted.append(e)
    return accepted


def main():
    print("Out-of-order sequence: 10:01, 10:04, 10:02, 10:03")
    for w in (1, 3, 5):
        result = accepted_events(w)
        print(f"Watermark={w}m -> accepted event minutes: {[e.minute for e in result]}")
    print(
        "\nThis deterministic demonstration explains the late-data boundary. "
        "Actual Structured Streaming output depends on window/output mode/query state."
    )


if __name__ == "__main__":
    main()
