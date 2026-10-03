from dataclasses import dataclass
from datetime import datetime

@dataclass
class Transaction:
    transaction_id: str
    user_id: str
    amount: float
    event_time: datetime
    merchant_id: str = "UNKNOWN"
    transaction_type: str = "UNKNOWN"

    @staticmethod
    def from_dict(value):
        event_time = value["event_time"]
        if isinstance(event_time, str):
            event_time = datetime.fromisoformat(
                event_time.replace("Z", "+00:00")
            )
        return Transaction(
            str(value["transaction_id"]),
            str(value["user_id"]),
            float(value["amount"]),
            event_time,
            str(value.get("merchant_id", "UNKNOWN")),
            str(value.get("transaction_type", "UNKNOWN"))
        )
