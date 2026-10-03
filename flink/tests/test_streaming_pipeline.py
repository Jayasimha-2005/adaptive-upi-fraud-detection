from flink.streaming.transaction_schema import Transaction

def test_transaction():
    tx = Transaction.from_dict({
        "transaction_id": "T1",
        "user_id": "U1",
        "amount": 100,
        "event_time": "2026-10-02T08:00:00+00:00"
    })
    assert tx.user_id == "U1"
    assert tx.amount == 100.0
