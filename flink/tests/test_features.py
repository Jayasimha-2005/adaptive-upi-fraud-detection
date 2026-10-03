from flink.features.transaction_features import amount_bucket
from flink.features.aggregation_features import aggregate_transactions


def test_amount_bucket():
    assert amount_bucket(50) == "LOW"
    assert amount_bucket(500) == "MEDIUM"
    assert amount_bucket(2400) == "HIGH"
    assert amount_bucket(15000) == "VERY_HIGH"


def test_aggregate_transactions():
    events = [
        {"amount": 500},
        {"amount": 700},
        {"amount": 1200}
    ]

    result = aggregate_transactions(events)

    assert result["transaction_count"] == 3
    assert result["total_amount"] == 2400.0
    assert result["average_amount"] == 800.0