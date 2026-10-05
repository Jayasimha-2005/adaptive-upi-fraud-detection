from spark.features.realtime_features import Event, calculate_features


def test_realtime_feature_values():
    events = [
        Event("CARD123", "M1", 500, 0),
        Event("CARD123", "M2", 800, 120),
        Event("CARD123", "M3", 12000, 180),
        Event("CARD123", "M4", 2000, 240),
    ]
    f = calculate_features(events, 240)
    assert f["transaction_count_5m"] == 4
    assert f["transaction_amount_5m"] == 15300
    assert f["transaction_count_10m"] == 4
    assert f["transaction_amount_10m"] == 15300
    assert f["unique_merchants_10m"] == 4
    assert f["time_since_previous_transaction"] == 60
