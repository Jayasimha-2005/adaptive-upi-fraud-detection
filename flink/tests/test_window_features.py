from flink.streaming.window_features import fraud_feature_window

def test_window():
    assert fraud_feature_window() is not None
