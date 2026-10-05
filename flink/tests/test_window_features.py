import pytest

def test_window():
    pytest.importorskip("pyflink")
    from flink.streaming.window_features import fraud_feature_window
    assert fraud_feature_window() is not None

