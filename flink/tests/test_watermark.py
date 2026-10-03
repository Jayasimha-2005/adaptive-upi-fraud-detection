from flink.streaming.watermark import transaction_watermark_strategy

def test_strategy():
    assert transaction_watermark_strategy() is not None
