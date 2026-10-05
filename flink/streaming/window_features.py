from pyflink.common.time import Time
from pyflink.datastream.window import SlidingEventTimeWindows


def fraud_feature_window():
    # 10-minute window, advancing every 5 minutes.
    return SlidingEventTimeWindows.of(
        Time.minutes(10),
        Time.minutes(5)
    )