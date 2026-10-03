from pyflink.common import Duration, WatermarkStrategy
from pyflink.common.watermark_strategy import TimestampAssigner


class TransactionTimestampAssigner(TimestampAssigner):

    def extract_timestamp(self, transaction, record_timestamp):
        return int(transaction.event_time.timestamp() * 1000)


def transaction_watermark_strategy():

    return (
        WatermarkStrategy
        .for_bounded_out_of_orderness(
            Duration.of_seconds(30)
        )
        .with_timestamp_assigner(
            TransactionTimestampAssigner()
        )
    )