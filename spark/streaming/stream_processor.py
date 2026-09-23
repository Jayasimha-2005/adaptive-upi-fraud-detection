import argparse
import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
)


# ============================================================
# CONFIGURATION
# ============================================================

SPARK_VERSION = "3.5.9"

KAFKA_PACKAGE = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9"
)

DEFAULT_BOOTSTRAP = "localhost:9092"
DEFAULT_TOPIC = "fraud-transactions"

DEFAULT_OUTPUT = "spark/output/streaming/features"
DEFAULT_CHECKPOINT = "spark/output/streaming/checkpoint"

DEFAULT_PREVIOUS_OUTPUT = (
    "spark/output/streaming/previous_transaction"
)

DEFAULT_PREVIOUS_CHECKPOINT = (
    "spark/output/streaming/previous_transaction_checkpoint_v2"
)


# ============================================================
# SPARK SESSION
# ============================================================

def create_spark() -> SparkSession:
    """
    Create the Spark session used by the streaming pipeline.
    """

    spark = (
        SparkSession.builder
        .appName("AdaptiveUPIFraudStreaming")
        .master("local[*]")
        .config("spark.jars.packages", KAFKA_PACKAGE)
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config(
            "spark.sql.streaming.forceDeleteTempCheckpointLocation",
            "true",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("Spark session created successfully.")

    return spark


# ============================================================
# KAFKA SOURCE
# ============================================================

def read_kafka_stream(
    spark: SparkSession,
    bootstrap_servers: str,
    topic: str,
) -> DataFrame:
    """
    Read raw messages from Kafka.
    """

    print("Connecting to Kafka...")

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            bootstrap_servers,
        )
        .option(
            "subscribe",
            topic,
        )
        .option(
            "startingOffsets",
            "latest",
        )
        .option(
            "failOnDataLoss",
            "false",
        )
        .load()
    )

    print("Kafka source created successfully.")

    return raw_stream


# ============================================================
# JSON SCHEMA
# ============================================================

TRANSACTION_SCHEMA = StructType(
    [
        StructField(
            "transaction_id",
            StringType(),
            True,
        ),
        StructField(
            "card_id",
            StringType(),
            True,
        ),
        StructField(
            "merchant_id",
            StringType(),
            True,
        ),
        StructField(
            "amount",
            DoubleType(),
            True,
        ),
        StructField(
            "event_time",
            TimestampType(),
            True,
        ),
    ]
)


# ============================================================
# PARSE JSON
# ============================================================

def parse_transactions(
    raw_stream: DataFrame,
) -> DataFrame:
    """
    Convert Kafka binary values into structured
    transaction records.
    """

    parsed_stream = (
        raw_stream
        .selectExpr(
            "CAST(value AS STRING) AS json_value"
        )
        .select(
            F.from_json(
                F.col("json_value"),
                TRANSACTION_SCHEMA,
            ).alias("transaction")
        )
        .select("transaction.*")
    )

    return parsed_stream


# ============================================================
# VALIDATION
# ============================================================

def validate_transactions(
    parsed_stream: DataFrame,
) -> DataFrame:
    """
    Remove invalid transaction records.
    """

    validated_stream = (
        parsed_stream
        .filter(
            F.col("transaction_id").isNotNull()
        )
        .filter(
            F.col("card_id").isNotNull()
        )
        .filter(
            F.col("merchant_id").isNotNull()
        )
        .filter(
            F.col("amount").isNotNull()
        )
        .filter(
            F.col("event_time").isNotNull()
        )
        .filter(
            F.col("amount") >= 0
        )
    )

    return validated_stream


# ============================================================
# REAL-TIME WINDOW FEATURES
# ============================================================

def build_window_features(
    validated_stream: DataFrame,
) -> DataFrame:
    """
    Build event-time fraud detection features.

    Watermark:
        10 minutes

    Main event-time window:
        10 minutes
        sliding every 1 minute

    Features:
        transaction_count_5m
        transaction_amount_5m
        transaction_count_10m
        transaction_amount_10m
        unique_merchants_10m

    The 5-minute features represent the final 5 minutes
    of each 10-minute event-time window.

    Example:

        10-minute window:
            10:00 -> 10:10

        5-minute portion:
            10:05 -> 10:10
    """

    print(
        "Building streaming window features..."
    )

    # --------------------------------------------------------
    # WATERMARK
    # --------------------------------------------------------

    watermarked_stream = (
        validated_stream
        .withWatermark(
            "event_time",
            "10 minutes",
        )
    )

    # --------------------------------------------------------
    # CREATE ONE 10-MINUTE SLIDING EVENT-TIME WINDOW
    # --------------------------------------------------------

    windowed_stream = (
        watermarked_stream
        .withColumn(
            "time_window",
            F.window(
                F.col("event_time"),
                "10 minutes",
                "1 minute",
            ),
        )
    )

    # --------------------------------------------------------
    # DETERMINE THE LAST 5 MINUTES
    # --------------------------------------------------------
    #
    # A 10-minute window:
    #
    #       start                 end
    #         |---------------------|
    #         0         5           10 min
    #
    # The last 5 minutes are:
    #
    #                   |-----------|
    #                   5           10 min
    #
    # We compare event_time with:
    #
    # window_start + 5 minutes
    #
    # Using Unix seconds here makes the comparison
    # reliable across Spark/Python versions.
    # --------------------------------------------------------

    five_minute_condition = (
        F.col("event_time").cast("long")
        >=
        (
            F.col("time_window.start").cast("long")
            + F.lit(300)
        )
    )

    # --------------------------------------------------------
    # AGGREGATION
    # --------------------------------------------------------

    result = (
        windowed_stream
        .groupBy(
            F.col("time_window"),
            F.col("card_id"),
        )
        .agg(

            # =================================================
            # 5-MINUTE FEATURES
            # =================================================

            F.sum(
                F.when(
                    five_minute_condition,
                    F.lit(1),
                ).otherwise(
                    F.lit(0)
                )
            ).cast("long").alias(
                "transaction_count_5m"
            ),

            F.sum(
                F.when(
                    five_minute_condition,
                    F.col("amount"),
                ).otherwise(
                    F.lit(0.0)
                )
            ).alias(
                "transaction_amount_5m"
            ),

            # =================================================
            # 10-MINUTE FEATURES
            # =================================================

            F.count("*").alias(
                "transaction_count_10m"
            ),

            F.sum(
                F.col("amount")
            ).alias(
                "transaction_amount_10m"
            ),

            F.approx_count_distinct(
                "merchant_id"
            ).alias(
                "unique_merchants_10m"
            ),
        )
    )

    # --------------------------------------------------------
    # FINAL OUTPUT COLUMNS
    # --------------------------------------------------------

    result = (
        result
        .select(
            F.col(
                "time_window.start"
            ).alias(
                "window_start"
            ),

            F.col(
                "time_window.end"
            ).alias(
                "window_end"
            ),

            F.col(
                "card_id"
            ),

            F.coalesce(
                F.col(
                    "transaction_count_5m"
                ),
                F.lit(0),
            ).alias(
                "transaction_count_5m"
            ),

            F.coalesce(
                F.col(
                    "transaction_amount_5m"
                ),
                F.lit(0.0),
            ).alias(
                "transaction_amount_5m"
            ),

            F.col(
                "transaction_count_10m"
            ),

            F.col(
                "transaction_amount_10m"
            ),

            F.col(
                "unique_merchants_10m"
            ),
        )
    )

    return result


# ============================================================
# PREVIOUS TRANSACTION FEATURE
# ============================================================

def calculate_previous_transaction_features(
    validated_stream: DataFrame,
    previous_output: str,
    previous_checkpoint: str,
):
    """
    Previous-transaction feature implementation.

    Currently disabled.

    The batch implementation of
    time_since_previous_transaction has been completed
    and verified separately.

    A robust stateful streaming implementation will be
    handled as a separate step.
    """

    print(
        "Previous-transaction feature is currently disabled."
    )

    return None


# ============================================================
# MAIN STREAMING PIPELINE
# ============================================================

def run_streaming(
    bootstrap_servers: str,
    topic: str,
    output: str,
    checkpoint: str,
    previous_output: str,
    previous_checkpoint: str,
):
    """
    Start the Spark Structured Streaming pipeline.

    Current active pipeline:

        Kafka
          ↓
        JSON parsing
          ↓
        Validation
          ↓
        Event-time processing
          ↓
        10-minute watermark
          ↓
        10-minute sliding window
          ↓
        ┌─────────────────────────────┐
        │                             │
        ▼                             ▼
    Last 5 minutes             Full 10 minutes
        │                             │
        └──────────────┬──────────────┘
                       ↓
                  Feature output
                       ↓
                    Parquet
    """

    spark = create_spark()

    # --------------------------------------------------------
    # READ KAFKA
    # --------------------------------------------------------

    raw_stream = read_kafka_stream(
        spark,
        bootstrap_servers,
        topic,
    )

    # --------------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------------

    parsed_stream = parse_transactions(
        raw_stream
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validated_stream = validate_transactions(
        parsed_stream
    )

    # --------------------------------------------------------
    # BUILD WINDOW FEATURES
    # --------------------------------------------------------

    window_features = build_window_features(
        validated_stream
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------------

    os.makedirs(
        output,
        exist_ok=True,
    )

    os.makedirs(
        checkpoint,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # START WINDOW FEATURE QUERY
    # --------------------------------------------------------

    print(
        "\nStarting window feature query..."
    )

    window_query = (
        window_features
        .writeStream
        .format("parquet")
        .outputMode("append")
        .option(
            "path",
            output,
        )
        .option(
            "checkpointLocation",
            checkpoint,
        )
        .trigger(
            processingTime="10 seconds"
        )
        .start()
    )

    print(
        "Window feature query started."
    )

    # --------------------------------------------------------
    # PREVIOUS TRANSACTION QUERY
    #
    # TEMPORARILY DISABLED
    # --------------------------------------------------------

    print(
        "\nPrevious-transaction query is "
        "temporarily disabled."
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "Spark streaming query started."
    )

    print(
        f"Kafka topic: {topic}"
    )

    print(
        f"Window output: {output}"
    )

    print(
        f"Checkpoint: {checkpoint}"
    )

    print(
        "5-minute feature: ENABLED"
    )

    print(
        "10-minute feature: ENABLED"
    )

    print(
        "Event-time processing: ENABLED"
    )

    print(
        "Watermark: 10 minutes"
    )

    print(
        "Previous-transaction feature: DISABLED"
    )

    print(
        "Waiting for Kafka transactions..."
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # WAIT FOR STREAMING QUERY
    # --------------------------------------------------------

    try:

        window_query.awaitTermination()

    except KeyboardInterrupt:

        print(
            "\nStopping Spark streaming..."
        )

    finally:

        print(
            "Stopping Spark session..."
        )

        spark.stop()


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Adaptive UPI Fraud Detection "
            "Spark Structured Streaming Pipeline"
        )
    )

    parser.add_argument(
        "--bootstrap",
        default=DEFAULT_BOOTSTRAP,
        help="Kafka bootstrap server",
    )

    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="Kafka topic",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Streaming Parquet output directory",
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Streaming checkpoint directory",
    )

    parser.add_argument(
        "--previous-output",
        default=DEFAULT_PREVIOUS_OUTPUT,
        help=(
            "Previous transaction output directory "
            "(currently disabled)"
        ),
    )

    parser.add_argument(
        "--previous-checkpoint",
        default=DEFAULT_PREVIOUS_CHECKPOINT,
        help=(
            "Previous transaction checkpoint "
            "(currently disabled)"
        ),
    )

    args = parser.parse_args()

    run_streaming(
        bootstrap_servers=args.bootstrap,
        topic=args.topic,
        output=args.output,
        checkpoint=args.checkpoint,
        previous_output=args.previous_output,
        previous_checkpoint=args.previous_checkpoint,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
    