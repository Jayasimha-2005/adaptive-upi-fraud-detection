import argparse
import os
import sys

# Ensure Spark uses the same Python interpreter as the driver.
# This is especially important on Windows when using a virtual environment.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from typing import Iterator

import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
    ArrayType,
)
from pyspark.sql.streaming.state import GroupStateTimeout


# ============================================================
# CONFIGURATION
# ============================================================

SPARK_VERSION = "3.5.9"

KAFKA_PACKAGE = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9"
)

DEFAULT_BOOTSTRAP = "localhost:9092"
DEFAULT_TOPIC = "fraud-transactions"

DEFAULT_OUTPUT = (
    "spark/output/streaming/features"
)

DEFAULT_CHECKPOINT = (
    "spark/output/streaming/checkpoint"
)

DEFAULT_PREVIOUS_OUTPUT = (
    "spark/output/streaming/previous_transaction"
)

DEFAULT_PREVIOUS_CHECKPOINT = (
    "spark/output/streaming/"
    "previous_transaction_checkpoint_v3"
)


# ============================================================
# SPARK SESSION
# ============================================================

def create_spark() -> SparkSession:
    """
    Create Spark session for the streaming pipeline.
    """

    spark = (
        SparkSession.builder
        .appName(
            "AdaptiveUPIFraudStreaming"
        )
        .master("local[2]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config(
            "spark.jars.packages",
            KAFKA_PACKAGE,
        )
        .config(
            "spark.sql.adaptive.enabled",
            "false",
        )
        .config(
            "spark.sql.shuffle.partitions",
            "8",
        )
        .config(
            "spark.sql.streaming."
            "forceDeleteTempCheckpointLocation",
            "true",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print(
        "Spark session created successfully."
    )

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
    Read raw transactions from Kafka.
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

    print(
        "Kafka source created successfully."
    )

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
    Convert Kafka binary values into
    structured transaction records.
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
        .select(
            "transaction.*"
        )
    )

    return parsed_stream


# ============================================================
# VALIDATION
# ============================================================

def validate_transactions(
    parsed_stream: DataFrame,
) -> DataFrame:
    """
    Remove invalid transactions.
    """

    validated_stream = (
        parsed_stream
        .filter(
            F.col("transaction_id")
            .isNotNull()
        )
        .filter(
            F.col("card_id")
            .isNotNull()
        )
        .filter(
            F.col("merchant_id")
            .isNotNull()
        )
        .filter(
            F.col("amount")
            .isNotNull()
        )
        .filter(
            F.col("event_time")
            .isNotNull()
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
    Build real-time event-time fraud features.

    Watermark:
        10 minutes

    Window:
        10 minutes

    Slide:
        1 minute

    Features:
        transaction_count_5m
        transaction_amount_5m
        transaction_count_10m
        transaction_amount_10m
        unique_merchants_10m
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
    # 10-MINUTE SLIDING WINDOW
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
    # LAST 5 MINUTES OF EACH 10-MINUTE WINDOW
    # --------------------------------------------------------

    five_minute_condition = (
        F.col("event_time").cast("long")
        >=
        (
            F.col(
                "time_window.start"
            ).cast("long")
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

            # ------------------------------------------------
            # 5-MINUTE FEATURES
            # ------------------------------------------------

            F.sum(
                F.when(
                    five_minute_condition,
                    F.lit(1),
                ).otherwise(
                    F.lit(0)
                )
            )
            .cast("long")
            .alias(
                "transaction_count_5m"
            ),

            F.sum(
                F.when(
                    five_minute_condition,
                    F.col("amount"),
                ).otherwise(
                    F.lit(0.0)
                )
            )
            .alias(
                "transaction_amount_5m"
            ),

            # ------------------------------------------------
            # 10-MINUTE FEATURES
            # ------------------------------------------------

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
    # FINAL OUTPUT
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
# PREVIOUS TRANSACTION STATE
# ============================================================

# State is stored as:
#
# (
#     previous_event_times
# )
#
# Example:
#
# CARD001
#     [
#         10:00,
#         10:03,
#         10:07
#     ]
#
# We retain the latest 1000 event timestamps per card.
#
# This gives the stateful processor enough history to
# calculate previous transactions even when events arrive
# somewhat out of order.

STATE_SCHEMA = StructType(
    [
        StructField(
            "event_times",
            ArrayType(
                TimestampType(),
                containsNull=False,
            ),
            True,
        ),
    ]
)


# ============================================================
# PREVIOUS TRANSACTION OUTPUT SCHEMA
# ============================================================

PREVIOUS_OUTPUT_SCHEMA = StructType(
    [
        StructField(
            "transaction_id",
            StringType(),
            False,
        ),
        StructField(
            "card_id",
            StringType(),
            False,
        ),
        StructField(
            "merchant_id",
            StringType(),
            False,
        ),
        StructField(
            "amount",
            DoubleType(),
            False,
        ),
        StructField(
            "event_time",
            TimestampType(),
            False,
        ),
        StructField(
            "previous_transaction_time",
            TimestampType(),
            True,
        ),
        StructField(
            "time_since_previous_transaction",
            DoubleType(),
            True,
        ),
    ]
)


# ============================================================
# STATEFUL PREVIOUS TRANSACTION FUNCTION
# ============================================================

def previous_transaction_state_function(
    key,
    pdf_iter: Iterator[pd.DataFrame],
    state,
):
    """
    Stateful per-card previous transaction calculation.

    Each card_id is one state group.

    For every transaction:

        previous_transaction_time
        time_since_previous_transaction

    are calculated from event_time.

    State keeps a rolling history of event timestamps.
    """

    # --------------------------------------------------------
    # COLLECT ALL DATA FOR THIS CARD IN THIS MICRO-BATCH
    # --------------------------------------------------------

    frames = []

    for pdf in pdf_iter:

        if pdf is not None and not pdf.empty:
            frames.append(pdf)

    # --------------------------------------------------------
    # NO NEW DATA
    # --------------------------------------------------------

    if not frames:
        return

    batch_pdf = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # NORMALIZE EVENT TIME
    # --------------------------------------------------------

    batch_pdf["event_time"] = pd.to_datetime(
        batch_pdf["event_time"],
        errors="coerce",
    )

    batch_pdf = batch_pdf.dropna(
        subset=["event_time"]
    )

    if batch_pdf.empty:
        return

    # --------------------------------------------------------
    # SORT TRANSACTIONS BY EVENT TIME
    # --------------------------------------------------------

    batch_pdf = batch_pdf.sort_values(
        by=[
            "event_time",
            "transaction_id",
        ],
        kind="mergesort",
    )

    # --------------------------------------------------------
    # LOAD EXISTING STATE
    # --------------------------------------------------------

    history = []

    if state.exists:

        current_state = state.get()

        if (
            current_state is not None
            and len(current_state) > 0
            and current_state[0] is not None
        ):
            history = list(
                current_state[0]
            )

    # --------------------------------------------------------
    # NORMALIZE STATE TIMESTAMPS
    # --------------------------------------------------------

    normalized_history = []

    for value in history:

        timestamp = pd.Timestamp(
            value
        ).to_pydatetime()

        normalized_history.append(
            timestamp
        )

    history = normalized_history

    # --------------------------------------------------------
    # CALCULATE PREVIOUS TRANSACTION
    # --------------------------------------------------------

    output_rows = []

    for _, row in batch_pdf.iterrows():

        current_time = pd.Timestamp(
            row["event_time"]
        ).to_pydatetime()

        # ----------------------------------------------------
        # FIND MOST RECENT EVENT BEFORE CURRENT EVENT
        # ----------------------------------------------------

        previous_candidates = [
            timestamp
            for timestamp in history
            if timestamp < current_time
        ]

        if previous_candidates:

            previous_time = max(
                previous_candidates
            )

            difference_seconds = (
                current_time
                - previous_time
            ).total_seconds()

        else:

            previous_time = None
            difference_seconds = None

        # ----------------------------------------------------
        # CREATE OUTPUT ROW
        # ----------------------------------------------------

        output_rows.append(
            {
                "transaction_id": str(
                    row["transaction_id"]
                ),
                "card_id": str(
                    row["card_id"]
                ),
                "merchant_id": str(
                    row["merchant_id"]
                ),
                "amount": float(
                    row["amount"]
                ),
                "event_time": current_time,
                "previous_transaction_time":
                    previous_time,
                "time_since_previous_transaction":
                    (
                        float(
                            difference_seconds
                        )
                        if difference_seconds
                        is not None
                        else None
                    ),
            }
        )

        # ----------------------------------------------------
        # ADD CURRENT EVENT TO STATE HISTORY
        # ----------------------------------------------------

        history.append(
            current_time
        )

        # Keep history sorted.
        history.sort()

        # Keep only the latest 1000 timestamps.
        if len(history) > 1000:
            history = history[-1000:]

    # --------------------------------------------------------
    # UPDATE STATE
    # --------------------------------------------------------

    state.update(
        (
            history,
        )
    )

    # --------------------------------------------------------
    # RETURN CURRENT BATCH OUTPUT
    # --------------------------------------------------------

    if output_rows:

        yield pd.DataFrame(
            output_rows,
            columns=[
                "transaction_id",
                "card_id",
                "merchant_id",
                "amount",
                "event_time",
                "previous_transaction_time",
                "time_since_previous_transaction",
            ],
        )


# ============================================================
# BUILD PREVIOUS TRANSACTION STREAM
# ============================================================

def calculate_previous_transaction_features(
    validated_stream: DataFrame,
) -> DataFrame:
    """
    Build stateful previous-transaction features.

    This is a genuine Structured Streaming stateful
    operation using applyInPandasWithState.
    """

    print(
        "Building stateful previous-transaction features..."
    )

    previous_features = (
        validated_stream
        .groupBy("card_id")
        .applyInPandasWithState(
            previous_transaction_state_function,
            outputStructType=PREVIOUS_OUTPUT_SCHEMA,
            stateStructType=STATE_SCHEMA,
            outputMode="Update",
            timeoutConf=GroupStateTimeout.NoTimeout,
        )
    )

    return previous_features


# ============================================================
# WRITE PREVIOUS TRANSACTION BATCH
# ============================================================

def write_previous_transaction_batch(
    batch_df: DataFrame,
    batch_id: int,
    output_path: str,
):
    """
    Write stateful previous-transaction output.

    applyInPandasWithState uses Update mode.
    Therefore we use foreachBatch to append the
    resulting records to Parquet.

    NOTE:
        Do NOT call batch_df.isEmpty() here.
        isEmpty() triggers another Spark action and can
        cause an additional Python worker execution.
    """

    print(
        f"Writing previous-transaction batch "
        f"{batch_id}..."
    )

    (
        batch_df
        .write
        .mode("append")
        .format("parquet")
        .save(output_path)
    )

    print(
        f"Previous-transaction batch "
        f"{batch_id} written."
    )


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
    Start the complete Spark Structured Streaming
    processing layer.

    Query 1:
        Kafka
          ↓
        JSON
          ↓
        Validation
          ↓
        Watermark
          ↓
        5m / 10m windows
          ↓
        Parquet

    Query 2:
        Kafka
          ↓
        JSON
          ↓
        Validation
          ↓
        Stateful card_id processing
          ↓
        previous_transaction_time
          ↓
        time_since_previous_transaction
          ↓
        Parquet
    """

    spark = create_spark()

    # --------------------------------------------------------
    # CREATE DIRECTORIES
    # --------------------------------------------------------

    os.makedirs(
        output,
        exist_ok=True,
    )

    os.makedirs(
        checkpoint,
        exist_ok=True,
    )

    os.makedirs(
        previous_output,
        exist_ok=True,
    )

    os.makedirs(
        previous_checkpoint,
        exist_ok=True,
    )

    # ========================================================
    # QUERY 1
    # WINDOW FEATURES
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "CREATING WINDOW FEATURE STREAM"
    )

    print(
        "=========================================="
    )

    raw_stream_window = read_kafka_stream(
        spark,
        bootstrap_servers,
        topic,
    )

    parsed_stream_window = (
        parse_transactions(
            raw_stream_window
        )
    )

    validated_stream_window = (
        validate_transactions(
            parsed_stream_window
        )
    )

    window_features = build_window_features(
        validated_stream_window
    )

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

    # ========================================================
    # QUERY 2
    # PREVIOUS TRANSACTION STATE
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "CREATING PREVIOUS-TRANSACTION STREAM"
    )

    print(
        "=========================================="
    )

    # IMPORTANT:
    # This is a separate Kafka streaming source/query.
    # Spark documentation recommends separate queries
    # when multiple stateful operations are required.

    raw_stream_previous = read_kafka_stream(
        spark,
        bootstrap_servers,
        topic,
    )

    parsed_stream_previous = (
        parse_transactions(
            raw_stream_previous
        )
    )

    validated_stream_previous = (
        validate_transactions(
            parsed_stream_previous
        )
    )

    previous_features = (
        calculate_previous_transaction_features(
            validated_stream_previous
        )
    )

    print(
        "\nStarting previous-transaction query..."
    )

    previous_query = (
        previous_features
        .writeStream
        .outputMode("update")
        .option(
            "checkpointLocation",
            previous_checkpoint,
        )
        .trigger(
            processingTime="10 seconds"
        )
        .foreachBatch(
            lambda batch_df, batch_id:
                write_previous_transaction_batch(
                    batch_df,
                    batch_id,
                    previous_output,
                )
        )
        .start()
    )

    print(
        "Previous-transaction query started."
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "COMPLETE SPARK STREAMING PIPELINE"
    )

    print(
        "=========================================="
    )

    print(
        f"Kafka bootstrap: "
        f"{bootstrap_servers}"
    )

    print(
        f"Kafka topic: {topic}"
    )

    print(
        "\nWINDOW FEATURES"
    )

    print(
        "5-minute features: ENABLED"
    )

    print(
        "10-minute features: ENABLED"
    )

    print(
        "Event-time processing: ENABLED"
    )

    print(
        "Watermark: 10 minutes"
    )

    print(
        "Sliding window: 10 minutes / 1 minute"
    )

    print(
        "\nPREVIOUS TRANSACTION FEATURES"
    )

    print(
        "Stateful processing: ENABLED"
    )

    print(
        "previous_transaction_time: ENABLED"
    )

    print(
        "time_since_previous_transaction: ENABLED"
    )

    print(
        "Per-card state: ENABLED"
    )

    print(
        "\nOUTPUTS"
    )

    print(
        f"Window output: {output}"
    )

    print(
        f"Previous transaction output: "
        f"{previous_output}"
    )

    print(
        f"Window checkpoint: {checkpoint}"
    )

    print(
        f"Previous checkpoint: "
        f"{previous_checkpoint}"
    )

    print(
        "\nWaiting for Kafka transactions..."
    )

    print(
        "=========================================="
    )

    # ========================================================
    # WAIT
    # ========================================================

    try:

        # Wait for either query to terminate.
        spark.streams.awaitAnyTermination()

    except KeyboardInterrupt:

        print(
            "\nStopping Spark streaming..."
        )

    finally:

        print(
            "Stopping streaming queries..."
        )

        try:
            if window_query.isActive:
                window_query.stop()
        except Exception:
            pass

        try:
            if previous_query.isActive:
                previous_query.stop()
        except Exception:
            pass

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
        help=(
            "Streaming window feature "
            "Parquet output directory"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=(
            "Window feature checkpoint "
            "directory"
        ),
    )

    parser.add_argument(
        "--previous-output",
        default=DEFAULT_PREVIOUS_OUTPUT,
        help=(
            "Previous transaction Parquet "
            "output directory"
        ),
    )

    parser.add_argument(
        "--previous-checkpoint",
        default=DEFAULT_PREVIOUS_CHECKPOINT,
        help=(
            "Previous transaction state "
            "checkpoint directory"
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