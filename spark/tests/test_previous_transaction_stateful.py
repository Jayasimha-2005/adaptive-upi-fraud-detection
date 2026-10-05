import os
import pytest

pytest.importorskip("pyspark")

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
)
from pyspark.sql.streaming.state import GroupStateTimeout


# ============================================================
# CONFIGURATION
# ============================================================

SPARK_VERSION = "3.5.9"

KAFKA_PACKAGE = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9"
)

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "fraud-transactions"

OUTPUT_PATH = (
    "spark/output/streaming/previous_transaction_test"
)

CHECKPOINT_PATH = (
    "spark/output/streaming/previous_transaction_test_checkpoint"
)


# ============================================================
# SPARK SESSION
# ============================================================

def create_spark():

    spark = (
        SparkSession.builder
        .appName("PreviousTransactionStatefulTest")
        .master("local[1]")
        .config(
            "spark.jars.packages",
            KAFKA_PACKAGE,
        )
        .config(
            "spark.sql.shuffle.partitions",
            "1",
        )
        .config(
            "spark.sql.adaptive.enabled",
            "false",
        )
        .config(
            "spark.sql.streaming.forceDeleteTempCheckpointLocation",
            "true",
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


# ============================================================
# KAFKA SCHEMA
# ============================================================

transaction_schema = StructType(
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
# STATE SCHEMA
# ============================================================

state_schema = StructType(
    [
        StructField(
            "last_transaction_time",
            TimestampType(),
            True,
        ),
    ]
)


# ============================================================
# OUTPUT SCHEMA
# ============================================================

output_schema = StructType(
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
# STATEFUL FUNCTION
# ============================================================

def calculate_previous_transaction(
    key,
    pdf_iter,
    state,
):

    # --------------------------------------------------------
    # Get existing state
    # --------------------------------------------------------

    if state.exists:

        previous_time = state.get[0]

    else:

        previous_time = None


    # --------------------------------------------------------
    # Combine all rows from this group
    # --------------------------------------------------------

    pdf_list = list(pdf_iter)

    if not pdf_list:

        return

    pdf = pd.concat(
        pdf_list,
        ignore_index=True,
    )


    # --------------------------------------------------------
    # Sort transactions by event time
    # --------------------------------------------------------

    pdf["event_time"] = pd.to_datetime(
        pdf["event_time"]
    )

    pdf = pdf.sort_values(
        "event_time"
    )


    # --------------------------------------------------------
    # Calculate previous transaction
    # --------------------------------------------------------

    previous_times = []

    time_differences = []

    current_previous_time = previous_time


    for event_time in pdf["event_time"]:

        previous_times.append(
            current_previous_time
        )

        if current_previous_time is None:

            time_differences.append(
                None
            )

        else:

            difference = (
                event_time
                - current_previous_time
            ).total_seconds()

            time_differences.append(
                float(difference)
            )

        current_previous_time = event_time


    # --------------------------------------------------------
    # Add calculated features
    # --------------------------------------------------------

    pdf[
        "previous_transaction_time"
    ] = previous_times

    pdf[
        "time_since_previous_transaction"
    ] = time_differences


    # --------------------------------------------------------
    # Update state with latest transaction
    # --------------------------------------------------------

    latest_transaction_time = (
        pdf["event_time"].iloc[-1]
    )

    state.update(
        (
            latest_transaction_time,
        )
    )


    # --------------------------------------------------------
    # Return required columns
    # --------------------------------------------------------

    result = pdf[
        [
            "transaction_id",
            "card_id",
            "merchant_id",
            "amount",
            "event_time",
            "previous_transaction_time",
            "time_since_previous_transaction",
        ]
    ]


    yield result


# ============================================================
# FOREACH BATCH
# ============================================================

def write_batch(
    batch_df,
    batch_id,
):

    print(
        f"\nFOREACH BATCH CALLED: {batch_id}"
    )

    print(
        "Writing stateful output directly to Parquet..."
    )

    (
        batch_df
        .write
        .mode("append")
        .parquet(
            OUTPUT_PATH
        )
    )

    print(
        f"SUCCESS: Batch {batch_id} written to Parquet"
    )

    print(
        f"Output path: {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_PATH,
        exist_ok=True,
    )

    os.makedirs(
        CHECKPOINT_PATH,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Create Spark
    # --------------------------------------------------------

    spark = create_spark()


    print()
    print("=" * 60)
    print("STATEFUL PREVIOUS-TRANSACTION TEST")
    print("=" * 60)
    print(
        f"PySpark version: {spark.version}"
    )
    print(
        f"Kafka topic: {TOPIC}"
    )
    print(
        f"Output: {OUTPUT_PATH}"
    )
    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )
    print("=" * 60)


    # --------------------------------------------------------
    # Read Kafka
    # --------------------------------------------------------

    kafka_stream = (
        spark.readStream
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            BOOTSTRAP_SERVERS,
        )
        .option(
            "subscribe",
            TOPIC,
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


    # --------------------------------------------------------
    # Convert Kafka value to string
    # --------------------------------------------------------

    json_stream = (
        kafka_stream
        .select(
            F.col("value")
            .cast("string")
            .alias("json_value")
        )
    )


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    parsed_stream = (
        json_stream
        .select(
            F.from_json(
                F.col("json_value"),
                transaction_schema,
            ).alias("data")
        )
        .select("data.*")
    )


    # --------------------------------------------------------
    # Validate transactions
    # --------------------------------------------------------

    validated_stream = (
        parsed_stream
        .filter(
            F.col("transaction_id").isNotNull()
        )
        .filter(
            F.col("card_id").isNotNull()
        )
        .filter(
            F.col("event_time").isNotNull()
        )
        .filter(
            F.col("amount").isNotNull()
        )
    )


    # --------------------------------------------------------
    # Watermark
    # --------------------------------------------------------

    watermarked_stream = (
        validated_stream
        .withWatermark(
            "event_time",
            "10 minutes",
        )
    )


    # --------------------------------------------------------
    # Stateful previous-transaction calculation
    # --------------------------------------------------------

    stateful_stream = (
        watermarked_stream
        .groupBy(
            "card_id"
        )
        .applyInPandasWithState(
            calculate_previous_transaction,
            outputStructType=output_schema,
            stateStructType=state_schema,
            outputMode="Update",
            timeoutConf=(
                GroupStateTimeout.EventTimeTimeout
            ),
        )
    )


    # --------------------------------------------------------
    # Start query
    # --------------------------------------------------------

    query = (
        stateful_stream
        .writeStream
        .outputMode("update")
        .foreachBatch(
            write_batch
        )
        .option(
            "checkpointLocation",
            CHECKPOINT_PATH,
        )
        .trigger(
            processingTime="10 seconds"
        )
        .start()
    )


    print()
    print("=" * 60)
    print("STATEFUL QUERY STARTED")
    print("=" * 60)
    print(
        "Waiting for Kafka transactions..."
    )
    print(
        "Send NEW transactions to Kafka."
    )
    print("=" * 60)


    # --------------------------------------------------------
    # Keep streaming query alive
    # --------------------------------------------------------

    try:

        query.awaitTermination()

    except KeyboardInterrupt:

        print()
        print(
            "Stopping stateful streaming query..."
        )

        query.stop()

    finally:

        spark.stop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()