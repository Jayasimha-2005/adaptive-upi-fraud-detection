"""Measured Spark batch and Kafka streaming benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from threading import Lock

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


KAFKA_PACKAGE = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.9"
)


TRANSACTION_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), True),
        StructField("card_id", StringType(), True),
        StructField("merchant_id", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("event_time", TimestampType(), True),
    ]
)


def benchmark_spark_batch(spark, input_path: str) -> dict:
    """Measure Spark batch read + count."""

    start = time.perf_counter()

    if input_path.endswith(".parquet"):
        df = spark.read.parquet(input_path)
    else:
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(input_path)
        )

    rows = df.count()

    elapsed = time.perf_counter() - start

    return {
        "mode": "batch",
        "records": rows,
        "processing_seconds": elapsed,
        "throughput_records_per_second": (
            rows / elapsed if elapsed else None
        ),
    }


def benchmark_spark_streaming(
    bootstrap_servers: str,
    topic: str,
    checkpoint: str,
    target_records: int,
    trigger_seconds: int = 5,
) -> dict:
    """
    Measure Kafka -> Spark Structured Streaming.

    Timing starts when the first non-empty micro-batch begins
    processing and ends after the target number of records has
    been counted.
    """

    checkpoint_path = Path(checkpoint)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("Fraud-Streaming-Benchmark")
        .master("local[*]")
        .config("spark.jars.packages", KAFKA_PACKAGE)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    raw_stream = (
        spark.readStream
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            bootstrap_servers,
        )
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
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
        .filter(F.col("transaction_id").isNotNull())
        .filter(F.col("card_id").isNotNull())
        .filter(F.col("merchant_id").isNotNull())
        .filter(F.col("amount").isNotNull())
        .filter(F.col("event_time").isNotNull())
        .filter(F.col("amount") >= 0)
    )

    lock = Lock()

    stats = {
        "records": 0,
        "batches": 0,
        "batch_records": [],
        "start_time": None,
        "end_time": None,
    }

    def process_batch(batch_df, batch_id):
        # Count the batch first.
        batch_start = time.perf_counter()
        count = batch_df.count()
        batch_end = time.perf_counter()

        with lock:
            # Start timing when the first real records arrive.
            if count > 0 and stats["start_time"] is None:
                stats["start_time"] = batch_start

            stats["records"] += count
            stats["batches"] += 1
            stats["batch_records"].append(count)

            # Stop timing after the target is actually processed.
            if (
                stats["records"] >= target_records
                and stats["end_time"] is None
            ):
                stats["end_time"] = batch_end

        print(
            f"Batch {batch_id}: "
            f"{count} records, "
            f"{batch_end - batch_start:.3f}s, "
            f"total={stats['records']}"
        )

    query = (
        parsed.writeStream
        .foreachBatch(process_batch)
        .option(
            "checkpointLocation",
            str(checkpoint_path),
        )
        .trigger(
            processingTime=f"{trigger_seconds} seconds"
        )
        .start()
    )

    print()
    print("=" * 60)
    print("STREAMING BENCHMARK STARTED")
    print("=" * 60)
    print(f"Kafka: {bootstrap_servers}")
    print(f"Topic: {topic}")
    print(f"Target records: {target_records}")
    print(f"Trigger: {trigger_seconds} seconds")
    print()
    print(
        "Now send the benchmark transactions to Kafka."
    )
    print(
        f"Send exactly {target_records} valid transactions."
    )
    print("=" * 60)

    try:
        while True:
            time.sleep(1)

            with lock:
                finished = (
                    stats["end_time"] is not None
                    and stats["records"] >= target_records
                )

            if finished:
                break

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")

    finally:
        query.stop()
        spark.stop()

    records = stats["records"]

    if (
        stats["start_time"] is not None
        and stats["end_time"] is not None
    ):
        elapsed = (
            stats["end_time"]
            - stats["start_time"]
        )
    else:
        elapsed = 0.0

    throughput = (
        records / elapsed
        if elapsed > 0
        else None
    )

    average_batch = (
        sum(stats["batch_records"])
        / len(stats["batch_records"])
        if stats["batch_records"]
        else 0
    )

    return {
        "mode": "streaming",
        "records": records,
        "processing_seconds": elapsed,
        "throughput_records_per_second": throughput,
        "micro_batches": stats["batches"],
        "average_records_per_micro_batch": average_batch,
    }


def write_result(result: dict, output: str) -> None:
    """Write measured result to CSV."""

    out = Path(output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=result.keys(),
        )
        writer.writeheader()
        writer.writerow(result)

    print()
    print(f"Saved measured result to {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Measured Spark batch/streaming benchmark"
    )

    parser.add_argument(
        "--mode",
        choices=["batch", "streaming"],
        default="batch",
    )

    parser.add_argument(
        "--input",
        help="Batch input path",
    )

    parser.add_argument(
        "--bootstrap",
        default="localhost:9092",
    )

    parser.add_argument(
        "--topic",
        default="fraud-transactions",
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "spark/benchmarks/results/"
            "streaming_checkpoint"
        ),
    )

    parser.add_argument(
        "--target-records",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--trigger-seconds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        default=(
            "spark/benchmarks/results/"
            "benchmark_results.csv"
        ),
    )

    args = parser.parse_args()

    if args.mode == "batch":
        if not args.input:
            parser.error(
                "--input is required for batch mode"
            )

        spark = (
            SparkSession.builder
            .appName("Fraud-Batch-Benchmark")
            .master("local[*]")
            .config(
                "spark.sql.adaptive.enabled",
                "true",
            )
            .getOrCreate()
        )

        try:
            result = benchmark_spark_batch(
                spark,
                args.input,
            )
        finally:
            spark.stop()

    else:
        result = benchmark_spark_streaming(
            bootstrap_servers=args.bootstrap,
            topic=args.topic,
            checkpoint=args.checkpoint,
            target_records=args.target_records,
            trigger_seconds=args.trigger_seconds,
        )

    write_result(result, args.output)

    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()