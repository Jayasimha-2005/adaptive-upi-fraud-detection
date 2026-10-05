"""Run the local Spark batch benchmark and preserve measured results."""

import argparse
from pathlib import Path
from spark.experiments.batch_vs_streaming import benchmark_spark_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="spark/benchmarks/results/batch_results.csv")
    a = p.parse_args()

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("Fraud-Benchmarks").master("local[*]").getOrCreate()
    try:
        result = benchmark_spark_batch(spark, a.input)
    finally:
        spark.stop()

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "mode,records,processing_seconds,throughput_records_per_second\n"
        f"{result['mode']},{result['records']},{result['processing_seconds']},"
        f"{result['throughput_records_per_second']}\n",
        encoding="utf-8",
    )
    print(f"Saved measured result to {out}")


if __name__ == "__main__":
    main()
