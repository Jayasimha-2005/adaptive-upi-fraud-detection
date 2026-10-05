import argparse
import time
from pathlib import Path
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[2]


def resolve_input_file(cli_input: str | None = None) -> Path:
    if cli_input:
        p = Path(cli_input)
        if p.is_file():
            return p
        raise FileNotFoundError(f"CRITICAL ERROR: Specified dataset file not found: {p}")

    candidates = [
        ROOT / "Datasets" / "IEEE CIS-20260829T103704Z-1-001" / "IEEE CIS" / "train_transaction.csv",
        ROOT / "Datasets" / "raw" / "train_transaction.csv",
        ROOT / "Datasets" / "train_transaction.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        "CRITICAL ERROR: IEEE-CIS train_transaction.csv not found in candidate paths:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nPlease place train_transaction.csv under Datasets/ or pass --input <path>."
    )


parser = argparse.ArgumentParser(description="Convert IEEE-CIS CSV to Parquet")
parser.add_argument("--input", default=None, help="Path to train_transaction.csv")
parser.add_argument("--output", default="spark/benchmarks/data/ieee_cis_parquet", help="Output directory")
args = parser.parse_args()

input_file = resolve_input_file(args.input)
output_dir = Path(args.output).resolve()

if output_dir.exists():
    raise FileExistsError(
        f"Output already exists: {output_dir}\n"
        "Do not overwrite it automatically; inspect it first."
    )

spark = (
    SparkSession.builder
    .appName("IEEE-CIS-CSV-to-Parquet")
    .master("local[2]")
    .config("spark.hadoop.fs.permissions.umask-mode", "000")
    .config("spark.local.dir", r"C:\spark-temp")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    start = time.perf_counter()

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(input_file))
    )

    df.write.mode("errorifexists").parquet(str(output_dir))

    conversion_seconds = time.perf_counter() - start

    parquet_df = spark.read.parquet(str(output_dir))
    records = parquet_df.count()

    print("\n=== Conversion Results ===")
    print(f"Records verified: {records:,}")
    print(f"Conversion time: {conversion_seconds:.3f} seconds")
    print(f"Conversion time: {conversion_seconds * 1000:.1f} milliseconds")
    print(f"Parquet folder: {output_dir}")
finally:
    spark.stop()
