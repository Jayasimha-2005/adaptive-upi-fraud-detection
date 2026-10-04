import time
from pathlib import Path
from pyspark.sql import SparkSession

input_file = Path(
    r"C:\Users\HADASSAH KIRAN\Downloads\Datasets\Datasets\IEEE CIS-20260829T103704Z-1-001\IEEE CIS\train_transaction.csv"
)

output_dir = Path("spark/benchmarks/data/ieee_cis_parquet").resolve()

if not input_file.exists():
    raise FileNotFoundError(f"CSV dataset not found: {input_file}")

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
