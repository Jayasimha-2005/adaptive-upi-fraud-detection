# Spark Benchmark Results (WSL)

## Environment

- Execution environment: Ubuntu under WSL
- Processing engine: Apache Spark 3.5.9
- Dataset: IEEE-CIS transaction dataset
- Dataset size: 590,540 records

## CSV vs Parquet Batch Aggregation

The same aggregation was run against CSV and Parquet data, with three trials per format.

| Metric | CSV | Parquet |
|---|---:|---:|
| Median processing time | 4.6154 s | 0.6370 s |
| Median throughput | 127,949.66 records/s | 927,010.66 records/s |

- Observed speedup: 7.25x for Parquet.
- Observed processing-time reduction: 86.20%.
- Both formats returned 590,540 records and 20,663 fraud records.
- CSV-to-Parquet conversion took 58.52 seconds and is not included in the aggregation timings.

These results describe this batch aggregation workload only; they do not establish a universal speedup for every Spark job.

## Kafka Streaming Test

A separate streaming test processed 1,000 synthetic transactions through Spark Structured Streaming and Kafka.

| Metric | Result |
|---|---:|
| Records processed | 1,000 |
| Elapsed processing time | 11.71 s |
| Throughput | 85.40 records/s |
| Micro-batches | 4 |
| Average records per micro-batch | 250 |

This is a separate streaming workload and should not be directly compared with the batch aggregation throughput as if both measured the same task.

## Result Files

- `results/csv_vs_parquet_wsl.csv`
- `results/streaming_wsl.csv`

## Reproducibility Notes

The Parquet dataset is generated locally and is not intended to be committed to Git. Dataset paths, Java versions, storage location, and machine configuration can affect results.
