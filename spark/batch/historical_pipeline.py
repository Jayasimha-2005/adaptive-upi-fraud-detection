"""IEEE-CIS historical processing with PySpark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def create_spark(
    app_name: str = "AdaptiveFraud-Spark-Batch",
) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.files.maxPartitionBytes", "128m")
        .config("spark.sql.debug.maxToStringFields", "100")
        .getOrCreate()
    )


def read_ieee_cis(
    spark: SparkSession,
    transaction_path: str,
    identity_path: Optional[str] = None,
) -> DataFrame:

    if not Path(transaction_path).exists():
        raise FileNotFoundError(
            f"Transaction file not found: {transaction_path}"
        )

    tx = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(transaction_path)
    )

    if "TransactionID" in tx.columns:
        tx = tx.withColumn(
            "TransactionID",
            F.col("TransactionID").cast("long"),
        )

    if "TransactionDT" in tx.columns:
        tx = tx.withColumn(
            "TransactionDT",
            F.col("TransactionDT").cast("long"),
        )

    if "TransactionAmt" in tx.columns:
        tx = tx.withColumn(
            "TransactionAmt",
            F.col("TransactionAmt").cast("double"),
        )

    if "TransactionID" in tx.columns:
        tx = tx.dropDuplicates(["TransactionID"])

    if not identity_path:
        return tx

    if not Path(identity_path).exists():
        raise FileNotFoundError(
            f"Identity file not found: {identity_path}"
        )

    identity = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(identity_path)
    )

    if "TransactionID" not in identity.columns:
        raise ValueError(
            "Identity dataset must contain TransactionID."
        )

    identity = identity.withColumn(
        "TransactionID",
        F.col("TransactionID").cast("long"),
    )

    identity = identity.dropDuplicates(["TransactionID"])

    return tx.join(
        F.broadcast(identity),
        on="TransactionID",
        how="left",
    )


def clean_transactions(df: DataFrame) -> DataFrame:

    out = df

    if "TransactionAmt" in out.columns:
        out = out.filter(
            F.col("TransactionAmt").isNotNull()
        )

        out = out.filter(
            F.col("TransactionAmt") >= 0
        )

    if "TransactionDT" in out.columns:
        out = out.filter(
            F.col("TransactionDT").isNotNull()
        )

    if "card1" in out.columns:
        out = out.withColumn(
            "card_id",
            F.col("card1").cast("string"),
        )

    return out


def add_historical_features(df: DataFrame) -> DataFrame:
    """
    Create historical card-level features.

    TransactionDT is the IEEE-CIS relative event-time counter in seconds.

    Features created:
    - card_transaction_count_before
    - card_amount_sum_before
    - card_amount_mean_before
    - card_amount_max_before
    - card_fraud_count_before
    - previous_transaction_time
    - time_since_previous_transaction
    """

    if "card_id" not in df.columns:
        return df

    if "TransactionDT" not in df.columns:
        return df

    if "TransactionAmt" not in df.columns:
        return df

    # Keep transactions belonging to the same card together.
    df = df.repartition("card_id")

    # ---------------------------------------------------------
    # Historical window
    # ---------------------------------------------------------
    #
    # This window contains all previous transactions for the
    # same card, excluding the current transaction.
    #
    historical_window = (
        Window
        .partitionBy("card_id")
        .orderBy("TransactionDT")
        .rowsBetween(
            Window.unboundedPreceding,
            -1,
        )
    )

    # ---------------------------------------------------------
    # Previous transaction window
    # ---------------------------------------------------------
    #
    # lag(1) gives the TransactionDT of the immediately
    # previous transaction for the same card.
    #
    previous_window = (
        Window
        .partitionBy("card_id")
        .orderBy("TransactionDT")
    )

    out = (
        df
        # -----------------------------------------------------
        # Existing historical features
        # -----------------------------------------------------
        .withColumn(
            "card_transaction_count_before",
            F.count("*").over(historical_window),
        )
        .withColumn(
            "card_amount_sum_before",
            F.sum("TransactionAmt").over(historical_window),
        )
        .withColumn(
            "card_amount_mean_before",
            F.avg("TransactionAmt").over(historical_window),
        )
        .withColumn(
            "card_amount_max_before",
            F.max("TransactionAmt").over(historical_window),
        )

        # -----------------------------------------------------
        # Previous transaction feature
        # -----------------------------------------------------
        .withColumn(
            "previous_transaction_time",
            F.lag("TransactionDT", 1).over(previous_window),
        )
        .withColumn(
            "time_since_previous_transaction",
            F.when(
                F.col("previous_transaction_time").isNotNull(),
                (
                    F.col("TransactionDT")
                    - F.col("previous_transaction_time")
                ).cast("double"),
            ).otherwise(None),
        )
    )

    # ---------------------------------------------------------
    # Fraud history
    # ---------------------------------------------------------

    if "isFraud" in out.columns:
        out = out.withColumn(
            "card_fraud_count_before",
            F.sum("isFraud").over(historical_window),
        )

    return out


def aggregate_card_history(df: DataFrame) -> DataFrame:
    """
    Create card-level historical aggregation features.
    """

    if "card_id" not in df.columns:
        raise ValueError(
            "card_id column is required for card-level aggregation."
        )

    if "TransactionAmt" not in df.columns:
        raise ValueError(
            "TransactionAmt column is required for aggregation."
        )

    aggregations = [
        F.count("*").alias(
            "transaction_count"
        ),

        F.sum("TransactionAmt").alias(
            "transaction_amount"
        ),

        F.avg("TransactionAmt").alias(
            "average_transaction_amount"
        ),

        F.min("TransactionAmt").alias(
            "minimum_transaction_amount"
        ),

        F.max("TransactionAmt").alias(
            "maximum_transaction_amount"
        ),
    ]

    if "isFraud" in df.columns:
        aggregations.extend(
            [
                F.sum("isFraud").alias(
                    "fraud_count"
                ),

                (
                    F.sum("isFraud")
                    / F.count("*")
                ).alias(
                    "fraud_rate"
                ),
            ]
        )

    return (
        df
        .groupBy("card_id")
        .agg(*aggregations)
    )


def write_parquet(
    df: DataFrame,
    output_path: str,
) -> None:

    (
        df
        .repartition(8)
        .write
        .mode("overwrite")
        .parquet(output_path)
    )


def write_aggregation_parquet(
    df: DataFrame,
    output_path: str,
) -> None:

    (
        df
        .repartition(8)
        .write
        .mode("overwrite")
        .parquet(output_path)
    )


def run(
    transaction_path: str,
    identity_path: Optional[str],
    output_path: str,
    aggregation_output_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:

    spark = create_spark()

    try:

        # ---------------------------------------------------------
        # 1. Read IEEE-CIS transaction + identity data
        # ---------------------------------------------------------

        print("Reading IEEE-CIS dataset...")

        df = read_ieee_cis(
            spark,
            transaction_path,
            identity_path,
        )

        # ---------------------------------------------------------
        # 2. Limit rows for testing
        # ---------------------------------------------------------

        if limit is not None:
            df = df.limit(limit)

            print(
                f"TEST MODE: processing only {limit} transactions"
            )

        # ---------------------------------------------------------
        # 3. Clean transactions
        # ---------------------------------------------------------

        print("Cleaning transactions...")

        df = clean_transactions(df)

        # ---------------------------------------------------------
        # 4. Historical feature engineering
        # ---------------------------------------------------------

        print("Adding historical features...")

        df = add_historical_features(df)

        # ---------------------------------------------------------
        # 5. Write transaction-level output
        # ---------------------------------------------------------

        print(
            "Writing transaction-level Parquet output..."
        )

        write_parquet(
            df,
            output_path,
        )

        print(
            f"Wrote Spark batch output to: {output_path}"
        )

        # ---------------------------------------------------------
        # 6. Card-level aggregation
        # ---------------------------------------------------------

        print("Creating card-level aggregation...")

        aggregated_df = aggregate_card_history(df)

        # ---------------------------------------------------------
        # 7. Default aggregation output path
        # ---------------------------------------------------------

        if aggregation_output_path is None:
            aggregation_output_path = (
                f"{output_path}_aggregated"
            )

        # ---------------------------------------------------------
        # 8. Write aggregation output
        # ---------------------------------------------------------

        print(
            "Writing card-level aggregation..."
        )

        write_aggregation_parquet(
            aggregated_df,
            aggregation_output_path,
        )

        print(
            "Wrote card aggregation to: "
            f"{aggregation_output_path}"
        )

        # ---------------------------------------------------------
        # 9. Count aggregated cards
        # ---------------------------------------------------------

        aggregation_count = aggregated_df.count()

        print(
            f"Unique cards: {aggregation_count}"
        )

    finally:

        spark.stop()


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "IEEE-CIS historical fraud processing "
            "using Apache Spark."
        )
    )

    parser.add_argument(
        "--transactions",
        required=True,
        help="Path to train_transaction.csv",
    )

    parser.add_argument(
        "--identity",
        default=None,
        help="Path to train_identity.csv",
    )

    parser.add_argument(
        "--output",
        default="spark/output/batch",
        help="Transaction-level Parquet output path",
    )

    parser.add_argument(
        "--aggregation-output",
        default=None,
        help="Card-level aggregation Parquet output path",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only N transactions for testing",
    )

    args = parser.parse_args()

    run(
        transaction_path=args.transactions,
        identity_path=args.identity,
        output_path=args.output,
        aggregation_output_path=args.aggregation_output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

