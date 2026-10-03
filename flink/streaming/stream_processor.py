import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# ============================================================
# 1. PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# 2. PYFLINK PYTHON ENVIRONMENT
# ============================================================

PYTHON_EXE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "flink_py311",
        "Scripts",
        "python.exe",
    )
)

os.environ["PYFLINK_PYTHON"] = PYTHON_EXE
os.environ["PYFLINK_CLIENT_EXECUTABLE"] = PYTHON_EXE
os.environ["PYTHON_EXECUTABLE"] = PYTHON_EXE


# ============================================================
# 3. KAFKA JAR FILES
# ============================================================

KAFKA_JAR_DIR = ROOT / "flink" / "jars"

KAFKA_CONNECTOR_JAR = (
    KAFKA_JAR_DIR /
    "flink-connector-kafka-3.3.0-1.20.jar"
)

KAFKA_CLIENT_JAR = (
    KAFKA_JAR_DIR /
    "kafka-clients-3.8.1.jar"
)

if not KAFKA_CONNECTOR_JAR.exists():
    raise FileNotFoundError(
        f"\nKafka connector JAR not found:\n"
        f"{KAFKA_CONNECTOR_JAR}\n"
    )

if not KAFKA_CLIENT_JAR.exists():
    raise FileNotFoundError(
        f"\nKafka client JAR not found:\n"
        f"{KAFKA_CLIENT_JAR}\n"
    )


print("=" * 70)
print("[Flink] Project root:")
print(ROOT)
print("[Flink] Python executable:")
print(PYTHON_EXE)
print(
    "[Flink] Python executable exists:",
    "YES" if os.path.exists(PYTHON_EXE) else "NO",
)
print("[Flink] Kafka connector JAR:")
print(KAFKA_CONNECTOR_JAR)
print("[Flink] Kafka client JAR:")
print(KAFKA_CLIENT_JAR)
print("=" * 70)


# ============================================================
# 4. PYFLINK IMPORTS
# ============================================================

from pyflink.common import Types
from pyflink.common.time import Time
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.window import SlidingEventTimeWindows
from pyflink.datastream.functions import AggregateFunction


# ============================================================
# 5. PROJECT IMPORTS
# ============================================================

from flink.streaming.transaction_schema import Transaction
from flink.streaming.watermark import (
    transaction_watermark_strategy,
)
from flink.streaming.kafka_source import (
    build_kafka_source,
)
from flink.features.transaction_features import (
    amount_bucket,
)


# ============================================================
# 6. BASIC WINDOW ACCUMULATOR
# ============================================================

@dataclass
class FraudAccumulator:
    user_id: str
    transaction_count: int
    total_amount: float


# ============================================================
# 7. BASIC WINDOW FEATURES
# ============================================================

@dataclass
class FraudWindowFeatures:
    user_id: str
    transaction_count: int
    total_amount: float
    average_amount: float
    amount_bucket: str


# ============================================================
# 8. COMBINED FRAUD FEATURES
# ============================================================

@dataclass
class CombinedFraudFeatures:

    user_id: str

    transaction_count_5m: int
    total_amount_5m: float
    average_amount_5m: float

    transaction_count_10m: int
    total_amount_10m: float
    average_amount_10m: float

    transaction_velocity_ratio: float
    amount_velocity_ratio: float

    amount_bucket_10m: str


# ============================================================
# 9. COMBINED ACCUMULATOR
# ============================================================

@dataclass
class CombinedAccumulator:
    user_id: str
    transactions: list


# ============================================================
# 10. NORMAL WINDOW AGGREGATOR
# ============================================================

class FraudWindowAggregator(AggregateFunction):

    def create_accumulator(self):

        return FraudAccumulator(
            user_id="",
            transaction_count=0,
            total_amount=0.0,
        )

    def add(self, transaction, accumulator):

        user_id = accumulator.user_id

        if not user_id:
            user_id = transaction.user_id

        return FraudAccumulator(
            user_id=user_id,
            transaction_count=(
                accumulator.transaction_count + 1
            ),
            total_amount=(
                accumulator.total_amount
                + float(transaction.amount)
            ),
        )

    def get_result(self, accumulator):

        if accumulator.transaction_count > 0:
            average_amount = (
                accumulator.total_amount
                / accumulator.transaction_count
            )
        else:
            average_amount = 0.0

        return FraudWindowFeatures(
            user_id=accumulator.user_id,
            transaction_count=(
                accumulator.transaction_count
            ),
            total_amount=(
                accumulator.total_amount
            ),
            average_amount=average_amount,
            amount_bucket=(
                amount_bucket(
                    accumulator.total_amount
                )
            ),
        )

    def merge(self, accumulator_a, accumulator_b):

        if accumulator_a.user_id:
            user_id = accumulator_a.user_id
        else:
            user_id = accumulator_b.user_id

        return FraudAccumulator(
            user_id=user_id,
            transaction_count=(
                accumulator_a.transaction_count
                + accumulator_b.transaction_count
            ),
            total_amount=(
                accumulator_a.total_amount
                + accumulator_b.total_amount
            ),
        )


# ============================================================
# 11. EVENT TIME TO SECONDS
# ============================================================

def event_time_to_seconds(event_time):

    if isinstance(event_time, (int, float)):
        return float(event_time)

    if isinstance(event_time, datetime):
        return event_time.timestamp()

    if isinstance(event_time, str):

        value = event_time.strip()

        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        dt = datetime.fromisoformat(value)

        return dt.timestamp()

    raise ValueError(
        "Unsupported event_time type: "
        f"{type(event_time)}"
    )


# ============================================================
# 12. COMBINED VELOCITY AGGREGATOR
# ============================================================

class CombinedVelocityAggregator(AggregateFunction):

    def create_accumulator(self):

        return CombinedAccumulator(
            user_id="",
            transactions=[],
        )

    def add(self, transaction, accumulator):

        user_id = accumulator.user_id

        if not user_id:
            user_id = transaction.user_id

        timestamp = event_time_to_seconds(
            transaction.event_time
        )

        new_transactions = list(
            accumulator.transactions
        )

        new_transactions.append(
            (
                timestamp,
                float(transaction.amount),
            )
        )

        return CombinedAccumulator(
            user_id=user_id,
            transactions=new_transactions,
        )

    def get_result(self, accumulator):

        transactions = accumulator.transactions

        if not transactions:

            return CombinedFraudFeatures(
                user_id=accumulator.user_id,

                transaction_count_5m=0,
                total_amount_5m=0.0,
                average_amount_5m=0.0,

                transaction_count_10m=0,
                total_amount_10m=0.0,
                average_amount_10m=0.0,

                transaction_velocity_ratio=0.0,
                amount_velocity_ratio=0.0,

                amount_bucket_10m="LOW",
            )

        latest_timestamp = max(
            timestamp
            for timestamp, amount
            in transactions
        )

        five_minute_boundary = (
            latest_timestamp - 5 * 60
        )

        transactions_5m = [
            (timestamp, amount)
            for timestamp, amount
            in transactions
            if timestamp >= five_minute_boundary
        ]

        count_5m = len(transactions_5m)

        total_5m = sum(
            amount
            for timestamp, amount
            in transactions_5m
        )

        if count_5m > 0:
            average_5m = total_5m / count_5m
        else:
            average_5m = 0.0

        count_10m = len(transactions)

        total_10m = sum(
            amount
            for timestamp, amount
            in transactions
        )

        if count_10m > 0:
            average_10m = total_10m / count_10m
        else:
            average_10m = 0.0

        if count_10m > 0:
            transaction_velocity_ratio = (
                count_5m / count_10m
            )
        else:
            transaction_velocity_ratio = 0.0

        if total_10m > 0:
            amount_velocity_ratio = (
                total_5m / total_10m
            )
        else:
            amount_velocity_ratio = 0.0

        return CombinedFraudFeatures(
            user_id=accumulator.user_id,

            transaction_count_5m=count_5m,
            total_amount_5m=total_5m,
            average_amount_5m=average_5m,

            transaction_count_10m=count_10m,
            total_amount_10m=total_10m,
            average_amount_10m=average_10m,

            transaction_velocity_ratio=(
                transaction_velocity_ratio
            ),

            amount_velocity_ratio=(
                amount_velocity_ratio
            ),

            amount_bucket_10m=(
                amount_bucket(total_10m)
            ),
        )

    def merge(self, accumulator_a, accumulator_b):

        if accumulator_a.user_id:
            user_id = accumulator_a.user_id
        else:
            user_id = accumulator_b.user_id

        combined_transactions = (
            list(accumulator_a.transactions)
            + list(accumulator_b.transactions)
        )

        return CombinedAccumulator(
            user_id=user_id,
            transactions=combined_transactions,
        )


# ============================================================
# 13. PARSE KAFKA TRANSACTION
# ============================================================

def parse_transaction(raw):

    data = json.loads(raw)

    # Support different possible user ID field names
    if "user_id" not in data:
        data["user_id"] = (
            data.get("userId")
            or data.get("customer_id")
            or data.get("customerId")
            or data.get("uid")
            or "UNKNOWN"
        )

    # Support different possible transaction ID field names
    if "transaction_id" not in data:
        data["transaction_id"] = (
            data.get("transactionId")
            or data.get("id")
            or data.get("txn_id")
            or data.get("txnId")
            or f"txn-{data.get('user_id', 'unknown')}-{data.get('amount', 0)}"
        )

    # Support timestamp field variations
    if "event_time" not in data:
        data["event_time"] = (
            data.get("timestamp")
            or data.get("transaction_time")
            or data.get("time")
            or datetime.now().isoformat()
        )

    # Support amount field
    if "amount" not in data:
        data["amount"] = (
            data.get("transaction_amount")
            or data.get("value")
            or 0.0
        )

    return Transaction.from_dict(data)
    
# ============================================================
# 14. FORMAT 5-MINUTE RESULT
# ============================================================

def format_result_5m(features):

    return (
        "[5-MIN WINDOW] "
        f"user={features.user_id}, "
        f"transaction_count_5m="
        f"{features.transaction_count}, "
        f"total_amount_5m="
        f"{features.total_amount}, "
        f"average_amount_5m="
        f"{features.average_amount:.2f}, "
        f"amount_bucket_5m="
        f"{features.amount_bucket}"
    )


# ============================================================
# 15. FORMAT 10-MINUTE RESULT
# ============================================================

def format_result_10m(features):

    return (
        "[10-MIN WINDOW] "
        f"user={features.user_id}, "
        f"transaction_count_10m="
        f"{features.transaction_count}, "
        f"total_amount_10m="
        f"{features.total_amount}, "
        f"average_amount_10m="
        f"{features.average_amount:.2f}, "
        f"amount_bucket_10m="
        f"{features.amount_bucket}"
    )


# ============================================================
# 16. FORMAT VELOCITY FEATURES
# ============================================================

def format_combined_features(features):

    return (
        "[VELOCITY FEATURES] "
        f"user={features.user_id}, "
        f"transaction_count_5m="
        f"{features.transaction_count_5m}, "
        f"total_amount_5m="
        f"{features.total_amount_5m:.2f}, "
        f"average_amount_5m="
        f"{features.average_amount_5m:.2f}, "
        f"transaction_count_10m="
        f"{features.transaction_count_10m}, "
        f"total_amount_10m="
        f"{features.total_amount_10m:.2f}, "
        f"average_amount_10m="
        f"{features.average_amount_10m:.2f}, "
        f"transaction_velocity_ratio="
        f"{features.transaction_velocity_ratio:.3f}, "
        f"amount_velocity_ratio="
        f"{features.amount_velocity_ratio:.3f}, "
        f"amount_bucket_10m="
        f"{features.amount_bucket_10m}"
    )


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print("[Flink Streaming] Starting...")

    print("[Flink] Creating execution environment...")

    env = (
        StreamExecutionEnvironment
        .get_execution_environment()
    )

    env.set_parallelism(1)

    # --------------------------------------------------------
    # Python executable
    # --------------------------------------------------------

    try:

        env.set_python_executable(
            PYTHON_EXE
        )

        print(
            "[Flink] set_python_executable(): SUCCESS"
        )

    except Exception as exc:

        print(
            "[Flink] WARNING: "
            "set_python_executable failed:"
        )

        print(exc)

    # --------------------------------------------------------
    # Add Kafka JARs to Flink classpath
    # --------------------------------------------------------

    connector_uri = (
        "file:///"
        + str(KAFKA_CONNECTOR_JAR)
        .replace("\\", "/")
    )

    client_uri = (
        "file:///"
        + str(KAFKA_CLIENT_JAR)
        .replace("\\", "/")
    )

    env.add_jars(connector_uri)
    env.add_jars(client_uri)

    print(
        "[Flink] Kafka connector JAR added:"
    )
    print(KAFKA_CONNECTOR_JAR)

    print(
        "[Flink] Kafka client JAR added:"
    )
    print(KAFKA_CLIENT_JAR)

    # ========================================================
    # STEP 1: CREATE KAFKA SOURCE
    # ========================================================

    print("[Flink] Creating Kafka source...")

    kafka_source = build_kafka_source(
        bootstrap_servers="localhost:9092",
        topic="fraud-transactions",
        group_id="flink-fraud-processing",
    )

    # ========================================================
    # STEP 2: KAFKA → FLINK
    # ========================================================

    print("[Flink] Connecting to Kafka...")
    print("[Flink] Topic: fraud-transactions")
    print("[Flink] Broker: localhost:9092")

    stream = env.from_source(
        kafka_source,
        transaction_watermark_strategy(),
        "Kafka Fraud Transaction Source",
    )

    # ========================================================
    # STEP 3: JSON → TRANSACTION
    # ========================================================

    print("[Flink] Parsing transactions...")

    transactions = stream.map(
        parse_transaction,
        output_type=Types.PICKLED_BYTE_ARRAY(),
    )

    # ========================================================
    # STEP 4: WATERMARKS
    # ========================================================

    print(
        "[Flink] Assigning timestamps "
        "and watermarks..."
    )

    timed_transactions = (
        transactions
        .assign_timestamps_and_watermarks(
            transaction_watermark_strategy()
        )
    )

    # ========================================================
    # STEP 5: KEY BY USER
    # ========================================================

    print(
        "[Flink] Keying transactions "
        "by user_id..."
    )

    keyed_transactions = (
        timed_transactions
        .key_by(
            lambda tx: tx.user_id,
            key_type=Types.STRING(),
        )
    )

    # ========================================================
    # STEP 6: 5-MINUTE WINDOW
    # ========================================================

    print(
        "[Flink] Creating 5-minute window..."
    )

    window_5m = (
        keyed_transactions
        .window(
            SlidingEventTimeWindows.of(
                Time.minutes(5),
                Time.minutes(5),
            )
        )
    )

    aggregated_5m = (
        window_5m
        .aggregate(
            FraudWindowAggregator()
        )
    )

    result_5m = (
        aggregated_5m
        .map(
            format_result_5m,
            output_type=Types.STRING(),
        )
    )

    # ========================================================
    # STEP 7: 10-MINUTE WINDOW
    # ========================================================

    print(
        "[Flink] Creating 10-minute window..."
    )

    window_10m = (
        keyed_transactions
        .window(
            SlidingEventTimeWindows.of(
                Time.minutes(10),
                Time.minutes(5),
            )
        )
    )

    aggregated_10m = (
        window_10m
        .aggregate(
            FraudWindowAggregator()
        )
    )

    result_10m = (
        aggregated_10m
        .map(
            format_result_10m,
            output_type=Types.STRING(),
        )
    )

    # ========================================================
    # STEP 8: VELOCITY WINDOW
    # ========================================================

    print(
        "[Flink] Creating velocity window..."
    )

    velocity_window = (
        keyed_transactions
        .window(
            SlidingEventTimeWindows.of(
                Time.minutes(10),
                Time.minutes(5),
            )
        )
    )

    # ========================================================
    # STEP 9: VELOCITY AGGREGATION
    # ========================================================

    print(
        "[Flink] Creating velocity features..."
    )

    velocity_features = (
        velocity_window
        .aggregate(
            CombinedVelocityAggregator()
        )
    )

    velocity_result = (
        velocity_features
        .map(
            format_combined_features,
            output_type=Types.STRING(),
        )
    )

    # ========================================================
    # STEP 10: OUTPUT
    # ========================================================

    result_5m.print()
    result_10m.print()
    velocity_result.print()

    # ========================================================
    # STEP 11: EXECUTE
    # ========================================================

    print("=" * 70)
    print(
        "[Flink Streaming] Executing Kafka job..."
    )
    print(
        "[Flink Streaming] Waiting for transactions..."
    )
    print(
        "[Flink Streaming] Kafka topic: "
        "fraud-transactions"
    )
    print(
        "[Flink Streaming] Kafka broker: "
        "localhost:9092"
    )
    print("=" * 70)

    env.execute(
        "Flink Kafka Real-Time Fraud "
        "Feature Processing"
    )


# ============================================================
# 18. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()