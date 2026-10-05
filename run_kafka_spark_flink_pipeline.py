"""
End-to-End Orchestrator: Kafka -> Spark -> Flink Integration Pipeline
Launches Member 1 (Kafka Ingestion Producer), Member 2 (Spark Structured Streaming Scorer),
and Member 2 (Flink Stateful CEP Velocity Anomaly Detector) concurrently.
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "kafka" / "transaction_generator"))
sys.path.insert(0, str(base_dir / "kafka" / "producer"))
sys.path.insert(0, str(base_dir / "kafka" / "consumer"))

# pyrefly: ignore [missing-import]
# type: ignore
from flink_cep_consumer import run_flink_cep_consumer
# pyrefly: ignore [missing-import]
# type: ignore
from spark_streaming_consumer import run_standalone_streaming


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run End-to-End Kafka -> Spark & Flink Streaming Integration Pipeline"
    )
    parser.add_argument("--broker", default="localhost:9092", help="Kafka broker (default: localhost:9092)")
    parser.add_argument("--topic", default="ieee_cis_transactions", help="Kafka topic (default: ieee_cis_transactions)")
    parser.add_argument("--records", type=int, default=1000, help="Number of records to stream for integration test (default: 1000)")
    parser.add_argument("--rate", type=int, default=500, help="Producer replay rate (tx/sec, default: 500)")
    parser.add_argument("--mode", choices=["all", "spark", "flink", "producer"], default="all", help="Integration mode to run")
    return parser.parse_args()


def producer_worker(broker, topic, records, rate):
    """Streams transactions from dataset into Kafka broker."""
    # pyrefly: ignore [missing-import]
    # type: ignore
    from ieee_cis_replay_producer import find_default_dataset, stream_dataset

    print(f"[PRODUCER PROCESS] Starting IEEE-CIS dataset ingestion into topic '{topic}'...")
    dataset_path = find_default_dataset()
    stream_dataset(
        broker=broker,
        topic=topic,
        dataset_path=dataset_path,
        rate=rate,
        max_count=records,
        acks="all",
        compact=True,
    )


def spark_worker(broker, topic, max_records):
    """Executes Spark Structured Streaming Micro-Batch ML Scoring."""
    print(f"[SPARK PROCESS] Initializing Spark ML inference consumer on topic '{topic}'...")
    run_standalone_streaming(
        broker=broker,
        topic=topic,
        trigger_seconds=2,
        max_records=max_records,
    )


def flink_worker(broker, topic, max_records):
    """Executes Flink CEP & Stateful Velocity Scoring Engine."""
    print(f"[FLINK PROCESS] Initializing Flink CEP Velocity consumer on topic '{topic}'...")
    run_flink_cep_consumer(
        broker=broker,
        topic=topic,
        window_seconds=60,
        velocity_threshold=3,
        amount_burst_threshold=5000.0,
        max_records=max_records,
    )


def main():
    args = parse_arguments()

    print("=" * 85)
    print("  KAFKA -> SPARK & FLINK END-TO-END STREAMING INTEGRATION PIPELINE")
    print("=" * 85)
    print(f"Broker                 : {args.broker}")
    print(f"Ingestion Topic        : {args.topic}")
    print(f"Target Records Count   : {args.records:,}")
    print(f"Streaming Rate         : {args.rate} tx/sec")
    print(f"Execution Mode         : {args.mode.upper()}")
    print("=" * 85)

    if args.mode == "producer":
        producer_worker(args.broker, args.topic, args.records, args.rate)
    elif args.mode == "spark":
        spark_worker(args.broker, args.topic, args.records)
    elif args.mode == "flink":
        flink_worker(args.broker, args.topic, args.records)
    else:
        # Run concurrently using multiprocessing
        processes = []

        # 1. Start Flink CEP Consumer
        p_flink = mp.Process(target=flink_worker, args=(args.broker, args.topic, args.records))
        p_flink.start()
        processes.append(p_flink)

        # 2. Start Spark Structured Streaming Consumer
        p_spark = mp.Process(target=spark_worker, args=(args.broker, args.topic, args.records))
        p_spark.start()
        processes.append(p_spark)

        # Give consumers 2 seconds to establish partitions & consumer group join
        time.sleep(2)

        # 3. Start Ingestion Producer Replay
        p_prod = mp.Process(target=producer_worker, args=(args.broker, args.topic, args.records, args.rate))
        p_prod.start()
        processes.append(p_prod)

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\n[INFO] Terminating integration pipeline processes...")
            for p in processes:
                p.terminate()

    print("\n[SUCCESS] Integration Pipeline Execution Complete.")


if __name__ == "__main__":
    main()
