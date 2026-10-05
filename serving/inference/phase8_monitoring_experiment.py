"""
inference/phase8_monitoring_experiment.py
Phase 8 — Controlled Monitoring Experiment.

Runs 100 sequential /predict requests through the FastAPI TestClient,
recording monitoring metrics (request counts, latency distribution, model version),
then triggers 10 deliberate validation errors to verify error monitoring.

Environment:      Local host, FastAPI TestClient (in-process, no network)
Sample size:      100 successful predictions + 10 error requests
Transactions:     Synthetic transactions (no raw IEEE-CIS dataset required)
Request mode:     Sequential (one at a time)
Model:            E1_LightGBM, threshold=0.616521, features=406

Output:
  benchmarks/results/phase8_monitoring_results.csv
  (Summary printed to stdout)
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from api.main import app
from inference.offline_inference import OfflineInferenceEngine, E1_DECISION_THRESHOLD
from monitoring.api_monitor import monitor
from monitoring.model_metadata import MODEL_METADATA


OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "results"
OUTPUT_CSV = OUTPUT_DIR / "phase8_monitoring_results.csv"

N_SUCCESS_REQUESTS = 100
N_ERROR_REQUESTS   = 10


def run_monitoring_experiment() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Phase 8 — Monitoring Experiment")
    print(f"Model:          {MODEL_METADATA['model_name']} (version {MODEL_METADATA['model_version']})")
    print(f"Threshold:      {MODEL_METADATA['decision_threshold']}")
    print(f"Feature count:  {MODEL_METADATA['feature_count']}")
    print(f"Sample size:    {N_SUCCESS_REQUESTS} successful + {N_ERROR_REQUESTS} error requests")
    print(f"Request mode:   Sequential (TestClient, in-process)")
    print("=" * 65)

    # ── Load engine ───────────────────────────────────────────────────────────
    if not isinstance(getattr(app.state, "engine", None), OfflineInferenceEngine):
        app.state.engine = OfflineInferenceEngine(
            model_path="models/E1/model.txt",
            preprocessor_path="models/E1/preprocessing.joblib",
            feature_names_path="models/E1/feature_names.json",
            threshold=E1_DECISION_THRESHOLD,
        )

    # ── Reset monitor so experiment starts clean ──────────────────────────────
    monitor.reset()

    rows = []  # For CSV output

    with TestClient(app) as client:
        # ── Step 1: 100 Successful /predict requests ──────────────────────────
        print(f"\n[Step 1] Sending {N_SUCCESS_REQUESTS} sequential /predict requests...")
        t_exp_start = time.perf_counter()

        for i in range(N_SUCCESS_REQUESTS):
            tx_id = 800000 + i
            # Vary TransactionAmt and TransactionDT to get a spread of predictions
            payload = {
                "TransactionID": tx_id,
                "TransactionDT":  14000000 + (i * 1000),
                "TransactionAmt": 10.0 + (i * 5.0),
                "ProductCD":      ["W", "C", "H", "S", "R"][i % 5],
                "card1":          1000 + (i * 7),
            }
            t0 = time.perf_counter()
            resp = client.post("/predict", json=payload)
            wall_ms = (time.perf_counter() - t0) * 1000.0

            data = resp.json()
            row = {
                "request_index":       i + 1,
                "transaction_id":      tx_id,
                "http_status":         resp.status_code,
                "fraud_probability":   round(data.get("fraud_probability", -1), 6),
                "decision":            data.get("decision", "ERROR"),
                "model_version":       MODEL_METADATA["model_version"],
                "total_inference_ms":  round(data.get("total_inference_time_ms", -1), 3),
                "wall_time_ms":        round(wall_ms, 3),
                "request_type":        "success",
            }
            rows.append(row)

        t_exp_success_elapsed = (time.perf_counter() - t_exp_start) * 1000.0
        print(f"  Completed in {t_exp_success_elapsed:.1f} ms total.")

        # ── Step 2: 10 deliberate validation error requests ───────────────────
        print(f"\n[Step 2] Sending {N_ERROR_REQUESTS} deliberate validation error requests...")
        error_scenarios = [
            {"transactions": []},           # empty batch → 422
            {"transactions": "bad_string"}, # wrong type  → 422
            {},                             # missing field → 422
            {"transactions": [{"TransactionID": "oops", "TransactionAmt": "not_a_float"}]},
            {"transactions": []},
            {"transactions": "invalid"},
            {},
            {"transactions": []},
            {"transactions": "bad"},
            {},
        ]
        for j, bad_payload in enumerate(error_scenarios):
            resp = client.post("/predict/batch", json=bad_payload)
            row = {
                "request_index":       N_SUCCESS_REQUESTS + j + 1,
                "transaction_id":      "N/A",
                "http_status":         resp.status_code,
                "fraud_probability":   -1,
                "decision":            "ERROR",
                "model_version":       MODEL_METADATA["model_version"],
                "total_inference_ms":  -1,
                "wall_time_ms":        -1,
                "request_type":        "error",
            }
            rows.append(row)

        print(f"  {N_ERROR_REQUESTS} error requests sent.")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Output] Results written to: {OUTPUT_CSV}")

    # ── Compute summary ───────────────────────────────────────────────────────
    summary = monitor.get_summary()

    success_rows = [r for r in rows if r["request_type"] == "success"]
    decisions = [r["decision"] for r in success_rows]
    fraud_count = decisions.count("FRAUD")
    legit_count = decisions.count("LEGIT")
    model_versions_seen = set(r["model_version"] for r in rows)

    print()
    print("=" * 65)
    print("PHASE 8 MONITORING EXPERIMENT RESULTS")
    print("=" * 65)
    print(f"  Environment:             Local host, FastAPI TestClient")
    print(f"  Model:                   {MODEL_METADATA['model_name']}")
    print(f"  Model version:           {MODEL_METADATA['model_version']}")
    print(f"  Decision threshold:      {MODEL_METADATA['decision_threshold']}")
    print(f"  Feature count:           {MODEL_METADATA['feature_count']}")
    print()
    print("  Request Counts (from MonitoringState):")
    print(f"    Total requests:        {summary['total_requests']}")
    print(f"    Successful (2xx):      {summary['successful_requests']}")
    print(f"    Failed (4xx+5xx):      {summary['failed_requests']}")
    print(f"    HTTP 2xx count:        {summary['http_2xx_count']}")
    print(f"    HTTP 4xx count:        {summary['http_4xx_count']}")
    print(f"    HTTP 5xx count:        {summary['http_5xx_count']}")
    print()
    print("  Prediction Breakdown (successful requests only):")
    print(f"    FRAUD decisions:       {fraud_count}")
    print(f"    LEGIT decisions:       {legit_count}")
    print()
    print("  Inference Latency (from MonitoringState, N={:d}):".format(summary['latency_sample_count']))
    print(f"    Min:                   {summary['latency_min_ms']} ms")
    print(f"    Mean:                  {summary['latency_mean_ms']} ms")
    print(f"    P50 (Median):          {summary['latency_p50_ms']} ms")
    print(f"    P95:                   {summary['latency_p95_ms']} ms")
    print(f"    P99:                   {summary['latency_p99_ms']} ms")
    print(f"    Max:                   {summary['latency_max_ms']} ms")
    print()
    print("  Model Version Tracking:")
    print(f"    model_version values observed: {model_versions_seen}")
    print(f"    All match MODEL_METADATA:      {model_versions_seen == {MODEL_METADATA['model_version']}}")
    print()
    print("  Error Monitoring:")
    print(f"    Error requests sent:   {N_ERROR_REQUESTS}")
    print(f"    Recorded as failed:    {summary['failed_requests']}")
    print(f"    Error detection rate:  {summary['failed_requests'] / N_ERROR_REQUESTS * 100:.1f}%")
    print("=" * 65)
    print(f"\nPhase 8 Monitoring Experiment: COMPLETE")

    return summary, rows


if __name__ == "__main__":
    run_monitoring_experiment()
