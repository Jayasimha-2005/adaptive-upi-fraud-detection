"""
tools/auto_api_test.py
Automatic IEEE-CIS Test Transaction Client for the certified Phase 1 FastAPI Fraud Serving API.

Usage Examples:
    # 1. Test single transaction by ID:
    python tools/auto_api_test.py --transaction-id 3663602

    # 2. Test 10 sequential transactions from dataset:
    python tools/auto_api_test.py --count 10

    # 3. Continuous Watch Mode (streams 1 transaction per second to live dashboard):
    python tools/auto_api_test.py --watch --interval 1

    # 4. Test a specific list of transaction IDs:
    python tools/auto_api_test.py --transaction-ids 3663549 3663550 3663602

    # 5. Save results to CSV:
    python tools/auto_api_test.py --count 100 --output results.csv

    # 6. Specify custom API URL or Dataset path:
    python tools/auto_api_test.py --count 5 --api-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

# Default search paths for IEEE-CIS test_transaction.csv
DEFAULT_DATASET_LOCATIONS = [
    Path("d:/fraud-model-serving/Datasets/IEEE CIS-20260829T103704Z-1-001/IEEE CIS/test_transaction.csv"),
    Path("d:/adaptive-upi-fraud-detection/IEEE CIS-20260829T103704Z-1-001/IEEE CIS/test_transaction.csv"),
    Path("./data/test_transaction.csv"),
    Path("./test_transaction.csv"),
    Path("../Datasets/IEEE CIS-20260829T103704Z-1-001/IEEE CIS/test_transaction.csv"),
]


def find_dataset_file(explicit_path: Optional[str] = None) -> Path:
    """Resolve the location of the IEEE-CIS test_transaction.csv dataset."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified dataset file not found: {explicit_path}")

    for candidate in DEFAULT_DATASET_LOCATIONS:
        if candidate.is_file():
            return candidate

    # Search current and parent directory for test_transaction.csv
    for root in [Path("."), Path(".."), Path("d:/")]:
        try:
            matches = list(root.glob("**/test_transaction.csv"))
            if matches:
                return matches[0]
        except Exception:
            continue

    raise FileNotFoundError(
        "Could not automatically locate test_transaction.csv. "
        "Please provide the path using --dataset-path <path>"
    )


def check_api_health(api_url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Verify that the FastAPI serving service is reachable and healthy."""
    health_url = f"{api_url.rstrip('/')}/health"
    req = urllib.request.Request(health_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data
            raise RuntimeError(f"Health check returned unexpected HTTP {resp.status}")
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as err:
        print("\n" + "=" * 70)
        print(" [!] API IS NOT REACHABLE")
        print("=" * 70)
        print(f" Target URL : {health_url}")
        print(f" Error      : {err}")
        print("\n Action Required:")
        print("   Start the Docker FastAPI container first:")
        print("   docker run -d --name adaptive-fraud-serving -p 8000:8000 adaptive-fraud-serving:v1")
        print("=" * 70 + "\n")
        sys.exit(1)


def reset_test_session(api_url: str, timeout: float = 5.0) -> str:
    """
    Call POST /api/dashboard/session/reset to initialize a clean test session.
    """
    reset_url = f"{api_url.rstrip('/')}/api/dashboard/session/reset"
    req = urllib.request.Request(reset_url, data=b"{}", method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("session_id", "UNKNOWN")
            raise RuntimeError(f"Session reset returned unexpected HTTP {resp.status}")
    except Exception as err:
        print(f"[!] Warning: Failed to reset dashboard session: {err}")
        return "UNKNOWN"


def row_to_payload(row: Dict[str, str]) -> Dict[str, Any]:
    """
    Convert a CSV row from test_transaction.csv into a clean dictionary payload
    compatible with the FastAPI /predict endpoint schema (TransactionRequest).
    Ensures no isFraud target column is included.
    """
    payload: Dict[str, Any] = {}
    for col, val in row.items():
        if col in ("isFraud", "is_fraud", "fraud_bool"):
            continue  # Strict target isolation
        if val == "" or val is None:
            continue
        try:
            # Parse numeric fields
            if "." in val or "e" in val.lower():
                payload[col] = float(val)
            else:
                payload[col] = int(val)
        except ValueError:
            payload[col] = val

    return payload


def stream_test_transactions(
    csv_path: Path,
    target_ids: Optional[Set[int]] = None,
    limit: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Stream selected transactions from test_transaction.csv without loading entire file into RAM."""
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            tx_id_str = row.get("TransactionID")
            if not tx_id_str:
                continue
            tx_id = int(tx_id_str)

            if target_ids is not None:
                if tx_id in target_ids:
                    yield row_to_payload(row)
                    count += 1
                    if len(target_ids) == count:
                        break
            else:
                yield row_to_payload(row)
                count += 1
                if limit is not None and count >= limit:
                    break


def send_prediction_request(
    api_url: str,
    payload: Dict[str, Any],
    timeout: float = 15.0,
) -> Tuple[int, Optional[Dict[str, Any]], float, Optional[str]]:
    """
    Send POST /predict request to the live FastAPI server.
    Returns (http_status, response_json, latency_ms, error_msg).
    """
    predict_url = f"{api_url.rstrip('/')}/predict"
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(predict_url, data=data_bytes, method="POST")
    req.add_header("Content-Type", "application/json")

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            body = resp.read().decode("utf-8")
            res_json = json.loads(body)
            return resp.status, res_json, latency_ms, None
    except urllib.error.HTTPError as he:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        err_body = he.read().decode("utf-8", errors="replace")
        return he.code, None, latency_ms, f"HTTP Error {he.code}: {err_body}"
    except Exception as ex:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return 0, None, latency_ms, f"Connection/Request Exception: {str(ex)}"


def main():
    parser = argparse.ArgumentParser(
        description="Automatic Test Transaction Client for Certified Phase 1 E1 FastAPI Fraud Serving API."
    )
    parser.add_argument(
        "--transaction-id",
        type=int,
        default=None,
        help="Test a single transaction by ID (e.g. 3663602)",
    )
    parser.add_argument(
        "--transaction-ids",
        type=int,
        nargs="+",
        default=None,
        help="Test a list of specific transaction IDs (e.g. 3663549 3663550 3663602)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Test N deterministic sequential transactions from test_transaction.csv",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuous streaming mode: send transactions repeatedly with delay interval",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Interval in seconds between requests in --watch mode (default: 1.0)",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Reset current dashboard test session before sending transactions",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Explicit path to IEEE-CIS test_transaction.csv",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="FastAPI server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV output file to save results (e.g. results.csv)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP request timeout in seconds (default: 15.0)",
    )

    args = parser.parse_args()

    # Determine execution mode
    target_ids: Optional[Set[int]] = None
    count_limit: Optional[int] = None

    if args.transaction_id is not None:
        target_ids = {args.transaction_id}
    elif args.transaction_ids is not None:
        target_ids = set(args.transaction_ids)
    elif args.count is not None:
        count_limit = args.count
    elif args.watch:
        count_limit = None  # continuous stream
    else:
        # Default fallback: test first 5 transactions
        count_limit = 5

    # 1. Check API Health
    print("=" * 75)
    print("  AUTOMATIC API TEST CLIENT -- IEEE-CIS TEST TRANSACTIONS")
    print("=" * 75)
    print(f"Target API Endpoint : {args.api_url.rstrip('/')}/predict")
    print(f"Checking API Health : {args.api_url.rstrip('/')}/health ...", end=" ")
    health_data = check_api_health(args.api_url, timeout=args.timeout)
    print(f"[ONLINE] Model: {health_data.get('model_name')}, Threshold: {health_data.get('decision_threshold')}")

    # Session Reset if requested
    if args.new_session:
        new_sess_id = reset_test_session(args.api_url, timeout=args.timeout)
        print(f"[NEW SESSION STARTED] Dashboard Session ID: {new_sess_id}")

    # 2. Locate Dataset File
    dataset_file = find_dataset_file(args.dataset_path)
    print(f"Dataset File Path   : {dataset_file} ({dataset_file.stat().st_size / 1e6:.1f} MB)")

    # 3. Stream and send transactions
    print("-" * 75)
    results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    success_count = 0
    fail_count = 0
    legit_count = 0
    fraud_count = 0

    if args.watch:
        mode_desc = f"Continuous Watch Mode (Interval: {args.interval}s -- Press Ctrl+C to stop)"
    elif args.transaction_id:
        mode_desc = f"Single Transaction ID {args.transaction_id}"
    elif target_ids:
        mode_desc = f"Transaction IDs: {list(target_ids)}"
    else:
        mode_desc = f"First {count_limit} transactions"

    print(f"Execution Mode      : {mode_desc}")
    print("-" * 75)

    idx = 0
    try:
        for payload in stream_test_transactions(dataset_file, target_ids=target_ids, limit=count_limit):
            idx += 1
            tx_id = payload.get("TransactionID", "UNKNOWN")
            amt = payload.get("TransactionAmt", 0.0)

            status, resp, latency_ms, err = send_prediction_request(args.api_url, payload, timeout=args.timeout)
            latencies.append(latency_ms)

            if status == 200 and resp:
                success_count += 1
                prob = resp.get("fraud_probability", 0.0)
                dec = resp.get("decision", "UNKNOWN")
                if dec == "FRAUD":
                    fraud_count += 1
                    dec_str = "[FRAUD]"
                else:
                    legit_count += 1
                    dec_str = "LEGIT"

                print(
                    f" [{idx}] TxID: {tx_id:<8} | Amt: ${amt:<8.2f} | Status: {status} | "
                    f"Prob: {prob:.6f} | Decision: {dec_str:<8} | Latency: {latency_ms:.2f} ms"
                )

                results.append({
                    "TransactionID": tx_id,
                    "TransactionAmt": amt,
                    "fraud_probability": prob,
                    "decision": dec,
                    "http_status": status,
                    "latency_ms": round(latency_ms, 2),
                    "success": True,
                    "error": "",
                })
            else:
                fail_count += 1
                print(f" [{idx}] TxID: {tx_id:<8} | Status: {status} | FAILED | Error: {err} | ({latency_ms:.2f} ms)")
                results.append({
                    "TransactionID": tx_id,
                    "TransactionAmt": amt,
                    "fraud_probability": "",
                    "decision": "ERROR",
                    "http_status": status,
                    "latency_ms": round(latency_ms, 2),
                    "success": False,
                    "error": err or "Unknown Error",
                })

            if args.watch and args.interval > 0:
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[!] Watch stream interrupted by user.")

    # 4. Summary Statistics
    print("=" * 75)
    print("  EXECUTION SUMMARY")
    print("=" * 75)
    total_sent = len(results)
    print(f" Total Transactions Sent : {total_sent}")
    print(f" Successfully Scored     : {success_count} ({success_count/total_sent*100:.1f}%)" if total_sent else 0)
    print(f" Failed Requests         : {fail_count}")
    print(f" Predicted LEGIT         : {legit_count}")
    print(f" Predicted FRAUD         : {fraud_count}")

    if latencies:
        sorted_lat = sorted(latencies)
        p50 = statistics.median(latencies)
        p95 = sorted_lat[int(0.95 * len(sorted_lat))]
        p99 = sorted_lat[int(0.99 * len(sorted_lat))]
        avg_lat = statistics.mean(latencies)
        print(f" Average API Latency     : {avg_lat:.2f} ms")
        print(f" P50 Latency (Median)    : {p50:.2f} ms")
        print(f" P95 Latency             : {p95:.2f} ms")
        print(f" P99 Latency             : {p99:.2f} ms")

    # 5. Save Output CSV if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "TransactionID",
            "TransactionAmt",
            "fraud_probability",
            "decision",
            "http_status",
            "latency_ms",
            "success",
            "error",
        ]
        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[+] Results successfully exported to: {out_path.resolve()}")

    print("=" * 75)


if __name__ == "__main__":
    main()
