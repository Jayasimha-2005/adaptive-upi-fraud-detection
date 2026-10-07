"""
monitoring/api_monitor.py
Phase 8 — Basic API request monitoring for the E1 fraud model serving system.

Provides a simple in-memory MonitoringState that accumulates:
  - Total request counts
  - Successful / failed request counts
  - HTTP status-code bucket counts (2xx / 4xx / 5xx)
  - Fraud / Legitimate decision counters
  - Real-time recent prediction buffer for the live dashboard
  - Per-prediction inference latency (ms) for percentile analysis

LIMITATIONS (Research Prototype):
  - In-memory only: state resets on every process restart.
  - Bounded ring buffer for recent predictions (max 200 items).
  - No persistence, no external metrics backend.

Usage:
    from monitoring.api_monitor import monitor

    monitor.record(http_status=200, latency_ms=85.3)
    monitor.record_prediction(
        transaction_id="3663602",
        fraud_probability=0.673801,
        decision="FRAUD",
        latency_ms=85.3,
        http_status=200,
    )
    summary = monitor.get_summary()
"""
from __future__ import annotations

import statistics
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MonitoringState:
    """
    In-memory accumulator for API serving metrics and live dashboard state.

    All counters reset when the application process restarts.
    """

    # ── Session tracking ─────────────────────────────────────────────────────
    session_id: str = field(default_factory=lambda: f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}")
    session_start_time: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    # ── Request counters ─────────────────────────────────────────────────────
    total_requests: int = 0
    successful_requests: int = 0        # HTTP 2xx
    failed_requests: int = 0            # HTTP 4xx or 5xx
    http_2xx_count: int = 0
    http_4xx_count: int = 0
    http_5xx_count: int = 0

    # ── Decision counters ────────────────────────────────────────────────────
    fraud_count: int = 0
    legit_count: int = 0

    # ── Latency accumulator ──────────────────────────────────────────────────
    latency_ms_list: List[float] = field(default_factory=list)

    # ── Recent predictions ring-buffer (bounded to 2000 items) ───────────────
    recent_predictions: deque = field(default_factory=lambda: deque(maxlen=2000))

    # ── Background Test Run Progress ─────────────────────────────────────────
    test_run_progress: Dict[str, Any] = field(default_factory=lambda: {
        "status": "idle",
        "requested": 0,
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "mode": "fixed",
        "message": "Ready",
    })

    def reset_session(self, new_session_id: Optional[str] = None) -> str:
        """
        Reset the current test session: clears predictions, counters, and assigns a new session ID.
        """
        self.session_id = new_session_id or f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.session_start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.http_2xx_count = 0
        self.http_4xx_count = 0
        self.http_5xx_count = 0
        self.fraud_count = 0
        self.legit_count = 0
        self.latency_ms_list.clear()
        self.recent_predictions.clear()
        self.test_run_progress = {
            "status": "idle",
            "requested": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "mode": "fixed",
            "message": "Ready",
        }
        return self.session_id

    def record(self, http_status: int, latency_ms: Optional[float] = None) -> None:
        """
        Record a single API request.

        Parameters
        ----------
        http_status : int
            The HTTP response status code (e.g. 200, 422, 500).
        latency_ms : float, optional
            Total inference latency in milliseconds. Omit for non-inference endpoints
            (e.g. /health) or when a request fails before inference runs.
        """
        self.total_requests += 1

        if 200 <= http_status < 300:
            self.successful_requests += 1
            self.http_2xx_count += 1
        elif 400 <= http_status < 500:
            self.failed_requests += 1
            self.http_4xx_count += 1
        elif 500 <= http_status < 600:
            self.failed_requests += 1
            self.http_5xx_count += 1

        if latency_ms is not None and latency_ms >= 0:
            self.latency_ms_list.append(latency_ms)

    def record_prediction(
        self,
        transaction_id: str,
        fraud_probability: Optional[float],
        decision: str,
        latency_ms: float,
        http_status: int = 200,
    ) -> None:
        """
        Record a structured prediction outcome for live dashboard display.
        """
        if decision == "FRAUD":
            self.fraud_count += 1
        elif decision == "LEGIT":
            self.legit_count += 1

        # Prepend to recent predictions (newest at index 0)
        self.recent_predictions.appendleft({
            "transaction_id": str(transaction_id),
            "fraud_probability": round(float(fraud_probability), 6) if fraud_probability is not None else None,
            "decision": str(decision),
            "latency_ms": round(float(latency_ms), 2) if latency_ms is not None else 0.0,
            "timestamp": time.strftime("%H:%M:%S"),
            "http_status": int(http_status),
        })

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary dictionary of all accumulated metrics (for /metrics).
        """
        n = len(self.latency_ms_list)
        latency_stats: Dict[str, Any]

        if n == 0:
            latency_stats = {
                "latency_sample_count": 0,
                "latency_min_ms":  None,
                "latency_mean_ms": None,
                "latency_p50_ms":  None,
                "latency_p95_ms":  None,
                "latency_p99_ms":  None,
                "latency_max_ms":  None,
            }
        else:
            sorted_lat = sorted(self.latency_ms_list)

            def _percentile(data: List[float], p: float) -> float:
                """Simple percentile via linear interpolation (no numpy required)."""
                if len(data) == 1:
                    return data[0]
                idx = (p / 100) * (len(data) - 1)
                lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
                return data[lo] + (data[hi] - data[lo]) * (idx - lo)

            latency_stats = {
                "latency_sample_count": n,
                "latency_min_ms":  round(sorted_lat[0], 3),
                "latency_mean_ms": round(statistics.mean(self.latency_ms_list), 3),
                "latency_p50_ms":  round(_percentile(sorted_lat, 50), 3),
                "latency_p95_ms":  round(_percentile(sorted_lat, 95), 3),
                "latency_p99_ms":  round(_percentile(sorted_lat, 99), 3),
                "latency_max_ms":  round(sorted_lat[-1], 3),
            }

        return {
            "session_id":          self.session_id,
            "session_start_time":  self.session_start_time,
            "total_requests":      self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests":     self.failed_requests,
            "http_2xx_count":      self.http_2xx_count,
            "http_4xx_count":      self.http_4xx_count,
            "http_5xx_count":      self.http_5xx_count,
            "fraud_count":         self.fraud_count,
            "legit_count":         self.legit_count,
            **latency_stats,
        }

    def get_dashboard_state(self, system_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Return comprehensive dashboard state including system info, metrics, and recent predictions.
        """
        summary = self.get_summary()
        total_decisions = self.fraud_count + self.legit_count
        fraud_rate = round((self.fraud_count / total_decisions * 100.0), 2) if total_decisions > 0 else 0.0

        return {
            "session": {
                "session_id": self.session_id,
                "session_start_time": self.session_start_time,
            },
            "system": system_info or {
                "status": "ONLINE",
                "model": "E1_LightGBM",
                "version": "1.0.0",
                "threshold": 0.616521,
                "features": 406,
            },
            "metrics": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "client_errors": self.http_4xx_count,
                "server_errors": self.http_5xx_count,
                "fraud_count": self.fraud_count,
                "legit_count": self.legit_count,
                "fraud_rate": fraud_rate,
                "latency_min_ms": summary.get("latency_min_ms"),
                "latency_mean_ms": summary.get("latency_mean_ms"),
                "latency_p50_ms": summary.get("latency_p50_ms"),
                "latency_p95_ms": summary.get("latency_p95_ms"),
                "latency_p99_ms": summary.get("latency_p99_ms"),
                "latency_max_ms": summary.get("latency_max_ms"),
            },
            "test_run": self.test_run_progress,
            "recent_predictions": list(self.recent_predictions)[:2000],
        }

    def reset(self) -> None:
        """Reset all counters, latency data, and prediction ring buffer."""
        self.reset_session()


# ── Module-level singleton ────────────────────────────────────────────────────
monitor = MonitoringState()
