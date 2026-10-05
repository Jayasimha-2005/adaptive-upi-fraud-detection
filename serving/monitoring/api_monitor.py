"""
monitoring/api_monitor.py
Phase 8 — Basic API request monitoring for the E1 fraud model serving system.

Provides a simple in-memory MonitoringState that accumulates:
  - Total request counts
  - Successful / failed request counts
  - HTTP status-code bucket counts (2xx / 4xx / 5xx)
  - Per-prediction inference latency (ms) for percentile analysis

LIMITATIONS (Research Prototype):
  - In-memory only: state resets on every process restart.
  - Not thread-safe under high concurrency (acceptable for sequential research use).
  - No persistence, no external metrics backend.

Usage:
    from monitoring.api_monitor import monitor

    monitor.record(http_status=200, latency_ms=85.3)
    summary = monitor.get_summary()
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MonitoringState:
    """
    In-memory accumulator for basic API serving metrics.

    All counters reset when the application process restarts.
    """

    # ── Request counters ─────────────────────────────────────────────────────
    total_requests: int = 0
    successful_requests: int = 0        # HTTP 2xx
    failed_requests: int = 0            # HTTP 4xx or 5xx
    http_2xx_count: int = 0
    http_4xx_count: int = 0
    http_5xx_count: int = 0

    # ── Latency accumulator ──────────────────────────────────────────────────
    latency_ms_list: List[float] = field(default_factory=list)

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

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary dictionary of all accumulated metrics.

        Returns
        -------
        dict
            Contains request counts, HTTP bucket counts, and latency statistics.
            Latency statistics are None if no successful predictions have been recorded.
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
            "total_requests":      self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests":     self.failed_requests,
            "http_2xx_count":      self.http_2xx_count,
            "http_4xx_count":      self.http_4xx_count,
            "http_5xx_count":      self.http_5xx_count,
            **latency_stats,
        }

    def reset(self) -> None:
        """Reset all counters and latency data. Useful for test isolation."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.http_2xx_count = 0
        self.http_4xx_count = 0
        self.http_5xx_count = 0
        self.latency_ms_list = []


# ── Module-level singleton ────────────────────────────────────────────────────
# api/main.py imports this instance and calls monitor.record() per request.
monitor = MonitoringState()
