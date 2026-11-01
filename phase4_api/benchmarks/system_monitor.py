"""
phase4_api/benchmarks/system_monitor.py
======================================
System Environment & Resource Monitoring for Phase 8 Benchmarks.

Part of Member 3 — API Serving Layer for Adaptive UPI / BAF Fraud Detection.
Phase 8: Performance Testing, Benchmarking & Parity Validation.
"""
from __future__ import annotations

import logging
import os
import platform
import sys
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("phase4_api.benchmarks.system_monitor")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available; system resource monitoring will be disabled.")


class SystemMonitor:
    """
    Monitors process and host system resource utilization during benchmarks.
    Gracefully degrades if psutil is unavailable or permissions are restricted.
    """

    def __init__(self) -> None:
        self.is_available = PSUTIL_AVAILABLE
        self._process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
        self._start_cpu_time: Optional[float] = None
        self._start_wall_time: Optional[float] = None

    @classmethod
    def get_environment_info(cls) -> Dict[str, Any]:
        """
        Gathers system platform, hardware, and library versions for reproducibility.
        """
        env: Dict[str, Any] = {
            "os": f"{platform.system()} {platform.release()}",
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        }

        # Hardware metrics
        if PSUTIL_AVAILABLE:
            try:
                env["cpu_count_logical"] = psutil.cpu_count(logical=True)
                env["cpu_count_physical"] = psutil.cpu_count(logical=False)
                ram_gb = psutil.virtual_memory().total / (1024**3)
                env["total_ram_gb"] = round(ram_gb, 2)
            except Exception as e:
                env["hardware_error"] = str(e)
        else:
            env["hardware_info"] = "psutil unavailable"

        # Library versions
        libraries = ["lightgbm", "fastapi", "uvicorn", "pydantic", "httpx", "requests", "pandas", "numpy"]
        versions: Dict[str, str] = {}
        for lib in libraries:
            try:
                mod = __import__(lib)
                versions[lib] = getattr(mod, "__version__", "unknown")
            except ImportError:
                versions[lib] = "not_installed"
        env["library_versions"] = versions

        return env

    def capture_snapshot(self) -> Dict[str, Any]:
        """
        Captures instantaneous CPU and memory metrics.
        """
        if not self.is_available or self._process is None:
            return {
                "monitoring_enabled": False,
                "reason": "psutil unavailable",
            }

        try:
            mem_info = self._process.memory_info()
            sys_mem = psutil.virtual_memory()
            return {
                "monitoring_enabled": True,
                "process_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
                "process_vms_mb": round(mem_info.vms / (1024 * 1024), 2),
                "system_cpu_percent": psutil.cpu_percent(interval=None),
                "system_memory_percent": sys_mem.percent,
                "system_available_ram_mb": round(sys_mem.available / (1024 * 1024), 2),
            }
        except Exception as e:
            logger.warning("Resource snapshot capture failed: %s", str(e))
            return {
                "monitoring_enabled": False,
                "error": str(e),
            }
