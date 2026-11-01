"""
phase4_api/benchmarks/benchmark_config.py
=========================================
Configuration Dataclass & Validator for Phase 8 Benchmarking Suite.

Part of Member 3 — API Serving Layer for Adaptive UPI / BAF Fraud Detection.
Phase 8: Performance Testing, Benchmarking & Parity Validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Optional

from phase4_api.src.batch_inference_engine import DEFAULT_MAX_BATCH_SIZE


@dataclass
class BenchmarkConfig:
    """
    Configuration parameters governing the Phase 8 benchmarking suite.
    """

    # Data & Source
    dataset_path: Optional[str] = None
    use_synthetic_if_missing: bool = False
    limit: int = 100

    # API Target
    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 30.0

    # Benchmarking Parameters
    batch_sizes: List[int] = field(default_factory=lambda: [1, 10, 50, 100])
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    warmup_requests: int = 10
    repetitions: int = 1

    # Parity Evaluation
    tolerance: float = 1e-8

    # Random Seed for Reproducibility
    random_seed: int = 42

    # Output Folders (Strictly inside phase4_api/)
    output_dir: str = "phase4_api/results"
    reports_dir: str = "phase4_api/reports"

    def __post_init__(self) -> None:
        """Validates configuration bounds and parameters."""
        if self.limit <= 0:
            raise ValueError(f"limit must be positive, got {self.limit}")

        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got {self.base_url}")

        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")

        if self.warmup_requests < 0:
            raise ValueError(f"warmup_requests cannot be negative, got {self.warmup_requests}")

        if self.repetitions <= 0:
            raise ValueError(f"repetitions must be positive, got {self.repetitions}")

        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")

        # Validate batch sizes
        for bs in self.batch_sizes:
            if bs <= 0:
                raise ValueError(f"Batch size must be > 0, got {bs}")
            if bs > DEFAULT_MAX_BATCH_SIZE:
                raise ValueError(
                    f"Batch size {bs} exceeds API maximum batch limit of {DEFAULT_MAX_BATCH_SIZE}"
                )

        # Validate concurrency levels
        for conc in self.concurrency_levels:
            if conc <= 0:
                raise ValueError(f"Concurrency level must be > 0, got {conc}")

        # Ensure output directories are inside phase4_api/
        out_path = Path(self.output_dir).resolve()
        rep_path = Path(self.reports_dir).resolve()
        
        # Make directories if needed
        out_path.mkdir(parents=True, exist_ok=True)
        rep_path.mkdir(parents=True, exist_ok=True)
