"""
Phase 2 — Reproducibility Utilities
======================================
Sets seed=42 for Python, NumPy, and PyTorch.
Configures deterministic CUDA behavior when available.
Records device selection.
"""

from __future__ import annotations

import os
import random
import logging
import numpy as np
import torch

log = logging.getLogger(__name__)

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """
    Set all random seeds for reproducibility.

    Covers:
    - Python built-in random
    - NumPy
    - PyTorch (CPU + CUDA)
    - PYTHONHASHSEED environment variable

    For CUDA determinism, also sets:
    - torch.backends.cudnn.deterministic = True
    - torch.backends.cudnn.benchmark = False
    Note: deterministic CUDA ops may be slower but guarantee reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log.info("Seeds set: Python=%d  NumPy=%d  PyTorch=%d", seed, seed, seed)


def get_device() -> torch.device:
    """
    Select compute device.

    Priority: CUDA → CPU.
    The model is designed to work on CPU. CUDA is used when available
    but is not required.

    Returns:
        torch.device: 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        log.info("Device: CPU")
    return device


def configure(seed: int = SEED) -> torch.device:
    """
    One-call setup: set seeds and detect device.

    Usage:
        from phase2_gru.src.utils.reproducibility import configure
        device = configure()

    Returns:
        torch.device
    """
    set_seed(seed)
    device = get_device()
    return device
