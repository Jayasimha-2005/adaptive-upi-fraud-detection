"""
Phase 2 — E2 Sequence Dataset
================================
Lightweight PyTorch Dataset wrapper around numpy arrays
produced by sequence_builder.build_sequences().

Design notes:
- Stores X as float32 numpy array, converts to tensor on __getitem__
- y stored as float32 scalar for BCEWithLogitsLoss compatibility
- No data augmentation (locked spec: no SMOTE, no oversampling)
- Works on CPU (num_workers=0 safe on Windows)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    """
    Dataset for E2 GRU sequences.

    Args:
        X : np.ndarray  shape [N, 4, 406]  float32
        y : np.ndarray  shape [N]          float32  (0.0 or 1.0)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        assert X.ndim == 3 and X.shape[1] == 4 and X.shape[2] == 406, \
            f"X shape must be [N,4,406], got {X.shape}"
        assert len(y) == len(X), \
            f"X/y length mismatch: {len(X)} vs {len(y)}"
        # Store as contiguous float32 numpy arrays
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])          # [4, 406]
        label = torch.tensor(self.y[idx]).unsqueeze(0)  # [1]
        return x, label


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for E2 sequences.

    Args:
        X          : float32 array [N, 4, 406]
        y          : float32 array [N]
        batch_size : mini-batch size (locked: 128)
        shuffle    : True for train, False for val/test
        num_workers: 0 on Windows to avoid multiprocessing issues
    """
    ds = SequenceDataset(X, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,   # CPU training — no pinning needed
        drop_last=False,
    )
