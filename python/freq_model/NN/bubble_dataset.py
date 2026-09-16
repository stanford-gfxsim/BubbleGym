"""Torch ``Dataset`` and reproducible split shared by every BubbleFreq trainer.

This module is the single source of truth for:

- ``BubbleDataset``  : 2-tensor wrapping numpy arrays for ``DataLoader``.
- ``SplitData``      : dataclass bundling train / val / test splits plus
                       per-row baseline ``log f`` arrays for residual
                       reconstruction inside metrics.
- ``split_dataset``  : seeded shuffle into the requested ratios.
- ``set_seed``       : deterministic seeding across random / numpy / torch.
- ``canonicalize_columns`` : strip leading ``#`` and whitespace from CSV columns
                             (the bubble_gym CSVs use both conventions).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


__all__ = [
    "BubbleDataset",
    "SplitData",
    "canonicalize_columns",
    "set_seed",
    "split_dataset",
]


class BubbleDataset(Dataset):
    """Wrap ``(x, y)`` numpy arrays as a torch ``Dataset``.

    ``y`` is reshaped to ``(N, 1)`` so the network's scalar output broadcasts
    cleanly with the loss.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    log_fs_train: np.ndarray
    log_fs_val: np.ndarray
    log_fs_test: np.ndarray


def set_seed(seed: int) -> None:
    """Seed Python, numpy, torch (CPU + CUDA) for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading ``#`` and whitespace from column names (non-mutating)."""
    df = df.copy()
    df.columns = df.columns.str.lstrip("#").str.strip()
    return df


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    log_fs: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitData:
    """Shuffle-split into train / val / test using a fixed RNG seed."""
    n = len(x)
    if n < 10:
        raise ValueError(f"Too few samples for split: {n}")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test}. "
            "Adjust val_ratio/test_ratio."
        )

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return SplitData(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_val=x[val_idx],
        y_val=y[val_idx],
        x_test=x[test_idx],
        y_test=y[test_idx],
        idx_train=train_idx,
        idx_val=val_idx,
        idx_test=test_idx,
        log_fs_train=log_fs[train_idx],
        log_fs_val=log_fs[val_idx],
        log_fs_test=log_fs[test_idx],
    )
