# src/cloudforger/core/splits.py

from __future__ import annotations

import numpy as np
import torch

from torch.utils.data import Dataset, Subset, random_split


def train_val_test_split(
    dataset: Dataset,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> tuple[Subset, Subset, Subset]:
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")

    n = len(dataset)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    n_test = n - n_train - n_val   # remainder avoids rounding gaps

    generator = torch.Generator().manual_seed(seed)

    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def train_val_test_indices(
    n: int, seed: int, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw-index counterpart to train_val_test_split, for callers that index
    numpy arrays directly instead of wrapping a torch Dataset (e.g. classical
    baselines that need "the same seed's test partition" a trained model
    used, without constructing a Dataset just to get indices). Produces an
    identical partition to train_val_test_split for the same (n, seed):
    both draw from torch.randperm(n, generator=seeded) sliced at the same
    cumulative offsets."""
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).numpy()
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]
