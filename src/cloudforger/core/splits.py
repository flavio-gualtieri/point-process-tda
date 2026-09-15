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


def resolve_split(
    n: int,
    seed: int,
    split_labels: "np.ndarray | list[str] | None" = None,
    reshuffle_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train, val, test) indices for one training pool.

    split_labels None: the legacy random cut, train_val_test_indices(n, seed).
    split_labels given (the DV3 `split` column, aligned to the pool's rows):
    train/val come from it -- fixed at generation, identical for every seed
    and method -- and test is EMPTY, because under DV3 the test data are the
    separate products A/B/C, never a slice of the training pool (see
    cloudforger.evaluation.dv3)."""
    if split_labels is None:
        return train_val_test_indices(n, seed)
    from ..evaluation.dv3 import split_indices_from_records  # local: evaluation imports paths -> filtration

    labels = list(split_labels)
    if len(labels) != n:
        raise ValueError(f"{len(labels)} split labels for a pool of {n} rows")
    train_idx, val_idx = split_indices_from_records(
        [{"split": s} for s in labels], reshuffle_seed=reshuffle_seed,
    )
    return train_idx, val_idx, np.empty(0, dtype=np.int64)
