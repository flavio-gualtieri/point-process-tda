# src/cloudforger/vectorization/landscapes/tent.py
"""The one primitive landscapes and silhouettes are both built from: the
tent function

    Lambda_i(t) = max(0, min(t - b_i, d_i - t))

evaluated for every diagram point i against every grid point t at once
(tent below returns the full (M, G) matrix, never a single scalar).
landscape_from_tents/silhouette_from_tents both just reduce that one matrix
differently -- along the point axis by order statistic (landscape) or by a
persistence-weighted sum (silhouette) -- so a diagram is only ever swept
into tent form once per (dim, grid).

No calibration, no PersistenceDiagram import: this module only knows about
raw (M, 2) birth/death pairs and a precomputed grid -- see
landscape_silhouette.py for the per-dim transformer that wires this to
PersistenceDiagram + a calibrated grid, and calibrated.py for how that grid
and K get chosen."""

from __future__ import annotations

import numpy as np


def tent(pairs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Lambda_i(t) for every pair i (rows) and every grid point t
    (columns) -- shape (M, G). M=0 (empty diagram) returns an empty (0, G)
    matrix, not an error, so callers (landscape_from_tents,
    silhouette_from_tents) can treat "no points" as "zero rows" uniformly."""
    pairs = np.asarray(pairs, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if pairs.size == 0:
        return np.empty((0, grid.shape[0]), dtype=float)

    births = pairs[:, 0][:, None]   # (M, 1)
    deaths = pairs[:, 1][:, None]   # (M, 1)
    t = grid[None, :]               # (1, G)
    return np.maximum(0.0, np.minimum(t - births, deaths - t))


def landscape_from_tents(tents: np.ndarray, K: int) -> np.ndarray:
    """lambda_k(t) for k = 1..K, shape (K, G): the k-th largest tent value
    at each grid point, sorted independently per column so lambda_1(t) is
    the pointwise max and lambda_k(t) >= lambda_{k+1}(t) everywhere by
    construction. Fewer than K tents nonzero at a given t (either because
    the diagram has fewer than K points at all, or because most tents don't
    reach that far) -> those ranks are 0, not a truncation error: the
    multiset {Lambda_i(t)} always contains M values (M = tents.shape[0]),
    and any k-th-largest beyond M is defined as 0."""
    M, G = tents.shape
    out = np.zeros((K, G), dtype=float)
    if M == 0:
        return out
    sorted_desc = -np.sort(-tents, axis=0)  # (M, G), descending per column
    k_take = min(K, M)
    out[:k_take] = sorted_desc[:k_take]
    return out


def silhouette_from_tents(
    tents: np.ndarray, pairs: np.ndarray, p: float
) -> tuple[np.ndarray, np.ndarray]:
    """(phi_normalized, phi_unnormalized), each shape (G,):

        phi^(p)(t) = sum_i |d_i - b_i|^p Lambda_i(t) / sum_i |d_i - b_i|^p

    phi_unnormalized is just the numerator -- the denominator divides out
    total feature count/weight, which the calling experiment may want to
    keep (see module docstring in calibrated.py). p=0 reduces to an
    unweighted mean over tents (|d-b|^0 == 1 for every point, matching
    numpy's 0**0 == 1 convention so a degenerate d==b pair doesn't raise).
    p=np.inf is a one-hot limit: weight concentrates entirely on the
    longest-persistence pair (ties broken by first occurrence, same as
    np.argmax), recovering "the longest tent" as p -> infinity. Empty
    diagrams (M=0) return two all-zero (G,) arrays."""
    pairs = np.asarray(pairs, dtype=float)
    G = tents.shape[1] if tents.ndim == 2 else 0
    if pairs.size == 0:
        zeros = np.zeros(G, dtype=float)
        return zeros, zeros

    persistence = pairs[:, 1] - pairs[:, 0]
    if np.isinf(p):
        weights = np.zeros_like(persistence)
        weights[np.argmax(persistence)] = 1.0
    else:
        weights = persistence ** p

    numerator = (weights[:, None] * tents).sum(axis=0)  # (G,)
    denominator = float(weights.sum())
    normalized = numerator / denominator if denominator > 0 else np.zeros_like(numerator)
    return normalized, numerator
