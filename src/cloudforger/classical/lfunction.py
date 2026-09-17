"""Ripley's L on the unit square W = [0,1]^2, isotropic edge correction."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

R_MAX = 0.25
RADII = R_MAX * np.arange(1, 513) / 512

_CORNERS = ((0, 2), (0, 3), (1, 2), (1, 3))


def isotropic_weights(centres: np.ndarray, d: np.ndarray) -> np.ndarray:
    """2 pi / (length of the circle of radius d around each centre inside W). Exact for d <= 0.5."""
    b = np.column_stack([centres[:, 0], 1 - centres[:, 0], centres[:, 1], 1 - centres[:, 1]])
    a = np.arccos(np.minimum(b / d[:, None], 1.0))
    overlap = sum(np.maximum(a[:, p] + a[:, q] - np.pi / 2, 0.0) for p, q in _CORNERS)
    outside = 2 * a.sum(axis=1) - overlap
    return 2 * np.pi / (2 * np.pi - outside)


def k_function(points: np.ndarray, radii: np.ndarray = RADII) -> np.ndarray:
    n = len(points)
    pairs = cKDTree(points).query_pairs(radii[-1], output_type="ndarray")
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    d = np.linalg.norm(points[i] - points[j], axis=1)
    w = isotropic_weights(points[i], d)
    counts = np.bincount(np.searchsorted(radii, d), weights=w, minlength=len(radii))
    return np.cumsum(counts[: len(radii)]) / (n * (n - 1))


def l_minus_r(points: np.ndarray, radii: np.ndarray = RADII) -> np.ndarray:
    return np.sqrt(k_function(points, radii) / np.pi) - radii


def l_minus_r_from_excess(excess: np.ndarray, radii: np.ndarray = RADII) -> np.ndarray:
    """L(r) - r from e = K(r) - pi r^2, written to avoid cancellation near CSR."""
    e = np.asarray(excess, float) / np.pi
    root = np.sqrt(np.clip(radii**2 + e, 0.0, None))
    return np.where(radii**2 + e <= 0, -radii, e / (root + radii))
