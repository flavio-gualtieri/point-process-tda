"""LGCP grid size: smallest power of two M whose discretisation bias in K is
below tol * (null s.d. of K-hat) on R(nbar).

The grid LGCP has intensity constant on M x M cells, so its pair correlation
averaged over grid position is the bilinear interpolation of exp(C) on the
lattice (Z/M)^2. Bias = K_grid - K, integrated on a D/4 midpoint rule.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..classical.lfunction import R_MAX, RADII
from ..departure.tables import Tables

_SUB = 4


@lru_cache(maxsize=4)
def _quadrature(M: int):
    D = 1.0 / M
    h = (np.arange(int(np.ceil(R_MAX * M * _SUB))) + 0.5) * D / _SUB
    f = np.floor(h * M).astype(int)
    t = h * M - f
    A = np.zeros((len(h), f.max() + 2))
    A[np.arange(len(h)), f] = 1 - t
    A[np.arange(len(h)), f + 1] = t
    radius = np.hypot(h[:, None], h[None, :])
    return A, radius, np.searchsorted(RADII, radius, side="left")


def grid_bias(sigma2: float, s: float, M: int) -> np.ndarray:
    A, radius, bins = _quadrature(M)
    k = np.arange(A.shape[1]) / M
    lattice = np.exp(sigma2 * np.exp(-np.hypot(k[:, None], k[None, :]) / s))
    diff = (A @ lattice @ A.T - np.exp(sigma2 * np.exp(-radius / s))) * 4 * (1 / (M * _SUB)) ** 2
    return np.cumsum(np.bincount(bins.ravel(), weights=diff.ravel(), minlength=len(RADII) + 1)[: len(RADII)])


def grid_size(sigma2: float, s: float, nbar: float, tables: Tables,
              tol: float = 0.1, M_min: int = 128, M_max: int = 4096) -> int | None:
    _, s0, _, mask = tables.moments(nbar)
    s0K = 2 * np.pi * RADII * s0[0]
    M = M_min
    while M <= M_max:
        if np.all(np.abs(grid_bias(sigma2, s, M))[mask[0]] <= tol * s0K[mask[0]]):
            return M
        M *= 2
    return None
