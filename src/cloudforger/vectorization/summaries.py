"""Persistence diagram -> fixed-length summary vector, for table learners (trees, linear models).

Per homology dimension, on the sqrt(n)-rescaled axis (mean-spacing units, as the classical sqrtn axis):
    Betti curve / n at GRID points      grid fitted once per (filtration, dim) -- see fit_grid
    count / n, total persistence / n, lifetime mean, s.d., max and quantiles (.1 .25 .5 .75 .9 .99)
    birth and death quantiles (.1 .5 .9)
H1 also: death / birth ratio quantiles (.5 .9 .99) and max, and the three longest lifetimes.

Undefined statistics (an empty diagram) are NaN, which gradient-boosted trees handle natively.
"""

from __future__ import annotations

import numpy as np

GRID = 32
QS = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
Q3 = (0.1, 0.5, 0.9)


def fit_grid(values: np.ndarray, size: int = GRID) -> np.ndarray:
    """Betti-curve grid over the pooled 0.5-99.5% quantiles of rescaled births and deaths."""
    return np.linspace(*np.quantile(values, [0.005, 0.995]), size)


def _q(x, qs):
    return np.quantile(x, qs) if len(x) else np.full(len(qs), np.nan)


def vector(h0: np.ndarray, h1: np.ndarray, n: int, grids) -> np.ndarray:
    s, out = np.sqrt(n), []
    for d, pairs in ((0, h0), (1, h1)):
        b, e = pairs[:, 0] * s, pairs[:, 1] * s
        life = e - b
        g = grids[d]
        out.append(((b[None, :] <= g[:, None]) & (e[None, :] > g[:, None])).sum(1) / n)
        stats = [len(life) / n, life.sum() / n] + ([life.mean(), life.std(), life.max()] if len(life) else [np.nan] * 3)
        out += [stats, _q(life, QS), _q(b, Q3), _q(e, Q3)]
        if d == 1:
            ratio = e / np.maximum(b, 1e-12)
            top = np.full(3, np.nan)
            top[:min(3, len(life))] = np.sort(life)[::-1][:3]
            out += [_q(ratio, (0.5, 0.9, 0.99)), [ratio.max() if len(ratio) else np.nan], top]
    return np.concatenate([np.asarray(x, float) for x in out])


def vector_names(tag: str, grid: int = GRID) -> list[str]:
    names = []
    for d in (0, 1):
        names += [f"{tag}_h{d}_betti@{j}" for j in range(grid)]
        names += [f"{tag}_h{d}_{k}" for k in ("count", "total", "life_mean", "life_sd", "life_max")]
        names += [f"{tag}_h{d}_life_q{q}" for q in QS] + [f"{tag}_h{d}_birth_q{q}" for q in Q3]
        names += [f"{tag}_h{d}_death_q{q}" for q in Q3]
        if d == 1:
            names += [f"{tag}_h1_ratio_q{q}" for q in (0.5, 0.9, 0.99)] + [f"{tag}_h1_ratio_max"]
            names += [f"{tag}_h1_top{j}" for j in (1, 2, 3)]
    return names
