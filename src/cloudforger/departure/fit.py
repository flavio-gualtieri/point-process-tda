"""Stored null curves -> spline tables in log n.

m0 and s0 shrink roughly like 1/n, so the splines are fitted to n*m0 and n*s0
(weighted by their Monte Carlo s.e.), one fit per radius. c95 is the (1 - alpha)
quantile of the calibration batch's statistic under the fitted moments.
"""

from __future__ import annotations

import numpy as np

from ..classical.lfunction import RADII
from .config import TABLES, Config
from .simulate import curves_path
from .tables import basis, knots, r_min


def raw_moments(x: np.ndarray):
    n = len(x)
    m = x.mean(axis=0)
    s = x.std(axis=0, ddof=1)
    mu4 = ((x - m) ** 4).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se_s = np.sqrt(np.maximum(mu4 - s**4, 0.0) / n) / (2 * s)
    return m, s, s / np.sqrt(n), se_s


def quantile_se(t: np.ndarray, q: float, h: float = 0.01) -> float:
    lo, hi = np.quantile(t, [q - h, q + h])
    return float((hi - lo) / (2 * h) * np.sqrt(q * (1 - q) / len(t)))


def wls(b: np.ndarray, y: np.ndarray, se: np.ndarray) -> np.ndarray:
    coef = np.zeros((b.shape[1], y.shape[1]))
    for j in range(y.shape[1]):
        ok = np.isfinite(se[:, j]) & (se[:, j] > 0)
        w = 1 / se[ok, j]
        coef[:, j] = np.linalg.lstsq(b[ok] * w[:, None], y[ok, j] * w, rcond=None)[0]
    return coef


def _z(y, fit, se, mask):
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (y - fit) / se
    return z[mask & np.isfinite(z)]


def fit(cfg: Config) -> dict:
    grid = cfg.grid()
    nn = grid[:, None].astype(float)
    load = lambda n, a, b: np.load(curves_path(n), mmap_mode="r")[a:b].astype(np.float64)

    m, s, se_m, se_s = map(np.stack, zip(*(raw_moments(load(n, 0, cfg.fit)) for n in grid)))
    t_m = knots(grid[0], grid[-1], cfg.knots_moments)
    b = basis(grid, t_m)
    coef_m = wls(b, nn * m, nn * se_m)
    coef_s = wls(b, nn * s, nn * se_s)
    m_fit, s_fit = b @ coef_m / nn, b @ coef_s / nn
    mask = RADII[None, :] >= r_min(grid, cfg.min_pairs)[:, None]

    c = np.empty(len(grid))
    se_c = np.empty(len(grid))
    q = 1 - cfg.alpha
    for g, n in enumerate(grid):
        t = np.where(mask[g], np.abs(load(n, cfg.fit, cfg.reps) - m_fit[g]) / s_fit[g], 0.0).max(axis=1)
        c[g], se_c[g] = np.quantile(t, q), quantile_se(t, q)
    t_c = knots(grid[0], grid[-1], cfg.knots_c95)
    b_c = basis(grid, t_c)
    coef_c = wls(b_c, c[:, None], se_c[:, None])[:, 0]

    TABLES.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TABLES, n_low=grid[0], n_high=grid[-1], min_pairs=cfg.min_pairs,
             t_moments=t_m, coef_m=coef_m, coef_s=coef_s, t_c95=t_c, coef_c=coef_c)

    z_m, z_s = _z(m, m_fit, se_m, mask), _z(s, s_fit, se_s, mask)
    z_c = (c - b_c @ coef_c) / se_c
    summary = lambda z: {"mean_z2": float(np.mean(z**2)), "p99_abs_z": float(np.quantile(np.abs(z), 0.99))}
    return {
        "grid": {"size": len(grid), "low": int(grid[0]), "high": int(grid[-1])},
        "m0": summary(z_m), "s0": summary(z_s),
        "c95": {**summary(z_c), "range": [float(c.min()), float(c.max())], "max_se": float(se_c.max())},
        "expected_mean_z2": {"moments": 1 - b.shape[1] / len(grid), "c95": 1 - b_c.shape[1] / len(grid)},
    }
