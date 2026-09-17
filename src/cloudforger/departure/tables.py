"""Smoothed CSR null tables and the studentised departure.

    S(x; n)        = max_{r in R(n)} |L_x(r) - r - m0(r; n)| / s0(r; n) / c95(n)
    delta_tilde    = the same with m0 = 0, applied to a model's L(r) - r at nbar
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from ..classical.lfunction import RADII
from .config import TABLES

DEGREE = 3


def knots(n_low: float, n_high: float, interior: int) -> np.ndarray:
    x = np.linspace(np.log(n_low), np.log(n_high), interior + 2)
    return np.concatenate([[x[0]] * DEGREE, x, [x[-1]] * DEGREE])


def basis(n: np.ndarray, t: np.ndarray) -> np.ndarray:
    return BSpline.design_matrix(np.log(np.asarray(n, float)), t, DEGREE).toarray()


def r_min(n: np.ndarray, min_pairs: float) -> np.ndarray:
    n = np.asarray(n, float)
    return np.sqrt(2 * min_pairs / (np.pi * n * (n - 1)))


class Tables:
    def __init__(self, path: Path = TABLES):
        z = np.load(path)
        self.n_low, self.n_high = float(z["n_low"]), float(z["n_high"])
        self.min_pairs = float(z["min_pairs"])
        self.t_moments, self.coef_m, self.coef_s = z["t_moments"], z["coef_m"], z["coef_s"]
        self.t_c95, self.coef_c = z["t_c95"], z["coef_c"]

    def moments(self, n):
        n = np.atleast_1d(np.asarray(n, float))
        if n.min() < self.n_low or n.max() > self.n_high:
            raise ValueError(f"n outside the tables' range [{self.n_low:g}, {self.n_high:g}]")
        b = basis(n, self.t_moments)
        m0 = b @ self.coef_m / n[:, None]
        s0 = b @ self.coef_s / n[:, None]
        c95 = basis(n, self.t_c95) @ self.coef_c
        mask = RADII[None, :] >= r_min(n, self.min_pairs)[:, None]
        return m0, s0, c95, mask

    def statistic(self, curves, n, center: bool = True) -> np.ndarray:
        curves = np.atleast_2d(curves)
        m0, s0, c95, mask = self.moments(np.broadcast_to(n, len(curves)))
        z = np.abs(curves - m0 * center) / s0
        return np.where(mask, z, 0.0).max(axis=1) / c95

    def delta_tilde(self, l_minus_r_model, nbar) -> np.ndarray:
        return self.statistic(l_minus_r_model, nbar, center=False)
