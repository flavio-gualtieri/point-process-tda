"""Smoothed CSR null tables and the studentised departure.

A REDUCTION turns one L-hat(r) - r curve into a single number that is 1 at the alpha rejection
boundary, so `statistic(...) > 1` reads as "reject CSR at alpha" for every one of them, and
`delta_tilde` is the same map applied to a model's closed-form L(r) - r at nbar.

Two kinds, told apart by WHEN they studentise.

`sup` studentises POINTWISE and then reduces over r:

    S(x; n) = max over r in R(n) of |L_x(r) - r - m0(r; n)| / s0(r; n)  / c_alpha(n)

Dividing by s0(r) is unstable at small r, where CSR patterns rarely hold any pair at all and
s0(r) -> 0, so the sup needs the window R(n) = {r : expected CSR pairs >= min_pairs}. That window
is the whole cost of this ordering. A Matern hard core of radius R lives entirely below 2R and its
detection boundary sits at n R ~ 1.44 for the `lo` arm alone and ~1.61 for the two-arm `ext` rule
(measured over the Matern bank), i.e. at R / r_min ~ 0.8-0.9 -- just inside the radii the window
throws away, which is why the sup is near-powerless on cores it should be catching four times in
five.

`lo` and `hi` take the raw extremum FIRST and studentise the scalar:

    T_lo = (m_lo(n) - min_r phi(r)) / s_lo(n),     T_hi = (max_r phi(r) - m_hi(n)) / s_hi(n)

with phi = L-hat(r) - r, both oriented so that large means "far from CSR". Nothing is divided by a
vanishing s0(r), so neither needs a window and neither discards a radius. Below a hard core radius
R no point has a neighbour, so phi is pinned to the line -r and min_r phi = -R exactly and
NOISELESSLY: T_lo reads the core radius straight off the curve rather than inferring it from a
ratio of two small noisy numbers.

`ext` (the default) is the two-arm rule max(T_lo / c_lo, T_hi / c_hi), calibrated jointly, because
which side a pattern departs on is not known in advance and choosing per family would be choosing
the statistic after seeing the answer. Each arm is dead -- at or below alpha -- wherever the other
fires, so the selection happens by itself; the joint calibration costs about 25% on each arm's
threshold and buys repulsion power the pointwise sup cannot reach at any window.

The scalar moments m_lo, s_lo are NOT a bias correction, unlike the pointwise m0. The minimum of a
wiggly curve is negative under CSR too (m_lo ~ -0.87/n, s_lo ~ 0.32/n), so they measure the free
dip any pattern gets from noise alone, and what T_lo asks is whether a model's dip is DEEPER than
that. One consequence is load-bearing downstream: delta_tilde is not 0 at CSR under `lo`, `hi` or
`ext`, it is negative. delta_tilde = 0 means "this departure equals CSR's own noise floor", not
"this is CSR", and values below it stay negative and ordered instead of piling up at a clamp.

A reduction is a pure function of (curve, n) plus its own fitted moments and c_alpha, so adding one
costs a refit from the stored curves. Never a re-simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


# ---------------------------------------------------------------- the reductions

def _raw_sup(tables: "Tables", curves: np.ndarray, n: np.ndarray, center: bool) -> np.ndarray:
    m0, s0 = tables.moments(n)
    z = np.abs(curves - m0 * center) / s0
    return np.where(tables.window(n, "sup"), z, 0.0).max(axis=1)


def _raw_lo(tables: "Tables", curves: np.ndarray, n: np.ndarray, center: bool) -> np.ndarray:
    return -curves.min(axis=1)          # negated so that large = deep dip, like every other arm


def _raw_hi(tables: "Tables", curves: np.ndarray, n: np.ndarray, center: bool) -> np.ndarray:
    return curves.max(axis=1)


def _raw_ext(tables: "Tables", curves: np.ndarray, n: np.ndarray, center: bool) -> np.ndarray:
    """Already in units of each arm's own c_alpha, so `ext`'s own c_alpha IS the joint scale k."""
    return np.maximum(tables.statistic(curves, n, "lo", center),
                      tables.statistic(curves, n, "hi", center))


@dataclass(frozen=True)
class Reduction:
    name: str
    raw: Callable[["Tables", np.ndarray, np.ndarray, bool], np.ndarray]
    windowed: bool = False      # does it need R(n)?
    scalar: bool = False        # does its raw value carry its own (m, s) splines in n?
    arms: tuple[str, ...] = ()  # reductions that must already be calibrated for this one to exist


REDUCTIONS: dict[str, Reduction] = {r.name: r for r in (      # declaration order is fit order
    Reduction("sup", _raw_sup, windowed=True),
    Reduction("lo", _raw_lo, scalar=True),
    Reduction("hi", _raw_hi, scalar=True),
    Reduction("ext", _raw_ext, arms=("lo", "hi")),
)}
DEFAULT = "ext"
SCALAR = tuple(name for name, red in REDUCTIONS.items() if red.scalar)


class Tables:
    def __init__(self, source: Path | dict = TABLES):
        z = np.load(source) if isinstance(source, (str, Path)) else source
        have = set(getattr(z, "files", None) or z)
        self.n_low, self.n_high = float(z["n_low"]), float(z["n_high"])
        self.min_pairs = float(z["min_pairs"])
        self.t_moments, self.coef_m, self.coef_s = z["t_moments"], z["coef_m"], z["coef_s"]
        self.t_scalar = z["t_scalar"]
        self.coef_sm = {n: z[f"coef_sm__{n}"] for n in SCALAR if f"coef_sm__{n}" in have}
        self.coef_ss = {n: z[f"coef_ss__{n}"] for n in SCALAR if f"coef_ss__{n}" in have}
        self.t_c = z["t_c"]
        self.coef_c = {name: z[f"coef_c__{name}"] for name in REDUCTIONS if f"coef_c__{name}" in have}
        if isinstance(source, (str, Path)) and DEFAULT not in self.coef_c:
            raise ValueError(f"{source}: no critical values for `{DEFAULT}` -- refit the tables")

    def reduction(self, name: str) -> Reduction:
        if name not in REDUCTIONS:
            raise ValueError(f"unknown reduction `{name}`; have {sorted(REDUCTIONS)}")
        if name not in self.coef_c:
            raise ValueError(f"`{name}` is defined but not calibrated in the tables -- refit them")
        return REDUCTIONS[name]

    def _checked(self, n) -> np.ndarray:
        n = np.atleast_1d(np.asarray(n, float))
        if n.min() < self.n_low or n.max() > self.n_high:
            raise ValueError(f"n outside the tables' range [{self.n_low:g}, {self.n_high:g}]")
        return n

    def moments(self, n) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise CSR mean and s.d. of L-hat(r) - r. Used by the pointwise reductions."""
        n = self._checked(n)
        b = basis(n, self.t_moments)
        return b @ self.coef_m / n[:, None], b @ self.coef_s / n[:, None]

    def scalar_moments(self, name: str, n) -> tuple[np.ndarray, np.ndarray]:
        """CSR mean and s.d. of ONE reduction's raw extremum. The free dip, not a bias."""
        n = self._checked(n)
        b = basis(n, self.t_scalar)
        return b @ self.coef_sm[name] / n, b @ self.coef_ss[name] / n

    def window(self, n, reduction: str = DEFAULT) -> np.ndarray:
        """R(n) as a boolean mask over RADII; all radii for an unwindowed reduction."""
        n = self._checked(n)
        if not REDUCTIONS[reduction].windowed:
            return np.ones((len(n), len(RADII)), dtype=bool)
        return RADII[None, :] >= r_min(n, self.min_pairs)[:, None]

    def critical(self, n, reduction: str = DEFAULT) -> np.ndarray:
        self.reduction(reduction)
        return basis(self._checked(n), self.t_c) @ self.coef_c[reduction]

    def statistic(self, curves, n, reduction: str = DEFAULT, center: bool = True) -> np.ndarray:
        curves = np.atleast_2d(curves)
        n = np.broadcast_to(np.asarray(n, float), len(curves))
        red = self.reduction(reduction)
        t = red.raw(self, curves, n, center)
        if red.scalar:
            m, s = self.scalar_moments(reduction, n)
            t = (t - m) / s
        return t / self.critical(n, reduction)

    def delta_tilde(self, l_minus_r_model, nbar, reduction: str = DEFAULT) -> np.ndarray:
        """`center=False` reaches only the pointwise reductions: the model curve is exact, so its
        pointwise reference is 0 rather than the estimator's bias m0(r). The scalar reductions
        ignore it and always subtract m_lo/m_hi, which are the noise floor and not a bias."""
        return self.statistic(l_minus_r_model, nbar, reduction, center=False)
