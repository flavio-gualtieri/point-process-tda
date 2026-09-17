"""Families: shape support, closed-form K(r) - pi r^2, validity rules, amplitude solve."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import brentq

from ..classical.lfunction import R_MAX, RADII, l_minus_r_from_excess
from ..departure.tables import Tables, r_min

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "simulation" / "config.yaml"
_R = np.linspace(0.0, 1.0, 4001)
_K = np.arange(1, 81)


@dataclass(frozen=True)
class Rules:
    eps: float
    cv_max: float
    matern_fill: float
    mu1_min: float
    min_pairs: float

    @classmethod
    def load(cls, tables: Tables, path: Path = CONFIG) -> Rules:
        return cls(**yaml.safe_load(path.read_text())["rules"], min_pairs=tables.min_pairs)

    def r_lo(self, nbar: float) -> float:
        return float(r_min(nbar, self.min_pairs))

    @property
    def gauss(self) -> float:
        return 1 / (2 * math.sqrt(math.log(1 / self.eps)))

    @property
    def expo(self) -> float:
        return 1 / math.log(1 / self.eps)


def gauss_step(r, sigma):
    return -np.expm1(-np.asarray(r) ** 2 / (4 * sigma**2))


class Family:
    name = ""
    amplitude = ""
    shape_keys: tuple[str, ...] = ()

    def __init__(self, rules: Rules):
        self.rules = rules

    def shape_box(self, nbar) -> dict[str, tuple[float, float]]:
        return {}

    def valid_shape(self, nbar, shape) -> bool:
        return all(lo <= shape[k] <= hi for k, (lo, hi) in self.shape_box(nbar).items())

    def amp_bounds(self, nbar, shape) -> tuple[float, float]:
        raise NotImplementedError

    def excess(self, r, nbar, shape, amp) -> np.ndarray:
        raise NotImplementedError

    def model(self, nbar, shape, amp) -> dict[str, float]:
        raise NotImplementedError


class Poisson(Family):
    name = "poisson"

    def excess(self, r, nbar, shape, amp):
        return np.zeros_like(np.asarray(r, float))

    def model(self, nbar, shape, amp):
        return {"nbar": nbar}


class Thomas(Family):
    name, amplitude, shape_keys = "thomas", "mu", ("sigma",)

    def shape_box(self, nbar):
        g = self.rules.gauss
        return {"sigma": (g * self.rules.r_lo(nbar), g * R_MAX)}

    def amp_bounds(self, nbar, shape):
        return 1e-6, nbar

    def excess(self, r, nbar, shape, amp):
        return gauss_step(r, shape["sigma"]) * amp / nbar

    def model(self, nbar, shape, amp):
        return {"kappa": nbar / amp, "mu": amp, "sigma": shape["sigma"]}


class Nested(Family):
    name, amplitude, shape_keys = "nested", "mu2", ("sigma2", "rho", "mu1")

    def rho_min(self):
        e = self.rules.eps
        return math.sqrt(math.log(1 / e) / -math.log1p(-e) - 1)

    def shape_box(self, nbar):
        g, lo = self.rules.gauss, self.rules.gauss * self.rules.r_lo(nbar)
        rho_max = math.sqrt((g * R_MAX / lo) ** 2 - 1)
        return {"sigma2": (lo, g * R_MAX / math.hypot(1, self.rho_min())),
                "rho": (self.rho_min(), rho_max),
                "mu1": (self.rules.mu1_min, (1 - self.rules.eps) / self.rules.eps)}

    def valid_shape(self, nbar, shape):
        return super().valid_shape(nbar, shape) and shape["sigma2"] * math.hypot(1, shape["rho"]) <= self.rules.gauss * R_MAX

    def amp_bounds(self, nbar, shape):
        return 1e-6, nbar / shape["mu1"]

    def excess(self, r, nbar, shape, amp):
        s2, rho, mu1 = shape["sigma2"], shape["rho"], shape["mu1"]
        inv_kappa = mu1 * amp / nbar
        return gauss_step(r, s2) * inv_kappa / mu1 + gauss_step(r, s2 * math.hypot(1, rho)) * inv_kappa

    def model(self, nbar, shape, amp):
        s2 = shape["sigma2"]
        return {"kappa": nbar / (shape["mu1"] * amp), "mu1": shape["mu1"], "mu2": amp,
                "sigma1": shape["rho"] * s2, "sigma2": s2}


class LGCP(Family):
    name, amplitude, shape_keys = "lgcp", "sigma2", ("s",)

    def shape_box(self, nbar):
        x = self.rules.expo
        return {"s": (x * self.rules.r_lo(nbar), x * R_MAX)}

    def amp_bounds(self, nbar, shape):
        return 1e-6, 10.0

    def excess(self, r, nbar, shape, amp):
        r = np.asarray(r, float)[..., None]
        a = _K / shape["s"]
        coef = np.exp(_K * math.log(amp) - np.cumsum(np.log(_K)))
        return 2 * np.pi * np.sum(coef * -(np.expm1(-a * r) + a * r * np.exp(-a * r)) / a**2, axis=-1)

    def model(self, nbar, shape, amp):
        return {"mu_log": math.log(nbar) - amp / 2, "sigma2": amp, "s": shape["s"]}


class Matern2(Family):
    name, amplitude = "matern2", "R"

    def amp_bounds(self, nbar, shape):
        return 1e-6, math.sqrt(self.rules.matern_fill / (math.pi * nbar))

    def excess(self, r, nbar, shape, amp):
        R, r = amp, np.asarray(r, float)
        lam_p = self.model(nbar, shape, amp)["lam_p"]
        t = np.linspace(R, 2 * R, 2001)
        cum = np.concatenate([[0.0], np.cumsum(np.diff(t) * np.pi * (t[1:] * _matern_pcf(t[1:], lam_p, R)
                                                                      + t[:-1] * _matern_pcf(t[:-1], lam_p, R)))])
        inner = np.interp(r, t, cum) - np.pi * r**2
        return np.where(r < R, -np.pi * r**2, np.where(r <= 2 * R, inner, cum[-1] - 4 * np.pi * R**2))

    def model(self, nbar, shape, amp):
        x = math.pi * amp**2 * nbar
        return {"R": amp, "lam_p": nbar * -math.log1p(-x) / x}


def _matern_pcf(r, lam_p, R):
    a = np.pi * R**2
    d = np.clip(r / (2 * R), 0.0, 1.0)
    U = 2 * a - 2 * R**2 * (np.arccos(d) - d * np.sqrt(1 - d**2))
    lam = -np.expm1(-lam_p * a) / a
    with np.errstate(divide="ignore", invalid="ignore"):
        rho2 = 2 * (U * -np.expm1(-lam_p * a) - a * -np.expm1(-lam_p * U)) / (a * U * (U - a))
    return np.where(r < R, 0.0, np.where(r >= 2 * R, 1.0, rho2 / lam**2))


FAMILIES: dict[str, type[Family]] = {f.name: f for f in (Poisson, Thomas, Nested, LGCP, Matern2)}


def cv(fam: Family, nbar, shape, amp) -> float:
    """sd(n) / nbar from K via the isotropic set covariogram of W (valid to r = 1)."""
    e = fam.excess(_R, nbar, shape, amp)
    return math.sqrt(1 / nbar + (1 - 3 / np.pi) * e[-1] + np.trapezoid((4 - 2 * _R) / np.pi * e, _R))


def delta(fam: Family, tables: Tables, nbar, shape, amp) -> float:
    return float(tables.delta_tilde(l_minus_r_from_excess(fam.excess(RADII, nbar, shape, amp)), nbar)[0])


def amp_max(fam: Family, nbar, shape) -> float:
    lo, hi = fam.amp_bounds(nbar, shape)
    f = lambda la: cv(fam, nbar, shape, math.exp(la)) - fam.rules.cv_max
    return hi if f(math.log(hi)) <= 0 else math.exp(brentq(f, math.log(lo), math.log(hi), xtol=1e-6))


def solve(fam: Family, tables: Tables, nbar, shape, target) -> float | None:
    lo, hi = fam.amp_bounds(nbar, shape)[0], amp_max(fam, nbar, shape)
    f = lambda la: delta(fam, tables, nbar, shape, math.exp(la)) - target
    return None if f(math.log(hi)) < 0 else math.exp(brentq(f, math.log(lo), math.log(hi), xtol=1e-8))
