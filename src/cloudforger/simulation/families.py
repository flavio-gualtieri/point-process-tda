"""Families: the conditional draw that makes a theta, and the closed-form K(r) - pi r^2.

Each family draws its parameters in model order, every bound conditioned on what is already
drawn. Two independent dials do the work for the cluster families: richness (points per cluster)
says whether the pattern is really a cluster process, and omega = sigma * sqrt(parent intensity)
says whether you can see it -- omega << 1 leaves crisp clusters, omega >> 1 smears them into CSR.

The dials stay independent only because every bound is inverted into the support rather than
rejected on: a rejection on cv would fall almost entirely in the crisp, few-cluster corner and
so would silently reshape the richness dial. See Family.draw_omega, cv_cap and cv_floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import brentq

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "simulation" / "config.yaml"
_R = np.linspace(0.0, 1.0, 4001)
_K = np.arange(1, 81)


@dataclass(frozen=True)
class Rules:
    cv_max: float
    kappa_min: float
    sigma_max: float
    omega: tuple[float, float]
    mu_min: float
    mu2_min: float
    meta_min: float
    rho: tuple[float, float]
    matern_fill: float
    core_min: float
    lgcp_var_min: float
    lgcp_s_min: float
    lgcp_s_max: float

    @classmethod
    def load(cls, path: Path = CONFIG) -> Rules:
        d = yaml.safe_load(path.read_text())["rules"]
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def log_uniform(rng, lo: float, hi: float) -> float:
    return float(lo * (hi / lo) ** rng.random())


def gauss_step(r, sigma):
    return -np.expm1(-np.asarray(r) ** 2 / (4 * sigma**2))


class Family:
    name = ""
    params: tuple[str, ...] = ()      # model keys, in the order the sampler takes them

    def __init__(self, rules: Rules):
        self.rules = rules

    def draw(self, rng, nbar: float) -> dict | None:
        """Model parameters, or None if the bounds leave no room at this nbar."""
        raise NotImplementedError

    def nbar(self, p: dict) -> float:
        raise NotImplementedError

    def excess(self, r, p: dict) -> np.ndarray:
        raise NotImplementedError

    def draw_omega(self, rng, kappa: float, build) -> dict | None:
        """Model with the cluster scale drawn log-uniform on the FEASIBLE omega = sigma sqrt(kappa),
        where `build` turns a sigma into model parameters.

        Both bounds are inverted into the support, never rejected on. sigma_max caps omega from
        above (a cost guard: the parent window grows as (1 + 10 sigma)^2). cv is strictly decreasing
        in omega -- tight clusters concentrate the count variance -- so cv <= cv_max is a LOWER
        bound on omega, and cv_floor solves for it. Rejecting on cv instead would tilt the design
        away from the crisp, few-cluster corner where the rejection concentrates.
        """
        lo, hi = self.rules.omega
        hi = min(hi, self.rules.sigma_max * math.sqrt(kappa))
        if hi <= lo:
            return None
        scale = lambda omega: build(omega / math.sqrt(kappa))
        lo = cv_floor(self, scale, lo, hi)
        return None if lo is None else scale(log_uniform(rng, lo, hi))


class Poisson(Family):
    name, params = "poisson", ("nbar",)

    def draw(self, rng, nbar):
        return {"nbar": nbar}

    def nbar(self, p):
        return p["nbar"]

    def excess(self, r, p):
        return np.zeros_like(np.asarray(r, float))


class Thomas(Family):
    name, params = "thomas", ("kappa", "mu", "sigma")

    def draw(self, rng, nbar):
        hi = nbar / self.rules.kappa_min
        if hi <= self.rules.mu_min:
            return None
        mu = log_uniform(rng, self.rules.mu_min, hi)
        kappa = nbar / mu
        return self.draw_omega(rng, kappa, lambda sigma: {"kappa": kappa, "mu": mu, "sigma": sigma})

    def nbar(self, p):
        return p["kappa"] * p["mu"]

    def excess(self, r, p):
        return gauss_step(r, p["sigma"]) / p["kappa"]


class Nested(Family):
    name, params = "nested", ("kappa", "mu1", "mu2", "sigma1", "sigma2")

    def draw(self, rng, nbar):
        c = self.rules
        hi = nbar / (c.kappa_min * c.mu2_min)
        if hi <= c.mu2_min:
            return None
        mu2 = log_uniform(rng, c.mu2_min, hi)
        lo, hi = max(1.0, c.meta_min / mu2), nbar / (c.kappa_min * mu2)
        if hi <= lo:
            return None
        mu1 = log_uniform(rng, lo, hi)
        kappa = nbar / (mu1 * mu2)
        rho = log_uniform(rng, *c.rho)   # before sigma1: the cv inversion needs the whole model
        return self.draw_omega(rng, kappa, lambda sigma1: {
            "kappa": kappa, "mu1": mu1, "mu2": mu2, "sigma1": sigma1, "sigma2": sigma1 / rho})

    def nbar(self, p):
        return p["kappa"] * p["mu1"] * p["mu2"]

    def excess(self, r, p):
        return (gauss_step(r, p["sigma2"]) / p["mu1"]
                + gauss_step(r, math.hypot(p["sigma1"], p["sigma2"]))) / p["kappa"]


class Matern2(Family):
    name, params = "matern2", ("R", "lam_p")

    def draw(self, rng, nbar):
        core = log_uniform(rng, self.rules.core_min, math.sqrt(self.rules.matern_fill / math.pi))
        R = core / math.sqrt(nbar)
        x = math.pi * R**2 * nbar
        return {"R": R, "lam_p": nbar * -math.log1p(-x) / x}

    def nbar(self, p):
        a = math.pi * p["R"] ** 2
        return -math.expm1(-p["lam_p"] * a) / a

    def excess(self, r, p):
        R, r = p["R"], np.asarray(r, float)
        t = np.linspace(R, 2 * R, 2001)
        pcf = _matern_pcf(t, p["lam_p"], R)
        cum = np.concatenate([[0.0], np.cumsum(np.diff(t) * np.pi * (t[1:] * pcf[1:] + t[:-1] * pcf[:-1]))])
        inner = np.interp(r, t, cum) - np.pi * r**2
        return np.where(r < R, -np.pi * r**2, np.where(r <= 2 * R, inner, cum[-1] - 4 * np.pi * R**2))


class LGCP(Family):
    name, params = "lgcp", ("mu_log", "sigma2", "s", "M")

    def draw(self, rng, nbar):
        c = self.rules
        lo = c.lgcp_s_min / math.sqrt(nbar)
        if lo >= c.lgcp_s_max:
            return None
        s = log_uniform(rng, lo, c.lgcp_s_max)
        build = lambda v: {"mu_log": math.log(nbar) - v / 2, "sigma2": v, "s": s}
        hi = cv_cap(self, build, c.lgcp_var_min, 10.0)
        return None if hi is None else build(log_uniform(rng, c.lgcp_var_min, hi))

    def nbar(self, p):
        return math.exp(p["mu_log"] + p["sigma2"] / 2)

    def excess(self, r, p):
        r = np.asarray(r, float)[..., None]
        a = _K / p["s"]
        coef = np.exp(_K * math.log(p["sigma2"]) - np.cumsum(np.log(_K)))
        return 2 * np.pi * np.sum(coef * -(np.expm1(-a * r) + a * r * np.exp(-a * r)) / a**2, axis=-1)


def _matern_pcf(r, lam_p, R):
    a = np.pi * R**2
    d = np.clip(r / (2 * R), 0.0, 1.0)
    U = 2 * a - 2 * R**2 * (np.arccos(d) - d * np.sqrt(1 - d**2))
    lam = -np.expm1(-lam_p * a) / a
    with np.errstate(divide="ignore", invalid="ignore"):
        rho2 = 2 * (U * -np.expm1(-lam_p * a) - a * -np.expm1(-lam_p * U)) / (a * U * (U - a))
    return np.where(r < R, 0.0, np.where(r >= 2 * R, 1.0, rho2 / lam**2))


FAMILIES: dict[str, type[Family]] = {f.name: f for f in (Poisson, Thomas, Nested, Matern2, LGCP)}


def cv(fam: Family, p: dict) -> float:
    """sd(n) / nbar from K via the isotropic set covariogram of W (valid to r = 1)."""
    e = fam.excess(_R, p)
    return math.sqrt(1 / fam.nbar(p) + (1 - 3 / np.pi) * e[-1] + np.trapezoid((4 - 2 * _R) / np.pi * e, _R))


def cv_cap(fam: Family, build, lo: float, hi: float) -> float | None:
    """Largest value with cv <= cv_max, where `build` turns a value into model parameters."""
    f = lambda v: cv(fam, build(math.exp(v))) - fam.rules.cv_max
    if f(math.log(lo)) > 0:
        return None
    return hi if f(math.log(hi)) <= 0 else math.exp(brentq(f, math.log(lo), math.log(hi), xtol=1e-6))


def cv_floor(fam: Family, build, lo: float, hi: float) -> float | None:
    """Smallest value with cv <= cv_max, for a `build` whose cv DECREASES in the value.

    The mirror of cv_cap. Returns None when even the largest value is too variable, which is a
    genuine infeasibility at this nbar rather than a rejectable draw.
    """
    f = lambda v: cv(fam, build(math.exp(v))) - fam.rules.cv_max
    if f(math.log(hi)) > 0:
        return None
    return lo if f(math.log(lo)) <= 0 else math.exp(brentq(f, math.log(lo), math.log(hi), xtol=1e-6))
