# src/cloudforger/generation/prior.py
"""The DV3 prior: draw n-bar, draw the shape, reject bad shapes, invert
(generation.tex, "The prior at a glance" and the box "How one prior draw is made").

The order of draws from a case's PARAMS stream is part of the dataset
definition -- change it and every theta changes:

    1. u ~ U(0,1), nbar = low * (high/low)**u     (one draw, never repeated)
    2. the family's shape coordinates, in the order of its draw_shape
    3. if the constraints fail at this nbar, repeat 2 (and only 2)

Every log-uniform coordinate uses the same transform, log_uniform below.
Fixed-theta sets (B, C) skip the drawing and call `fixed` instead.

Each family also knows its closed-form excess K(r) - pi r^2 (for delta-tilde,
nulls.py; kept separate from pi r^2 so near-CSR values lose no precision), its
structure scale tau in spacings, and any per-case numerical setting the
sampler needs (the LGCP grid).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from .spec import Spec

# Gaussian clusters: |K - pi r^2| reaches half its limit at r = 2 sqrt(ln 2) sigma.
HALF_STEP = 2.0 * math.sqrt(math.log(2.0))
THOMAS_TAU_PER_S = HALF_STEP


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(low * (high / low) ** rng.random())


@dataclass(frozen=True)
class PriorDraw:
    nbar: float
    design: dict[str, float]    # shape coordinates (dimensionless)
    model: dict[str, float]     # model parameters handed to the sampler
    regime: dict[str, Any]      # tau_K (and tau_K2); delta_tilde is added by the plan
    tries: int                  # shape draws until acceptance (1 for fixed theta)
    numerics: dict[str, Any] = field(default_factory=dict)   # e.g. grid_M, set by the plan


class FamilyPrior:
    """Shape prior of one family. Subclasses override the hooks they need."""

    design_keys: tuple[str, ...] = ()
    model_keys: tuple[str, ...] = ()

    def draw_shape(self, rng: np.random.Generator, nbar: float) -> dict[str, float]:
        return {}

    def accepts(self, nbar: float, shape: dict[str, float]) -> bool:
        return True

    def invert(self, nbar: float, shape: dict[str, float]) -> dict[str, float]:
        return {}

    def regime(self, nbar: float, shape: dict[str, float]) -> dict[str, Any]:
        return {}

    def excess(self, r: np.ndarray, nbar: float, shape: dict[str, float]) -> np.ndarray:
        """K(r) - pi r^2 in closed form (0 for CSR)."""
        return np.zeros_like(np.asarray(r, float))

    def K(self, r: np.ndarray, nbar: float, shape: dict[str, float]) -> np.ndarray:
        return np.pi * np.asarray(r, float) ** 2 + self.excess(r, nbar, shape)

    def numerics(self, nbar: float, shape: dict[str, float], tabs: dict[str, Any]) -> dict[str, Any]:
        return {}


def _lu(spec: dict) -> tuple[float, float]:
    return float(spec["low"]), float(spec["high"])


def _gauss_step(r: np.ndarray, sigma: float) -> np.ndarray:
    return 1.0 - np.exp(-np.asarray(r, float) ** 2 / (4.0 * sigma**2))


# --- Poisson ----------------------------------------------------------------------

class PoissonPrior(FamilyPrior):
    """CSR: nbar is the whole parameter vector."""


# --- Thomas -----------------------------------------------------------------------

class ThomasPrior(FamilyPrior):
    design_keys = ("mu", "s")
    model_keys = ("kappa", "sigma")

    def __init__(self, mu: dict, s: dict, constraints: dict):
        self.mu_low, self.mu_high = _lu(mu)
        self.s_low, self.s_high = _lu(s)
        self.sigma_max = float(constraints["sigma_max"])
        self.kappa_min = float(constraints["kappa_min"])

    def draw_shape(self, rng, nbar):
        mu = log_uniform(rng, self.mu_low, self.mu_high)
        s = log_uniform(rng, self.s_low, self.s_high)
        return {"mu": mu, "s": s}

    def accepts(self, nbar, shape):
        model = self.invert(nbar, shape)
        return model["sigma"] <= self.sigma_max and model["kappa"] >= self.kappa_min

    def invert(self, nbar, shape):
        return {"kappa": nbar / shape["mu"], "sigma": shape["s"] / math.sqrt(nbar)}

    def regime(self, nbar, shape):
        return {"tau_K": THOMAS_TAU_PER_S * shape["s"]}

    def excess(self, r, nbar, shape):
        m = self.invert(nbar, shape)
        return _gauss_step(r, m["sigma"]) / m["kappa"]

    def acceptance_probability(self, nbar: float) -> float:
        """Closed form: mu and s are independent and each constraint involves
        only one of them, so P(accept) = P(s <= sigma_max sqrt(nbar)) * P(mu <= nbar / kappa_min)."""

        def p_below(x: float, low: float, high: float) -> float:
            return min(1.0, max(0.0, math.log(x / low) / math.log(high / low)))

        return (p_below(self.sigma_max * math.sqrt(nbar), self.s_low, self.s_high)
                * p_below(nbar / self.kappa_min, self.mu_low, self.mu_high))


# --- nested Thomas ------------------------------------------------------------------

class NestedPrior(FamilyPrior):
    """Meta-parents (kappa) -> Poisson(mu1) parents at N(0, sigma1^2) -> Poisson(mu2)
    children at N(0, sigma2^2). Design: s2 = sigma2 sqrt(nbar), rho = sigma1 / sigma2."""

    design_keys = ("mu1", "mu2", "s2", "rho")
    model_keys = ("kappa", "sigma", "sigma1")   # sigma = inner (child) scale sigma2

    def __init__(self, mu1: dict, mu2: dict, s2: dict, rho: dict, constraints: dict):
        self.ranges = {"mu1": _lu(mu1), "mu2": _lu(mu2), "s2": _lu(s2), "rho": _lu(rho)}
        self.kappa_min = float(constraints["kappa_min"])
        self.outer_max = float(constraints["outer_max"])

    def draw_shape(self, rng, nbar):
        return {k: log_uniform(rng, *self.ranges[k]) for k in self.design_keys}

    def accepts(self, nbar, shape):
        m = self.invert(nbar, shape)
        return (m["kappa"] >= self.kappa_min
                and 2.0 * math.hypot(m["sigma1"], m["sigma"]) <= self.outer_max)

    def invert(self, nbar, shape):
        sigma2 = shape["s2"] / math.sqrt(nbar)
        return {"kappa": nbar / (shape["mu1"] * shape["mu2"]), "sigma": sigma2, "sigma1": shape["rho"] * sigma2}

    def regime(self, nbar, shape):
        # one scale per step of K: inner at sigma2, outer at sqrt(sigma1^2 + sigma2^2)
        return {"tau_K": HALF_STEP * shape["s2"], "tau_K2": HALF_STEP * shape["s2"] * math.hypot(1.0, shape["rho"])}

    def excess(self, r, nbar, shape):
        m = self.invert(nbar, shape)
        return (_gauss_step(r, m["sigma"]) / (m["kappa"] * shape["mu1"])
                + _gauss_step(r, math.hypot(m["sigma1"], m["sigma"])) / m["kappa"])


# --- Matern hard-core type II ---------------------------------------------------------

def matern2_pcf(r: np.ndarray, lam_p: float, R: float) -> np.ndarray:
    """Pair correlation of Matern II (notes, eq. 3): 0 below R, 1 beyond 2R."""
    r = np.asarray(r, float)
    a = np.pi * R**2
    d = np.clip(r / (2.0 * R), 0.0, 1.0)
    U = 2.0 * a - 2.0 * R**2 * (np.arccos(d) - d * np.sqrt(1.0 - d**2))   # |b(0,R) u b(r,R)|
    with np.errstate(divide="ignore", invalid="ignore"):
        rho2 = 2.0 * (U * -np.expm1(-lam_p * a) - a * -np.expm1(-lam_p * U)) / (a * U * (U - a))
    lam = -np.expm1(-lam_p * a) / a
    return np.where(r < R, 0.0, np.where(r >= 2.0 * R, 1.0, rho2 / lam**2))


class Matern2Prior(FamilyPrior):
    """Design tau = R sqrt(nbar); lam_p from pi R^2 lambda = 1 - exp(-lam_p pi R^2)."""

    design_keys = ("tau",)
    model_keys = ("R", "lam_p")

    def __init__(self, tau: dict):
        self.tau_low, self.tau_high = _lu(tau)
        if math.pi * self.tau_high**2 >= 1.0:
            raise ValueError("Matern II needs pi tau^2 < 1 (tau < 0.564)")

    def draw_shape(self, rng, nbar):
        return {"tau": log_uniform(rng, self.tau_low, self.tau_high)}

    def invert(self, nbar, shape):
        x = math.pi * shape["tau"] ** 2
        return {"R": shape["tau"] / math.sqrt(nbar), "lam_p": nbar * -math.log1p(-x) / x}

    def _K_pieces(self, nbar, shape, n_fine=4001):
        m = self.invert(nbar, shape)
        R = m["R"]
        t = np.linspace(R, 2.0 * R, n_fine)
        cum = cumulative_trapezoid(2.0 * np.pi * t * matern2_pcf(t, m["lam_p"], R), t, initial=0.0)
        return R, t, cum

    def excess(self, r, nbar, shape):
        R, t, cum = self._K_pieces(nbar, shape)
        r = np.asarray(r, float)
        return np.where(r < R, -np.pi * r**2, np.where(r <= 2.0 * R, np.interp(r, t, cum) - np.pi * r**2,
                                                       cum[-1] - 4.0 * np.pi * R**2))

    def regime(self, nbar, shape):
        # Excess K - pi r^2 is -pi r^2 below R and constant beyond 2R; its half point lies below R.
        R, _, cum = self._K_pieces(nbar, shape)
        limit = cum[-1] - 4.0 * np.pi * R**2
        r_half = math.sqrt(abs(limit) / (2.0 * math.pi))
        if r_half > R:
            raise ValueError("Matern II half point beyond R; closed form for tau_K does not apply")
        return {"tau_K": r_half * math.sqrt(nbar)}


# --- log-Gaussian Cox process -------------------------------------------------------------

_LGCP_TERMS = np.arange(1, 81)


def _lgcp_coeffs(sigma2: float) -> np.ndarray:
    # sigma2^k / k!, from g = exp(sigma2 e^{-r/s}) = sum_k sigma2^k e^{-k r / s} / k!
    return np.exp(_LGCP_TERMS * math.log(sigma2) - np.cumsum(np.log(_LGCP_TERMS)))


def lgcp_excess(r: np.ndarray, sigma2: float, s: float) -> np.ndarray:
    """K(r) - pi r^2 for covariance sigma2 exp(-r/s) (series, 80 terms)."""
    r = np.asarray(r, float)[..., None]
    a = _LGCP_TERMS / s
    return 2.0 * np.pi * np.sum(_lgcp_coeffs(sigma2) * -(np.expm1(-a * r) + a * r * np.exp(-a * r)) / a**2, axis=-1)


class LGCPPrior(FamilyPrior):
    """Lambda = exp(mu + Y), Y ~ GP(0, sigma2 exp(-r/s)). Design (sigma2, sp = s sqrt(nbar))."""

    design_keys = ("sigma2", "sp")
    model_keys = ("mu_log", "s_abs")

    def __init__(self, sigma2: dict, sp: dict, constraints: dict, grid: dict):
        self.sigma2_low, self.sigma2_high = _lu(sigma2)
        self.sp_low, self.sp_high = _lu(sp)
        self.s_max = float(constraints["s_max"])
        self.grid = {"tol": float(grid["tol"]), "M_min": int(grid["M_min"]), "M_max": int(grid["M_max"])}

    def draw_shape(self, rng, nbar):
        sigma2 = log_uniform(rng, self.sigma2_low, self.sigma2_high)
        sp = log_uniform(rng, self.sp_low, self.sp_high)
        return {"sigma2": sigma2, "sp": sp}

    def accepts(self, nbar, shape):
        return self.invert(nbar, shape)["s_abs"] <= self.s_max

    def invert(self, nbar, shape):
        return {"mu_log": math.log(nbar) - shape["sigma2"] / 2.0, "s_abs": shape["sp"] / math.sqrt(nbar)}

    def excess(self, r, nbar, shape):
        return lgcp_excess(r, shape["sigma2"], self.invert(nbar, shape)["s_abs"])

    def regime(self, nbar, shape):
        # Excess rises monotonically to 2 pi s^2 sum_k sigma2^k / (k! k^2); tau = sqrt(nbar) r_half = sp r_half / s.
        sigma2, s = shape["sigma2"], 1.0     # r_half / s does not depend on s
        limit = 2.0 * np.pi * float(np.sum(_lgcp_coeffs(sigma2) / _LGCP_TERMS**2))
        r_half = brentq(lambda x: float(lgcp_excess(x, sigma2, s)) - 0.5 * limit, 1e-9, 50.0, xtol=1e-12)
        return {"tau_K": r_half * shape["sp"]}

    def numerics(self, nbar, shape, tabs):
        from .lgcp_grid import choose_grid_M
        M, _ = choose_grid_M(shape["sigma2"], self.invert(nbar, shape)["s_abs"], nbar, tabs, **self.grid)
        return {"grid_M": M}


PRIORS: dict[str, type[FamilyPrior]] = {
    "poisson": PoissonPrior,
    "thomas": ThomasPrior,
    "nested": NestedPrior,
    "matern2": Matern2Prior,
    "lgcp": LGCPPrior,
}


def build_priors(spec: Spec) -> dict[str, FamilyPrior]:
    missing = sorted(set(spec.families) - set(PRIORS))
    if missing:
        raise NotImplementedError(f"no prior implemented for {missing}")
    return {name: PRIORS[name](**block) for name, block in spec.families.items()}


def _finish(prior: FamilyPrior, nbar: float, shape: dict[str, float], tries: int) -> PriorDraw:
    return PriorDraw(nbar, dict(shape), prior.invert(nbar, shape), prior.regime(nbar, shape), tries)


def draw(prior: FamilyPrior, rng: np.random.Generator, nbar_low: float, nbar_high: float,
         max_tries: int = 10_000) -> PriorDraw:
    """One prior draw from a case's PARAMS stream (training and A)."""
    nbar = log_uniform(rng, nbar_low, nbar_high)
    for tries in range(1, max_tries + 1):
        shape = prior.draw_shape(rng, nbar)
        if prior.accepts(nbar, shape):
            return _finish(prior, nbar, shape, tries)
    raise RuntimeError(f"{type(prior).__name__}: no shape accepted in {max_tries} tries at nbar={nbar:.1f}")


def fixed(prior: FamilyPrior, nbar: float, shape: dict[str, float]) -> PriorDraw:
    """A fixed theta (B cells, C levels): same constraints and inversion, no randomness."""
    if not prior.accepts(nbar, shape):
        raise ValueError(f"{type(prior).__name__}: shape {shape} violates the constraints at nbar={nbar}")
    return _finish(prior, nbar, shape, tries=1)
