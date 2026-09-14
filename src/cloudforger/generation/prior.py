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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .spec import Spec

# Thomas: r_{1/2} = 2 sqrt(ln 2) sigma, so tau = sqrt(nbar) r_{1/2} = 2 sqrt(ln 2) s.
THOMAS_TAU_PER_S = 2.0 * math.sqrt(math.log(2.0))


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(low * (high / low) ** rng.random())


@dataclass(frozen=True)
class PriorDraw:
    nbar: float
    design: dict[str, float]   # shape coordinates (dimensionless)
    model: dict[str, float]    # model parameters handed to the sampler
    regime: dict[str, Any]     # tau_K, delta_tilde (None until the WP2a null tables exist)
    tries: int                 # shape draws until acceptance (1 for fixed theta)


class FamilyPrior:
    """Shape prior of one family. Subclasses override the four hooks."""

    design_keys: tuple[str, ...] = ()
    model_keys: tuple[str, ...] = ()

    def draw_shape(self, rng: np.random.Generator, nbar: float) -> dict[str, float]:
        return {}

    def accepts(self, nbar: float, shape: dict[str, float]) -> bool:
        return True

    def invert(self, nbar: float, shape: dict[str, float]) -> dict[str, float]:
        return {}

    def regime(self, nbar: float, shape: dict[str, float]) -> dict[str, Any]:
        return {"tau_K": None, "delta_tilde": None}


class PoissonPrior(FamilyPrior):
    """CSR: nbar is the whole parameter vector."""

    def regime(self, nbar, shape):
        return {"tau_K": None, "delta_tilde": 0.0}  # CSR is the delta = 0 anchor by definition


class ThomasPrior(FamilyPrior):
    design_keys = ("mu", "s")
    model_keys = ("kappa", "sigma")

    def __init__(self, mu: dict, s: dict, constraints: dict):
        self.mu_low, self.mu_high = float(mu["low"]), float(mu["high"])
        self.s_low, self.s_high = float(s["low"]), float(s["high"])
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
        return {"tau_K": THOMAS_TAU_PER_S * shape["s"], "delta_tilde": None}

    def acceptance_probability(self, nbar: float) -> float:
        """Closed form: mu and s are independent and each constraint involves
        only one of them, so P(accept) = P(s <= sigma_max sqrt(nbar)) * P(mu <= nbar / kappa_min)."""

        def p_below(x: float, low: float, high: float) -> float:
            return min(1.0, max(0.0, math.log(x / low) / math.log(high / low)))

        return (p_below(self.sigma_max * math.sqrt(nbar), self.s_low, self.s_high)
                * p_below(nbar / self.kappa_min, self.mu_low, self.mu_high))


PRIORS: dict[str, type[FamilyPrior]] = {"poisson": PoissonPrior, "thomas": ThomasPrior}


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
