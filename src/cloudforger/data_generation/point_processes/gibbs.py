# src/cloudforger/data_generation/point_processes/gibbs.py
"""Finite Gibbs point processes with a pairwise interaction.

  * StraussProcess       -- density proportional to beta^n(x) * gamma^{S_R(x)},
                            small-scale inhibition (Strauss 1975).
  * LGCPStraussProcess   -- the same inhibition modulated by a log-Gaussian
                            random field, i.e. small-scale regularity on top of
                            larger-scale LGCP clustering (Vihrs et al. 2022).

Neither has a tractable normalizing constant, so both are simulated with a
birth-death Metropolis-Hastings sampler (Geyer & Moller 1994) -- the one
genuinely new simulation mechanism relative to the Neyman-Scott (Poisson
superposition) and Cox (thinning) families already in this package.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from ...core.base import PointProcess
from ...core.region import Region
from .cox import GaussianRandomField


def _birth_death_mh(
    region: Region,
    log_activity: Callable[[np.ndarray], np.ndarray],
    gamma: float,
    radius: float,
    n_steps: int,
    rng: np.random.Generator,
    init_intensity: float,
) -> np.ndarray:
    """Birth-death Metropolis-Hastings for a pairwise-interaction process with
    Papangelou conditional intensity

        lambda(u | x) = exp(log_activity(u)) * gamma ** t(u, x),

    where t(u, x) is the number of points of x within ``radius`` of u; gamma in
    [0, 1] gives inhibition. ``log_activity`` maps an (m, d) array to (m,).
    Runs on ``region`` (already edge-expanded by the caller); returns the final
    configuration. Proposals are 1/2 birth (uniform location) and 1/2 death
    (uniform index); the standard acceptance ratios are beta*gamma^t*|W|/(n+1)
    for a birth and its reciprocal for a death.
    """
    low = np.asarray(region.low, dtype=float)
    span = np.asarray(region.high, dtype=float) - low
    d = region.dimension
    vol = float(region.volume)
    log_vol = math.log(vol)
    r2 = float(radius) * float(radius)
    log_gamma = -math.inf if gamma <= 0.0 else math.log(gamma)

    n = int(rng.poisson(init_intensity * vol))
    cap = max(64, 2 * n + 64)
    pts = np.empty((cap, d))
    if n:
        pts[:n] = low + span * rng.random((n, d))

    coin_birth = rng.random(n_steps) < 0.5
    u_unit = rng.random((n_steps, d))
    death_u = rng.random(n_steps)
    with np.errstate(divide="ignore"):
        log_accept = np.log(rng.random(n_steps))

    for k in range(n_steps):
        if coin_birth[k]:
            u = low + span * u_unit[k]
            t = 0 if n == 0 else int(np.count_nonzero(((pts[:n] - u) ** 2).sum(1) <= r2))
            pen = 0.0 if t == 0 else t * log_gamma
            log_ratio = float(log_activity(u[None, :])[0]) + pen + log_vol - math.log(n + 1)
            if log_accept[k] < log_ratio:
                if n == cap:
                    cap *= 2
                    bigger = np.empty((cap, d))
                    bigger[:n] = pts[:n]
                    pts = bigger
                pts[n] = u
                n += 1
        elif n > 0:
            i = int(death_u[k] * n)
            pi = pts[i].copy()
            t = int(np.count_nonzero(((pts[:n] - pi) ** 2).sum(1) <= r2)) - 1
            pen = 0.0 if t == 0 else t * log_gamma
            log_ratio = math.log(n) - float(log_activity(pi[None, :])[0]) - pen - log_vol
            if log_accept[k] < log_ratio:
                pts[i] = pts[n - 1]
                n -= 1

    return pts[:n].copy()


def _adaptive_steps(n_steps: int | None, expected_count: float) -> int:
    """~40 birth-death sweeps from an already-well-scaled start -- ample for the
    weak Strauss interaction here, cheap enough for a Vihrs-sized sweep. Pass an
    explicit ``n_steps`` (e.g. Vihrs' 100k) for strong interaction / publication."""
    return int(n_steps) if n_steps is not None else max(15_000, int(40 * expected_count))


class StraussProcess(PointProcess):
    """Strauss process: density proportional to ``beta^n(x) * gamma^{S_R(x)}``
    w.r.t. a unit-rate Poisson process, where ``S_R(x)`` is the number of point
    pairs closer than ``radius`` and ``gamma`` in [0, 1] controls inhibition
    (``gamma = 1`` or ``radius = 0`` -> Poisson). Labels ``(beta, gamma, radius)``;
    Vihrs (2022) Sec. 3.2. Simulated on the window grown by ``margin_factor *
    radius`` (Vihrs uses 2), then clipped back.
    """

    def __init__(
        self,
        beta: float,
        gamma: float,
        radius: float,
        n_steps: int | None = None,
        margin_factor: float = 2.0,
    ):
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.radius = float(radius)
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.radius < 0:
            raise ValueError("radius must be non-negative")
        self.n_steps = n_steps
        self.margin_factor = float(margin_factor)

    @property
    def name(self) -> str:
        return "strauss"

    @property
    def params(self) -> dict[str, Any]:
        return {"beta": self.beta, "gamma": self.gamma, "radius": self.radius}

    def _sample_points(self, n, region: Region, rng: np.random.Generator) -> np.ndarray:
        sim_region = region.expanded(self.margin_factor * self.radius)
        log_beta = math.log(self.beta)
        pts = _birth_death_mh(
            sim_region,
            lambda p: np.full(p.shape[0], log_beta),
            self.gamma,
            self.radius,
            _adaptive_steps(self.n_steps, self.beta * sim_region.volume),
            rng,
            init_intensity=self.beta,
        )
        return pts[region.contains(pts)]


class LGCPStraussProcess(PointProcess):
    """LGCP-Strauss process (Vihrs et al. 2022): Strauss-type small-scale
    inhibition ``(gamma, radius)`` whose activity is modulated by a log-Gaussian
    random field ``exp(mu + Y(u))`` with ``Y`` of variance ``sigma2`` and
    correlation length ``s``. Collapses to an LGCP if ``gamma = 1`` / ``radius =
    0`` and to a Strauss process if ``sigma2 = 0``. Labels ``(mu, sigma2, s,
    gamma, radius)``.
    """

    def __init__(
        self,
        mu: float,
        sigma2: float,
        s: float,
        gamma: float,
        radius: float,
        covariance: str = "exp",
        n_modes: int = 512,
        field_seed: int | None = None,
        n_steps: int | None = None,
        margin_factor: float = 2.0,
    ):
        self.mu = float(mu)
        self.sigma2 = float(sigma2)
        self.s = float(s)
        self.gamma = float(gamma)
        self.radius = float(radius)
        if self.sigma2 < 0:
            raise ValueError("sigma2 must be non-negative")
        if self.s <= 0:
            raise ValueError("s must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.radius < 0:
            raise ValueError("radius must be non-negative")
        self.covariance = str(covariance)
        self.n_modes = int(n_modes)
        self.field_seed = field_seed
        self.n_steps = n_steps
        self.margin_factor = float(margin_factor)

    @property
    def name(self) -> str:
        return "lgcp_strauss"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "mu": self.mu,
            "sigma2": self.sigma2,
            "s": self.s,
            "gamma": self.gamma,
            "radius": self.radius,
            "covariance": self.covariance,
        }

    def _sample_points(self, n, region: Region, rng: np.random.Generator) -> np.ndarray:
        sim_region = region.expanded(self.margin_factor * self.radius)
        field_rng = rng if self.field_seed is None else np.random.default_rng(self.field_seed)
        field = GaussianRandomField(
            sim_region.dimension, self.sigma2, self.s, field_rng,
            n_modes=self.n_modes, covariance=self.covariance,
        )
        expected = math.exp(self.mu + 0.5 * self.sigma2)
        pts = _birth_death_mh(
            sim_region,
            lambda p: self.mu + field(p),
            self.gamma,
            self.radius,
            _adaptive_steps(self.n_steps, expected * sim_region.volume),
            rng,
            init_intensity=expected,
        )
        return pts[region.contains(pts)]
