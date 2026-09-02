# src/cloudforger/data_generation/point_processes/cox.py
"""Log-Gaussian Cox process (LGCP) and its Gaussian-random-field engine.

An LGCP is a doubly-stochastic Poisson process whose random intensity is
``Lambda(u) = exp(mu + Y(u))`` for a stationary zero-mean Gaussian random
field ``Y`` with marginal variance ``sigma2`` and correlation length ``s``
(Moller et al. 1998; the parametrization used in Vihrs 2022, Sec. 3.1).

Unlike the Neyman-Scott family in neyman_scott.py there is no parent /
offspring structure: points inside the window depend only on the field
inside the window, so -- in contrast to every cluster process here -- no
edge buffer is needed.

The field is drawn with random Fourier features (same idea as
inhom_thomas._RFFField, but standalone, variance-controlled, and offering
the exponential / Matern-1/2 covariance Vihrs uses, not only the RBF one).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.base import PointProcess
from ...core.region import Region


class GaussianRandomField:
    """Stationary zero-mean Gaussian random field on R^d, marginal variance
    ``variance`` and correlation length ``scale``, approximated by
    ``n_modes`` random Fourier features.

    covariance:
      "exp" -- k(h) = exp(-||h|| / scale)          (Matern-1/2; Vihrs' choice)
      "rbf" -- k(h) = exp(-||h||^2 / (2 scale^2))  (squared exponential)

    One instance is one fixed realization of the field; call it on an
    ``(m, d)`` array of points to get the ``(m,)`` field values.
    """

    def __init__(
        self,
        dimension: int,
        variance: float,
        scale: float,
        rng: np.random.Generator,
        n_modes: int = 512,
        covariance: str = "exp",
    ):
        d = int(dimension)
        variance = float(variance)
        scale = float(scale)
        if variance < 0:
            raise ValueError("variance must be non-negative")
        if scale <= 0:
            raise ValueError("scale must be positive")
        n_modes = int(n_modes)

        if covariance == "rbf":
            # spectral density of exp(-||h||^2 / (2 scale^2)) is N(0, scale^-2 I)
            w = rng.normal(scale=1.0 / scale, size=(n_modes, d))
        elif covariance == "exp":
            # spectral density of exp(-||h|| / scale) is a multivariate t with
            # 1 d.o.f. and shape scale^-2 I, i.e. w = Z / sqrt(G) with
            # Z ~ N(0, scale^-2 I) and G ~ chi^2_1 (holds for any d).
            z = rng.normal(scale=1.0 / scale, size=(n_modes, d))
            g = rng.chisquare(1.0, size=(n_modes, 1))
            w = z / np.sqrt(g)
        else:
            raise ValueError(f"unknown covariance {covariance!r}; use 'exp' or 'rbf'")

        self.covariance = covariance
        self.variance = variance
        self.scale = scale
        self._w = w
        self._b = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)
        # deterministic amplitude, random phase only (Rahimi & Recht): the
        # marginal variance is then n_modes-independent, not itself random.
        self._norm = np.sqrt(2.0 * variance / n_modes)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        m = points.shape[0]
        if m == 0:
            return np.empty(0)
        out = np.empty(m)
        chunk = 20_000  # cap the (chunk, n_modes) cos() temporary, cf. _RFFField._raw
        for i in range(0, m, chunk):
            proj = points[i : i + chunk] @ self._w.T + self._b
            out[i : i + chunk] = self._norm * np.cos(proj).sum(axis=1)
        return out


class LGCPProcess(PointProcess):
    """Log-Gaussian Cox process with constant log-intensity mean ``mu`` and
    an exponential-covariance Gaussian random field of variance ``sigma2``
    and correlation length ``s``. Expected point count per unit area is
    ``exp(mu + sigma2 / 2)``. Estimation labels: ``(mu, sigma2, s)``.
    """

    def __init__(
        self,
        mu: float,
        sigma2: float,
        s: float,
        covariance: str = "exp",
        n_modes: int = 512,
        field_seed: int | None = None,
        probe_size: int = 8192,
    ):
        self.mu = float(mu)
        self.sigma2 = float(sigma2)
        self.s = float(s)
        if self.sigma2 < 0:
            raise ValueError("sigma2 must be non-negative")
        if self.s <= 0:
            raise ValueError("s must be positive")
        self.covariance = str(covariance)
        self.n_modes = int(n_modes)
        self.field_seed = field_seed
        self.probe_size = int(probe_size)

    @property
    def name(self) -> str:
        return "lgcp"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "mu": self.mu,
            "sigma2": self.sigma2,
            "s": self.s,
            "covariance": self.covariance,
            "expected_intensity": float(np.exp(self.mu + 0.5 * self.sigma2)),
        }

    def _sample_points(self, n, region: Region, rng: np.random.Generator) -> np.ndarray:
        # field_seed=None -> a fresh field per realization (the field is a
        # nuisance the estimator must marginalize); set it to reuse one field.
        field_rng = rng if self.field_seed is None else np.random.default_rng(self.field_seed)
        field = GaussianRandomField(
            region.dimension, self.sigma2, self.s, field_rng,
            n_modes=self.n_modes, covariance=self.covariance,
        )

        # Thinning: bound the log-intensity by a uniform probe of the window
        # plus a half-e-fold cushion; the rare points above the bound are just
        # kept w.p. 1 (same tolerance as inhom_thomas' eta_max clip).
        probe = region.sample_uniform(self.probe_size, rng)
        log_lam_max = float((self.mu + field(probe)).max()) + 0.5

        n_pois = int(rng.poisson(np.exp(log_lam_max) * region.volume))
        if n_pois == 0:
            return np.empty((0, region.dimension))
        cand = region.sample_uniform(n_pois, rng)
        keep_p = np.minimum(np.exp(self.mu + field(cand) - log_lam_max), 1.0)
        return cand[rng.random(n_pois) < keep_p]
