# src/cloudforger/data_generation/point_processes/inhom_thomas.py

from __future__ import annotations

from typing import Any
import numpy as np

from ...core.base import PointProcess
from ...core.cloud import PointCloud
from ...core.region import Region
from .thomas import ThomasProcess


class _RFFField:
    """n_covariates smooth Gaussian random fields via random Fourier features for
    an RBF kernel, standardized to (mean 0, unit variance) over a reference sample.

    Standardizing over the domain (not over the retained points) is what makes the
    fields "normalized" in a use-case-agnostic way: the background mean is 0, so the
    deviation of the covariate values at observed points away from 0 is exactly the
    signal that identifies beta.
    """

    def __init__(
        self,
        n_covariates: int,
        dimension: int,
        length_scale: float,
        n_modes: int,
        rng: np.random.Generator,
        correlation: float = 0.0,
    ):
        self.n = int(n_covariates)
        self.d = int(dimension)
        # angular frequencies for an RBF kernel of the given length scale
        self.W = rng.normal(scale=1.0 / length_scale, size=(self.n, n_modes, self.d))
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=(self.n, n_modes))
        self.theta = rng.normal(size=(self.n, n_modes))
        self.scale = np.sqrt(2.0 / n_modes)

        # optional cross-covariate correlation (e.g. altitude/soil are correlated)
        if correlation and self.n > 1:
            m = np.full((self.n, self.n), correlation) + (1.0 - correlation) * np.eye(self.n)
            self.L = np.linalg.cholesky(m)
        else:
            self.L = None

        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    def _raw(self, points: np.ndarray) -> np.ndarray:
        p = points.shape[0]
        out = np.empty((p, self.n))
        chunk = 20_000
        for s in range(0, p, chunk):
            pts = points[s : s + chunk]
            proj = np.einsum("nmd,cd->cnm", self.W, pts) + self.b[None]
            out[s : s + chunk] = self.scale * np.einsum("cnm,nm->cn", np.cos(proj), self.theta)
        if self.L is not None:
            out = out @ self.L.T
        return out

    def fit_standardizer(self, reference_points: np.ndarray) -> None:
        raw = self._raw(reference_points)
        self._mu = raw.mean(axis=0)
        self._sigma = raw.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0

    def __call__(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return np.empty((0, self.n))
        return (self._raw(points) - self._mu) / self._sigma


class InhomThomas(PointProcess):
    """Inhomogeneous Thomas process with n covariates.

    Labels for a downstream estimator are (parent_intensity, cluster_scale, beta).
    ``expected_points`` calibrates the offspring rate so the retained count stays
    roughly constant across the sweep, which decouples count from beta (there is no
    identifiable intercept) and keeps training patterns well-behaved.
    """

    def __init__(
        self,
        parent_intensity: float,
        cluster_scale: float,
        beta,
        expected_points: float = 800.0,
        length_scale: float = 0.25,
        n_modes: int = 32,
        covariate_correlation: float = 0.0,
        field_seed: int | None = None,
        edge_buffer: float | None = None,
        reference_size: int = 4096,
        max_base_points: int = 100_000,
    ):
        beta = np.asarray(beta, dtype=float).ravel()
        if parent_intensity <= 0:
            raise ValueError("parent_intensity must be positive")
        if cluster_scale <= 0:
            raise ValueError("cluster_scale must be positive")
        if expected_points <= 0:
            raise ValueError("expected_points must be positive")
        if length_scale <= 0:
            raise ValueError("length_scale must be positive")

        self.parent_intensity = float(parent_intensity)
        self.cluster_scale = float(cluster_scale)
        self.beta = beta
        self.n_covariates = beta.size
        self.expected_points = float(expected_points)
        self.length_scale = float(length_scale)
        self.n_modes = int(n_modes)
        self.covariate_correlation = float(covariate_correlation)
        self.field_seed = field_seed
        self.edge_buffer = edge_buffer
        self.reference_size = int(reference_size)
        self.max_base_points = int(max_base_points)

    @property
    def name(self) -> str:
        return "inhom_thomas"

    @property
    def params(self) -> dict[str, Any]:
        return {
            "parent_intensity": self.parent_intensity,
            "cluster_scale": self.cluster_scale,
            "beta": self.beta.tolist(),
            "n_covariates": self.n_covariates,
            "expected_points": self.expected_points,
            "length_scale": self.length_scale,
            "covariate_correlation": self.covariate_correlation,
        }

    def _sample_with_covariates(
        self, region: Region, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        d = region.dimension

        # Covariate field. field_seed=None -> a fresh field per realization, so the
        # model learns to be field-agnostic; set it to reuse one field across seeds.
        field_rng = rng if self.field_seed is None else np.random.default_rng(self.field_seed)
        field = _RFFField(
            self.n_covariates, d, self.length_scale, self.n_modes, field_rng, self.covariate_correlation
        )

        # Reference sample over the region: standardizes the field and estimates the
        # retention fraction f = mean exp(eta - max eta) used to calibrate the count.
        ref = region.sample_uniform(self.reference_size, rng)
        field.fit_standardizer(ref)
        eta_ref = field(ref) @ self.beta
        eta_max = float(eta_ref.max())
        f = float(np.mean(np.exp(eta_ref - eta_max)))

        # Choose the base offspring rate so that lam_base * f ~= expected_points.
        lam_base = self.expected_points / max(f, 1e-12)
        base_expected = min(lam_base * region.volume, float(self.max_base_points))
        mean_offspring = max(base_expected / (self.parent_intensity * region.volume), 1e-6)

        base = ThomasProcess(
            parent_intensity=self.parent_intensity,
            mean_offspring=mean_offspring,
            cluster_scale=self.cluster_scale,
            edge_buffer=self.edge_buffer,
        )
        candidates = base._sample_points(None, region, rng)
        if candidates.shape[0] == 0:
            return np.empty((0, d)), np.empty((0, self.n_covariates))

        z = field(candidates)
        eta = z @ self.beta
        # Independent thinning; clip guards against candidates above the reference max.
        p = np.minimum(np.exp(eta - eta_max), 1.0)
        keep = rng.random(candidates.shape[0]) < p
        return candidates[keep], z[keep]

    def _sample_points(self, n, region: Region, rng: np.random.Generator) -> np.ndarray:
        points, _ = self._sample_with_covariates(region, rng)
        return points

    def sample(self, n=None, region=None, seed=None) -> PointCloud:
        if region is None and isinstance(n, Region):
            region, n = n, None
        if region is None:
            raise TypeError("Must specify a region to sample from.")

        rng = np.random.default_rng(seed)
        points, covariates = self._sample_with_covariates(region, rng)
        return PointCloud(
            points=points,
            generator_name=self.name,
            generator_params=self.params,
            seed=seed,
            region=region,
            covariates=covariates,
        )