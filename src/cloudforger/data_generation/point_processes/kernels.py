# src/cloudforger/data_generation/point_processes/kernels.py
"""Offspring displacement kernels for Neyman-Scott cluster processes.

Each kernel bundles two things the process engine needs and that must never
disagree: how to draw an offspring displacement (`sample`) and how wide an
edge buffer that displacement law implies (`support_radius`).

Kernel shape parameters (a scale, a radius) are frozen at construction; the
ambient `dimension` is NOT -- it is a property of the region being sampled
and is passed per call, matching the (n, dimension, rng) signature
NeymanScottProcess already uses for its displacement sampler. This keeps
the kernels -- and every process built on them -- dimension-generic, so
nothing here forces a `dimension` argument onto the process constructors or
onto design.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.stats import norm


class Kernel(ABC):

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Kernel parameters, merged into PointProcess.params (and thus
        selectable as training labels). Values must be plain Python floats,
        not numpy scalars, so the YAML manifest dump does not choke."""
        ...

    @abstractmethod
    def sample(self, n: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
        """`n` iid displacement vectors, shape (n, dimension)."""
        ...

    @abstractmethod
    def support_radius(self, eps: float = 1e-4) -> float:
        """Edge-buffer width implied by this kernel: a parent this far
        outside the window can still place offspring inside it with more
        than `eps` per-axis probability. For compact-support kernels this
        is the exact support and `eps` is ignored."""
        ...


class GaussianKernel(Kernel):
    """Isotropic Gaussian displacement, N(0, sigma^2 I_d). Dimension-agnostic:
    the same sigma applies on every axis of whatever dimension `sample` is
    called with. Per-axis / rotated covariance is AnisotropicGaussianKernel
    below (2-D only)."""

    def __init__(self, sigma: float):
        sigma = float(sigma)
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = sigma

    @property
    def params(self) -> dict[str, Any]:
        return {"cluster_scale": float(self.sigma)}

    def sample(self, n: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(scale=self.sigma, size=(n, dimension))

    def support_radius(self, eps: float = 1e-4) -> float:
        # eps = per-axis tail mass left outside the buffer:
        # P(|X_axis| > r) = eps  ->  r = sigma * Phi^{-1}(1 - eps).
        # eps=1e-4 gives ~3.72 sigma; the historical hardcoded 4 sigma is
        # eps ~ 3.2e-5 -- see the ThomasProcess migration note.
        return float(self.sigma * norm.isf(eps))


class AnisotropicGaussianKernel(Kernel):
    """Zero-mean Gaussian displacement with a full 2-D covariance: an
    axis-aligned (sigma_1, sigma_2) spread rotated by `theta` radians,

        Sigma = R(theta) @ diag(sigma_1**2, sigma_2**2) @ R(theta).T,

    with R(theta) rotating the axis-1 direction anticlockwise from +x.
    Sampling: z ~ N(0, I_2), scale the axes by (sigma_1, sigma_2), rotate.
    `sigma_1 == sigma_2` recovers the isotropic GaussianKernel and `theta`
    then has no effect.

    2-D only -- rotation is a planar notion -- so `sample` raises for
    dimension != 2. Parameter degeneracies to resolve at the config / label
    layer (not here, so the kernel stays a plain building block):
        (sigma_1, sigma_2, theta) ~ (sigma_2, sigma_1, theta +- pi/2)
        theta ~ theta + pi
    e.g. constrain sigma_1 >= sigma_2 and theta in [0, pi).
    """

    def __init__(self, sigma_1: float, sigma_2: float, theta: float = 0.0):
        sigma_1 = float(sigma_1)
        sigma_2 = float(sigma_2)
        theta = float(theta)
        if sigma_1 <= 0 or sigma_2 <= 0:
            raise ValueError("sigma_1 and sigma_2 must be positive")
        if not np.isfinite(theta):
            raise ValueError("theta must be finite")
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.theta = theta

    @property
    def params(self) -> dict[str, Any]:
        return {
            "cluster_sigma_1": float(self.sigma_1),
            "cluster_sigma_2": float(self.sigma_2),
            "cluster_theta": float(self.theta),
        }

    def sample(self, n: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
        if dimension != 2:
            raise ValueError(
                f"AnisotropicGaussianKernel is 2-D only, got dimension={dimension}"
            )
        z = rng.normal(size=(n, 2)) * np.array([self.sigma_1, self.sigma_2])
        c, s = np.cos(self.theta), np.sin(self.theta)
        rot = np.array([[c, -s], [s, c]])
        return z @ rot.T

    def support_radius(self, eps: float = 1e-4) -> float:
        # Rotation only reorients the ellipse, so the widest per-axis std in
        # the ambient frame is max(sigma_1, sigma_2); reuse GaussianKernel's
        # per-axis tail-mass convention r = sigma * Phi^{-1}(1 - eps).
        return float(max(self.sigma_1, self.sigma_2) * norm.isf(eps))


class BallKernel(Kernel):
    """Uniform displacement inside the ball of the given radius -- the Matern
    cluster process kernel. Compact support, so support_radius is the radius
    exactly and `eps` is ignored (do NOT add a tail-mass margin here)."""

    def __init__(self, cluster_radius: float):
        cluster_radius = float(cluster_radius)
        if cluster_radius <= 0:
            raise ValueError("cluster_radius must be positive")
        self.cluster_radius = cluster_radius

    @property
    def params(self) -> dict[str, Any]:
        return {"cluster_radius": float(self.cluster_radius)}

    def sample(self, n: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
        if n == 0:
            return np.empty((0, dimension))

        directions = rng.normal(size=(n, dimension))
        norms = np.linalg.norm(directions, axis=1)

        # rng.normal can (astronomically rarely) return an exact zero vector
        # -> 0/0 -> NaN direction; resample those until non-zero, matching
        # the guard in the original uniform_ball_displacements.
        zero = norms == 0
        while np.any(zero):
            directions[zero] = rng.normal(size=(int(zero.sum()), dimension))
            norms = np.linalg.norm(directions, axis=1)
            zero = norms == 0

        directions = directions / norms[:, None]
        radii = self.cluster_radius * rng.random(n) ** (1.0 / dimension)
        return directions * radii[:, None]

    def support_radius(self, eps: float = 1e-4) -> float:
        return float(self.cluster_radius)
