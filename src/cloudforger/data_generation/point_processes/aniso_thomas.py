# src/cloudforger/data_generation/point_processes/aniso_thomas.py
"""Thomas process with an anisotropic (elliptical) Gaussian offspring kernel
and a homogeneous parent process.

Identical to ThomasProcess except the offspring displacement law is
AnisotropicGaussianKernel: an axis-aligned (sigma_1, sigma_2) Gaussian
rotated by `cluster_theta`, with

    sigma_1 = cluster_scale * sqrt(cluster_aspect)   (long axis, aspect >= 1)
    sigma_2 = cluster_scale / sqrt(cluster_aspect)   (short axis)

so the geometric-mean scale sqrt(sigma_1 sigma_2) == cluster_scale is held
fixed. A design therefore reuses ThomasProcess's cluster_scale range
unchanged and adds cluster_aspect (>= 1) and cluster_theta (in [0, pi) --
the ellipse orientation is pi-periodic) as two new axes. cluster_aspect == 1
reproduces the isotropic ThomasProcess exactly.

`c1` mirrors ThomasProcess's overlap-index diagnostic nu = 2 r sqrt(kappa)
using the geometric-mean per-axis RMS displacement r = cluster_scale, so a
design's sampled overlap index is recoverable as a self-check and stays
comparable across the isotropic and anisotropic Thomas families.
"""

from __future__ import annotations

import math

from .neyman_scott import NeymanScottProcess, poisson_counts
from .kernels import AnisotropicGaussianKernel, GaussianKernel


class AnisotropicThomasProcess(NeymanScottProcess):

    def __init__(
        self,
        parent_intensity: float,
        mean_offspring: float,
        cluster_scale: float,
        cluster_aspect: float = 1.0,
        cluster_theta: float = 0.0,
        edge_buffer: float | None = None,
    ):
        if cluster_scale <= 0:
            raise ValueError("cluster_scale must be positive")
        if cluster_aspect < 1.0:
            raise ValueError("cluster_aspect must be >= 1 (sigma_1 is the long axis)")

        root = math.sqrt(cluster_aspect)
        if cluster_aspect == 1.0:
            kernel = GaussianKernel(cluster_scale)
        else:
            kernel = AnisotropicGaussianKernel(
                sigma_1=cluster_scale * root,
                sigma_2=cluster_scale / root,
                theta=cluster_theta,
            )

        super().__init__(
            parent_intensity=parent_intensity,
            kernel=kernel,
            offspring_count_sampler=poisson_counts(mean_offspring),
            edge_buffer=edge_buffer,
            process_name="aniso_thomas",
            param_dict={
                "mean_offspring": float(mean_offspring),
                "cluster_scale": float(cluster_scale),
                "cluster_aspect": float(cluster_aspect),
                "cluster_theta": float(cluster_theta),
                "cluster_sigma_1": float(cluster_scale * root),
                "cluster_sigma_2": float(cluster_scale / root),
                "c1": 2.0 * cluster_scale * math.sqrt(parent_intensity),
            },
        )
