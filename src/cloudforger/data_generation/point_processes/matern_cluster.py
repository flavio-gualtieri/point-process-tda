# src/cloudforger/data_generation/point_processes/matern_cluster.py
"""Matern *cluster* process: a Neyman-Scott process with offspring uniform on
a disk of radius `cluster_radius` around each parent (BallKernel).

Not to be confused with MaternHardCoreProcess (matern.py, registered as
"matern") -- that is a repulsive Type-II thinning process, a different model
that happens to share the Matern name; this one shares the Neyman-Scott
mechanism with ThomasProcess and only swaps the offspring kernel. Registered
as "matern_cluster" so the two are never ambiguous in a config's
`process.name` or in a results path.

The report's overlap index nu = r*sqrt(kappa) is defined per-family via r,
the RMS *per-axis* offspring displacement -- for ThomasProcess r = cluster_scale
(the Gaussian's own std). For a uniform-on-disk(R) offspring, E[X^2] = R^2/4
by symmetry (E[rho^2] = R^2/2 for the radial coordinate, split evenly
between the two axes), so r = R/2 here. `c1` mirrors ThomasProcess's own
diagnostic field: with r = cluster_radius/2, `c1 = 2*r*sqrt(parent_intensity)
= cluster_radius*sqrt(parent_intensity)`, recovering a design's raw sampled
`c` (see configs/runs/matern_cluster/*.yaml) as a self-check.
"""

import math

from .neyman_scott import NeymanScottProcess, poisson_counts
from .kernels import BallKernel


class MaternClusterProcess(NeymanScottProcess):

    def __init__(
        self,
        parent_intensity: float,
        mean_offspring: float,
        cluster_radius: float,
        edge_buffer: float | None = None,
    ):
        super().__init__(
            parent_intensity=parent_intensity,
            kernel=BallKernel(cluster_radius),
            offspring_count_sampler=poisson_counts(mean_offspring),
            edge_buffer=edge_buffer,
            process_name="matern_cluster",
            param_dict={
                "mean_offspring": mean_offspring,
                "c1": cluster_radius * math.sqrt(parent_intensity),
            },
        )
