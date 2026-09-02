# src/cloudforger/data_generation/point_processes/nested_thomas.py

from __future__ import annotations

import math

import numpy as np

from ...core.region import Region
from .neyman_scott import NeymanScottProcess, poisson_counts
from .thomas import ThomasProcess
from .kernels import GaussianKernel


class NestedThomasProcess(NeymanScottProcess):

    def __init__(
        self,
        meta_parent_intensity: float,
        meta_offspring: float,
        meta_cluster_scale: float,
        mean_offspring: float,
        cluster_scale: float,
        edge_buffer: float | None = None,
        meta_edge_buffer: float | None = None,
    ):
        self.meta_process = ThomasProcess(
            parent_intensity=meta_parent_intensity,
            mean_offspring=meta_offspring,
            cluster_scale=meta_cluster_scale,
            edge_buffer=meta_edge_buffer,
        )

        def _parent_sampler(region: Region, rng: np.random.Generator) -> np.ndarray:
            return self.meta_process._sample_points(None, region, rng)

        super().__init__(
            parent_intensity=meta_parent_intensity,
            offspring_count_sampler=poisson_counts(mean_offspring),
            kernel=GaussianKernel(cluster_scale),
            edge_buffer=edge_buffer,
            process_name="nested_thomas",
            param_dict={
                "meta_offspring": meta_offspring,
                "meta_cluster_scale": meta_cluster_scale,
                "mean_offspring": mean_offspring,
                "c1": 2.0 * meta_cluster_scale * math.sqrt(meta_parent_intensity),
                "c2": 2.0 * cluster_scale * math.sqrt(meta_parent_intensity * meta_offspring),
            },
            parent_sampler=_parent_sampler,
        )
