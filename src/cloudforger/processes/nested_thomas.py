# src/cloudforger/processes/nested_thomas.py

from __future__ import annotations

import math

import numpy as np

from ..core.region import Region
from .neyman_scott import NeymanScottProcess, gaussian_displacements, poisson_counts
from .thomas import ThomasProcess


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
        if edge_buffer is None:
            edge_buffer = 4.0 * cluster_scale

        # Raises ValueError up front (via NeymanScottProcess.__init__) if
        # meta_parent_intensity/meta_cluster_scale are non-positive -- no
        # need to re-validate those here.
        self.meta_process = ThomasProcess(
            parent_intensity=meta_parent_intensity,
            mean_offspring=meta_offspring,
            cluster_scale=meta_cluster_scale,
            edge_buffer=meta_edge_buffer,
        )

        def _parent_sampler(region: Region, rng: np.random.Generator) -> np.ndarray:
            return self.meta_process._sample_points(None, region, rng)

        super().__init__(
            # NOT the density of this process's own immediate (fine)
            # cluster centers -- that's an emergent quantity, roughly
            # meta_parent_intensity * meta_offspring. This is the intensity
            # of the one genuinely-Poisson layer at the root of the whole
            # two-level hierarchy (self.meta_process's own parents),
            # exposed here under the same "parent_intensity" name
            # NeymanScottProcess always uses for that root layer.
            parent_intensity=meta_parent_intensity,
            offspring_count_sampler=poisson_counts(mean_offspring),
            displacement_sampler=gaussian_displacements(cluster_scale),
            edge_buffer=edge_buffer,
            process_name="nested_thomas",
            param_dict={
                "meta_offspring": meta_offspring,
                "meta_cluster_scale": meta_cluster_scale,
                "mean_offspring": mean_offspring,
                "cluster_scale": cluster_scale,
                "c1": 2.0 * meta_cluster_scale * math.sqrt(meta_parent_intensity),
                "c2": 2.0 * cluster_scale * math.sqrt(meta_parent_intensity * meta_offspring),
            },
            parent_sampler=_parent_sampler,
        )
