# src/cloudforger/processes/nested_thomas.py

from __future__ import annotations

import numpy as np

from ..core.region import Region
from .neyman_scott import NeymanScottProcess, gaussian_displacements, poisson_counts
from .thomas import ThomasProcess


class NestedThomasProcess(NeymanScottProcess):
    """Two-level ("clusters of clusters") Thomas process.

    Level 1 (meta): an ordinary ThomasProcess(meta_parent_intensity,
    meta_offspring, meta_cluster_scale) generates a set of coarse cluster
    centers over an expanded region.
    Level 2 (fine): each of those coarse centers is treated as a *parent*
    of this process -- i.e. it becomes the center of its own Poisson
    (mean_offspring)/Gaussian(cluster_scale) cluster, exactly like a flat
    ThomasProcess's own parents. The final point cloud is only the fine
    (level-2) offspring; the coarse centers themselves are never emitted.

    5 free parameters instead of ThomasProcess's 3: meta_parent_intensity,
    meta_offspring, meta_cluster_scale (coarse layer) plus mean_offspring,
    cluster_scale (fine layer, same meaning as in ThomasProcess). Setting
    meta_offspring=1 makes each coarse center generate exactly one fine
    center, which (up to the extra meta_cluster_scale jitter) reduces this
    back to a flat ThomasProcess at the coarse layer's own intensity.

    Implementation note: this composes two ThomasProcess-shaped stages by
    plugging the coarse process in as `parent_sampler` on the *fine*
    NeymanScottProcess (see neyman_scott.py) -- so nesting doesn't require
    any new sampling math, just reusing the coarse ThomasProcess's own
    `_sample_points` as the fine level's parent generator. Both levels
    share one rng (mirroring how InhomThomas drives its own internal
    ThomasProcess, see inhom_thomas.py's `_sample_with_covariates`), so a
    single seed on `.sample(...)` deterministically reproduces the whole
    two-level cloud.

    `edge_buffer` (fine level, default 4*cluster_scale) and
    `meta_edge_buffer` (coarse level, default 4*meta_cluster_scale) both
    apply via Region.expanded(), which just pads low/high additively --
    nesting the two expansions (this class expands by edge_buffer, then the
    coarse ThomasProcess expands that already-expanded region by its own
    meta_edge_buffer) covers both displacement scales automatically,
    without summing paddings by hand.
    """

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
            },
            parent_sampler=_parent_sampler,
        )
