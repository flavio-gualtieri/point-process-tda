# src/cloudforger/stats/pair_dist.py

from __future__ import annotations

from ..core.cloud import PointCloud
from .base import CloudStatistic
from ..core.region import Box

import numpy as np


class PairDistanceCDF(CloudStatistic):

    @property
    def name(self) -> str:
        return "pair_distance_cdf"
    
    @property
    def grid_range(self) -> tuple[float, float]:
        return (0.0, 1.0)
    
    def _sample_values(self, cloud: PointCloud, rng: np.random.Generator) -> np.ndarray:
        pts = cloud.points
        n = len(pts)

        i = rng.integers(0, n, size=self._n_samples)
        j = rng.integers(0, n, size=self._n_samples)
        collision = i == j

        while collision.any():
            j[collision] = rng.integers(0, n, size=collision.sum())
            collision = i == j

        diffs = pts[i] - pts[j]
        dists = np.sqrt(np.sum(diffs**2, axis=1))

        max_distance = self._region_diameter(cloud)
        return dists / max_distance
    
    @staticmethod
    def _region_diameter(cloud: PointCloud) -> float:
        """Diameter of the cloud's region — the largest possible distance
        between two points in it. For a Box, the diagonal length.

        This is a fixed property of the domain, identical for every cloud
        from the same region, so it does NOT leak per-cloud structure
        into the normalization.
        """
        region = cloud.region
        if region is None:
            raise ValueError(
                "PairDistanceCDF requires cloud.region to be set; "
                "regenerate the dataset with the updated PointProcess."
            )
        if isinstance(region, Box):
            span = region.high - region.low
            return float(np.sqrt(np.sum(span ** 2)))
        raise NotImplementedError(
            f"Region diameter not defined for {type(region).__name__}"
        )