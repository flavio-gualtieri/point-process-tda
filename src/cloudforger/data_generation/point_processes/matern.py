# src/cloudforger/data_generation/point_processes/matern.py

import numpy as np
from scipy.spatial import cKDTree
from ...core.base import PointProcess
from ...core.region import Region

class MaternHardCoreProcess(PointProcess):
    """Matérn Type II hard-core process via dependent thinning.

    Primary points ~ Poisson(parent_intensity) with iid U(0,1) marks; a primary
    is deleted if another primary within hardcore_radius R has a smaller mark.
    Whether a point of W survives depends only on primaries in its R-disc, so
    the primaries are simulated on W expanded by R, thinned, then cropped to W:
    exact for the stationary process. (Thinning inside W only lets no primary
    outside W delete an edge point, inflating the edge-band intensity.)
    """

    def __init__(self, parent_intensity: float, hardcore_radius: float):
        if parent_intensity <= 0:
            raise ValueError("parent_intensity must be positive")
        if hardcore_radius <= 0:
            raise ValueError("hardcore_radius must be positive")
        self.parent_intensity = parent_intensity
        self.hardcore_radius = hardcore_radius

    @property
    def name(self) -> str:
        return "matern_ii"

    @property
    def params(self) -> dict:
        return {
            "parent_intensity": self.parent_intensity,
            "hardcore_radius": self.hardcore_radius,
        }

    def _sample_points(
        self, n: int, region: Region, rng: np.random.Generator
    ) -> np.ndarray:
        sim = region.expanded(self.hardcore_radius)
        n_primary = int(rng.poisson(self.parent_intensity * sim.volume))
        primary = sim.sample_uniform(n_primary, rng)
        marks = rng.uniform(size=n_primary)
        self.last_n_primary = n_primary  # per-draw diagnostic (DV3 manifest)

        # In every R-close pair the later (larger-mark) point is deleted,
        # whether or not the earlier one itself survives: that is type II.
        pairs = cKDTree(primary).query_pairs(self.hardcore_radius, output_type="ndarray")
        deleted = np.zeros(n_primary, dtype=bool)
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            deleted[np.where(marks[i] > marks[j], i, j)] = True

        keep = ~deleted & region.contains(primary)
        return primary[keep]
