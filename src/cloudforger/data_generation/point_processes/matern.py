# src/cloudforger/data_generation/point_processes/matern.py

import numpy as np
from scipy.spatial import cKDTree
from ...core.base import PointProcess
from ...core.region import Region

class MaternHardCoreProcess(PointProcess):
    """Matérn Type II hard-core process via dependent thinning."""

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
        # Step 1: generate parent Poisson process
        expected = self.parent_intensity * region.volume
        n_parents = int(rng.poisson(expected))
        parents = region.sample_uniform(n_parents, rng)

        # Step 2: assign iid uniform marks
        marks = rng.uniform(size=n_parents)

        # Step 3: dependent thinning
        # Retain point i if no neighbor within hardcore_radius has smaller mark
        tree = cKDTree(parents)
        keep = np.ones(n_parents, dtype=bool)
        for i in range(n_parents):
            neighbors = tree.query_ball_point(parents[i], self.hardcore_radius)
            for j in neighbors:
                if j != i and marks[j] < marks[i]:
                    keep[i] = False
                    break

        return parents[keep]