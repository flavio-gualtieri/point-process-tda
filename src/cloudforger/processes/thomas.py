# src/pointforge/processes/thomas.py
import numpy as np
from ..core.base import PointProcess
from ..core.region import Region, Box

class ThomasProcess(PointProcess):
    """Thomas cluster process (Neyman-Scott with Gaussian kernels)."""

    def __init__(
        self,
        parent_intensity: float,
        mean_offspring: float,
        cluster_scale: float,
    ):
        if parent_intensity <= 0:
            raise ValueError("parent_intensity must be positive")
        if mean_offspring <= 0:
            raise ValueError("mean_offspring must be positive")
        if cluster_scale <= 0:
            raise ValueError("cluster_scale must be positive")
        self.parent_intensity = parent_intensity
        self.mean_offspring = mean_offspring
        self.cluster_scale = cluster_scale

    @property
    def name(self) -> str:
        return "thomas"

    @property
    def params(self) -> dict:
        return {
            "parent_intensity": self.parent_intensity,
            "mean_offspring": self.mean_offspring,
            "cluster_scale": self.cluster_scale,
        }

    def _sample_points(
        self, n: int, region: Region, rng: np.random.Generator
    ) -> np.ndarray:
        # Edge correction: expand region by ~4 sigma in each direction
        # so offspring near boundaries have properly distributed parents
        # Step 1: generate parent points in the expanded region
        pad = 4 * self.cluster_scale
        expanded = region.expanded(pad)
        n_parents = int(rng.poisson(self.parent_intensity * expanded.volume))
        parents = expanded.sample_uniform(n_parents, rng)

        # Step 2: generate offspring counts per parent
        offspring_counts = rng.poisson(self.mean_offspring, size=n_parents)
        total = int(offspring_counts.sum())
        if total == 0:
            return np.empty((0, region.dimension))

        # Step 3: scatter offspring around each parent
        parent_idx = np.repeat(np.arange(n_parents), offspring_counts)
        offsets = rng.normal(
            scale=self.cluster_scale,
            size=(total, region.dimension),
        )
        offspring = parents[parent_idx] + offsets

        # Step 4: clip to the original region
        mask = region.contains(offspring)
        return offspring[mask]