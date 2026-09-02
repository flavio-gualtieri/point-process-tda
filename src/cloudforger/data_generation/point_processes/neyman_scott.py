# src/cloudforger/data_generation/point_processes/neyman_scott.py

from __future__ import annotations

from typing import Any, Callable
import numpy as np

from .kernels import Kernel
from ...core.base import PointProcess
from ...core.region import Region


CountSampler = Callable[[int, np.random.Generator], np.ndarray]
ParentSampler = Callable[[Region, np.random.Generator], np.ndarray]


def poisson_counts(mean: float) -> CountSampler:
    if mean <= 0:
        raise ValueError("mean must be positive")

    def sampler(n_parents: int, rng: np.random.Generator) -> np.ndarray:
        return rng.poisson(mean, size=n_parents)

    return sampler


class NeymanScottProcess(PointProcess):
    """
    General Neyman–Scott cluster process.

    Algorithm:
    1. Sample parent points in an expanded region -- uniform-Poisson by
       default, or via `parent_sampler` if given (e.g. another point process,
       for a nested/hierarchical cluster process -- see NestedThomasProcess).
    2. Sample offspring counts per parent.
    3. Sample offspring displacements from a kernel.
    4. Keep only offspring inside the original region.
    """

    def __init__(
        self,
        parent_intensity: float,
        kernel: Kernel,
        offspring_count_sampler: CountSampler,
        edge_buffer: float | None = None,
        buffer_eps: float = 1e-4,
        process_name: str = "neyman_scott",
        param_dict: dict[str, Any] | None = None,
        parent_sampler: ParentSampler | None = None,
    ):
        if parent_intensity <= 0:
            raise ValueError("parent_intensity must be positive")
        if edge_buffer is None:
            edge_buffer = kernel.support_radius(buffer_eps)
        if not (np.isfinite(edge_buffer) and edge_buffer >= 0):
            raise ValueError("edge_buffer must be finite and non-negative")

        self.parent_intensity = parent_intensity
        self.kernel = kernel
        self.offspring_count_sampler = offspring_count_sampler
        self.edge_buffer = edge_buffer
        self.process_name = process_name
        self.param_dict = param_dict or {}
        self.parent_sampler = parent_sampler

    @property
    def name(self) -> str:
        return self.process_name

    @property
    def params(self) -> dict[str, Any]:
        return {
            "parent_intensity": self.parent_intensity,
            "edge_buffer": self.edge_buffer,
            **self.kernel.params,
            **self.param_dict,
        }

    def _sample_points(
        self,
        n: int | None,
        region: Region,
        rng: np.random.Generator,
    ) -> np.ndarray:
        expanded = region.expanded(self.edge_buffer)

        if self.parent_sampler is not None:
            parents = np.asarray(self.parent_sampler(expanded, rng), dtype=float)
            if parents.ndim != 2 or parents.shape[1] != region.dimension:
                raise ValueError(
                    f"parent_sampler must return shape (n_parents, {region.dimension}), got {parents.shape}"
                )
        else:
            n_parents = int(rng.poisson(self.parent_intensity * expanded.volume))
            if n_parents == 0:
                return np.empty((0, region.dimension))
            parents = expanded.sample_uniform(n_parents, rng)

        n_parents = len(parents)
        if n_parents == 0:
            return np.empty((0, region.dimension))

        offspring_counts = np.asarray(
            self.offspring_count_sampler(n_parents, rng),
            dtype=int,
        )

        if offspring_counts.shape != (n_parents,):
            raise ValueError("offspring_count_sampler must return shape (n_parents,)")
        if np.any(offspring_counts < 0):
            raise ValueError("offspring counts must be non-negative")

        total = int(offspring_counts.sum())
        if total == 0:
            return np.empty((0, region.dimension))

        parent_idx = np.repeat(np.arange(n_parents), offspring_counts)

        offsets = np.asarray(
            self.kernel.sample(total, region.dimension, rng),
            dtype=float,
        )

        if offsets.shape != (total, region.dimension):
            raise ValueError("kernel.sample must return shape (total, dimension)")

        offspring = parents[parent_idx] + offsets

        return offspring[region.contains(offspring)]