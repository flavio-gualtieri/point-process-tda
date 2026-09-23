"""Ragged diagrams -> a fixed (capacity, 3) array a DataLoader can batch.

The persistence-image arm rasterizes a diagram into a calibrated box; the PersLay arm hands the
diagram itself to the network and lets it learn the vectorization. Everything that has to happen
before the network is therefore only what batching needs: the same sqrt(n) coordinate scaling the
imager applies, a standardization so the layer's learnable centres start inside the data, and
padding to a fixed length, since the diagrams of one batch have different cardinalities.

Columns are (birth, persistence, mass), the imager's plane rather than (birth, death), so both arms
read the same coordinates. `mass` is Scaling.mass's per-point factor (1/n under density scaling,
1 without it) and is exactly 0 on padded rows, so a sum over rows is masked and n-scaled in one
step -- the same normalization that makes images of different n comparable.

Capacity is fitted the way the imager's box is: it holds every point of `coverage` of the TRAINING
diagrams, capped at max_points so an H0 diagram of a dense pattern cannot set the memory bill for
all of them. A longer diagram keeps its most persistent points, which are the ones the image's
linear weight would also have made brightest.
"""

from __future__ import annotations

import math

import numpy as np

from ..persistence_images import Scaling


class DiagramPadder:
    """One (capacity, mean, std) -> padded arrays of a fixed shape.

    The standardization is a single (mean, std) per coordinate, pooled over the training diagrams'
    points -- never per diagram, which would erase exactly the differences in scale the task is
    about, the same reason the image z-score is pooled over train patterns and pixels.
    """

    def __init__(
        self,
        capacity: int,
        mean: np.ndarray,
        std: np.ndarray,
        scaling: Scaling = Scaling(),
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity!r}")
        self.capacity = int(capacity)
        self.mean = np.asarray(mean, dtype=np.float64).reshape(2)
        self.std = np.maximum(np.asarray(std, dtype=np.float64).reshape(2), 1e-12)
        self.scaling = scaling

    @property
    def shape(self) -> tuple[int, int]:
        return (self.capacity, 3)

    def transform(self, pairs: np.ndarray, n: int) -> np.ndarray:
        """One diagram as (capacity, 3): standardized (birth, persistence) and the point's mass."""
        out = np.zeros(self.shape, dtype=np.float32)
        pairs = np.asarray(pairs, dtype=float).reshape(-1, 2)
        if len(pairs) == 0:
            return out
        pairs = self.scaling.pairs(pairs, n)
        points = np.column_stack([pairs[:, 0], pairs[:, 1] - pairs[:, 0]])
        if len(points) > self.capacity:                      # keep the most persistent points
            keep = np.argpartition(-points[:, 1], self.capacity - 1)[: self.capacity]
            points = points[keep]
        out[: len(points), :2] = (points - self.mean) / self.std
        out[: len(points), 2] = self.scaling.mass(np.ones(len(points)), n)
        return out

    @property
    def params(self) -> dict:
        return {"capacity": self.capacity, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "coords": self.scaling.coords, "density": self.scaling.density}


def fit_padder(
    pairs: list[np.ndarray],
    n_points: np.ndarray,
    coverage: float = 0.99,
    max_points: int = 1024,
    scaling: Scaling = Scaling(),
) -> tuple[DiagramPadder, float]:
    """Padder fitted on the given (training) diagrams, with the fraction of them it truncates.

    Statistics are accumulated rather than pooled into one array: H0 carries n - 1 pairs per
    pattern, so the training diagrams of one channel hold tens of millions of points.
    """
    sizes, total, total_sq, count = [], np.zeros(2), np.zeros(2), 0
    for p, n in zip(pairs, n_points):
        p = np.asarray(p, dtype=float).reshape(-1, 2)
        sizes.append(len(p))
        if len(p) == 0:
            continue
        p = scaling.pairs(p, int(n))
        points = np.column_stack([p[:, 0], p[:, 1] - p[:, 0]])
        total += points.sum(axis=0)
        total_sq += (points ** 2).sum(axis=0)
        count += len(points)
    if count == 0:
        raise ValueError("fit_padder: every diagram is empty")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 0.0))
    capacity = min(max_points, max(1, math.ceil(np.percentile(sizes, 100.0 * coverage))))
    truncated = float(np.mean(np.asarray(sizes) > capacity))
    return DiagramPadder(capacity, mean, std, scaling), truncated
