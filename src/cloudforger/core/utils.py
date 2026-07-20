# src/cloudforger/core/utils.py
import numpy as np
from .cloud import PointCloud

def standardize_size(cloud: PointCloud, n: int, seed: int) -> PointCloud:
    """Subsample (or raise) to produce exactly n points."""
    if cloud.n_points < n:
        raise ValueError(f"cloud has {cloud.n_points} < {n} points; regenerate with higher intensity")
    if cloud.n_points == n:
        return cloud
    rng = np.random.default_rng(seed)
    idx = rng.choice(cloud.n_points, size=n, replace=False)
    return PointCloud(
        points=cloud.points[idx],
        generator_name=cloud.generator_name,
        generator_params=cloud.generator_params,
        seed=cloud.seed,
    )