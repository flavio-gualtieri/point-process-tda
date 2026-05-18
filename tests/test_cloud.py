import numpy as np
from cloudforger.core.cloud import PointCloud

def test_pointcloud_construction():
    pts = np.random.rand(100, 2)
    pc = PointCloud(points=pts, generator_name="test", seed=42)
    assert pc.n_points == 100
    assert pc.dimension == 2
    assert pc.seed == 42