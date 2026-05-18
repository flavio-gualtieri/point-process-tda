# tests/test_poisson.py
import numpy as np
from cloudforger.core.region import Box
from cloudforger.processes.poisson import PoissonProcess

def test_reproducibility():
    proc = PoissonProcess(intensity=None)
    region = Box(low=[0, 0], high=[1, 1])
    cloud_a = proc.sample(n=100, region=region, seed=42)
    cloud_b = proc.sample(n=100, region=region, seed=42)
    np.testing.assert_array_equal(cloud_a.points, cloud_b.points)

def test_count_in_conditional_mode():
    proc = PoissonProcess()
    region = Box(low=[0, 0], high=[1, 1])
    cloud = proc.sample(n=100, region=region, seed=0)
    assert cloud.n_points == 100

def test_intensity_metadata():
    proc = PoissonProcess(intensity=50.0)
    region = Box(low=[0, 0], high=[1, 1])
    cloud = proc.sample(n=0, region=region, seed=0)
    assert cloud.generator_params["intensity"] == 50.0
    assert cloud.generator_name == "poisson"