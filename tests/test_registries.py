# tests/test_registries.py
"""Smoke tests: every registry can build its registered classes and run
them end to end on a tiny synthetic example -- adding a new process/
filtration/feature should be verifiable by extending these, not by writing
a new bespoke script."""

from __future__ import annotations

import numpy as np

from cloudforger.core.region import Box
from cloudforger.features import REGISTRY as FEATURE_REGISTRY
from cloudforger.filtration import REGISTRY as FILTRATION_REGISTRY
from cloudforger.processes import REGISTRY as PROCESS_REGISTRY


def _sample_cloud(process_name: str, **params):
    proc = PROCESS_REGISTRY.build(process_name, **params)
    region = Box(low=np.zeros(2), high=np.ones(2))
    return proc.sample(region, seed=0)


def test_process_registry_covers_expected_names():
    assert {"poisson", "matern", "thomas", "nested_thomas", "inhom_thomas", "neyman_scott"} <= set(
        PROCESS_REGISTRY.names()
    )


def test_thomas_process_builds_and_samples():
    cloud = _sample_cloud("thomas", parent_intensity=50.0, mean_offspring=8.0, cluster_scale=0.03, edge_buffer=0.1)
    assert cloud.n_points >= 0
    assert cloud.points.shape[1] == 2


def test_filtration_registry_and_path_tag():
    assert {"rips", "dtm"} <= set(FILTRATION_REGISTRY.names())
    dtm = FILTRATION_REGISTRY.build("dtm", k=5)
    assert dtm.path_tag() == "dtm_k5"
    rips = FILTRATION_REGISTRY.build("rips")
    assert rips.path_tag() == "rips"


def test_dtm_filtration_end_to_end():
    cloud = _sample_cloud("thomas", parent_intensity=50.0, mean_offspring=8.0, cluster_scale=0.03, edge_buffer=0.1)
    diagram = FILTRATION_REGISTRY.build("dtm", k=5).compute(cloud)
    assert set(diagram.dimensions()) <= {0, 1}


def test_feature_registry_betti_and_entropy():
    assert {"betti_curve", "persistence_entropy"} <= set(FEATURE_REGISTRY.names())
    cloud = _sample_cloud("thomas", parent_intensity=50.0, mean_offspring=8.0, cluster_scale=0.03, edge_buffer=0.1)
    diagram = FILTRATION_REGISTRY.build("dtm", k=5).compute(cloud)

    betti = FEATURE_REGISTRY.build("betti_curve", homology_dims=(0, 1), grid_size=64)
    bc = betti.compute(diagram)
    assert bc.vector().shape == (128,)

    entropy = FEATURE_REGISTRY.build("persistence_entropy", homology_dims=(0, 1))
    ent = entropy.compute(diagram)
    assert set(ent.keys()) == {0, 1}


def test_calibrated_imager():
    from cloudforger.vectorizers.calibrated import build_calibrated_imager

    clouds = [_sample_cloud("thomas", parent_intensity=50.0, mean_offspring=8.0, cluster_scale=0.03, edge_buffer=0.1) for _ in range(3)]
    dtm = FILTRATION_REGISTRY.build("dtm", k=5)
    diagrams = [dtm.compute(c) for c in clouds]

    imager = build_calibrated_imager(diagrams, homology_dims=(0, 1), resolution=16, verbose=False)
    images = imager.transform(diagrams[0])
    assert images[0].shape == (16, 16)
    assert images[1].shape == (16, 16)
