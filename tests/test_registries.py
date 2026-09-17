# tests/test_registries.py
"""Smoke tests: the filtrations and every registry run end to end on a tiny synthetic example."""

from __future__ import annotations

import numpy as np

from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.featurization.filtrations import FILTRATIONS, tag
from cloudforger.vectorization.scalar_features import REGISTRY as FEATURE_REGISTRY


def _points(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((150, 2))


def _dtm_diagram(seed: int = 0) -> PersistenceDiagram:
    return PersistenceDiagram(FILTRATIONS["dtm"](_points(seed), k=5), "uniform", {})


def test_filtration_tags():
    assert [tag(s) for s in ({"name": "rips"}, {"name": "dtm", "k": 5}, {"name": "alpha"})] == [
        "rips", "dtm_k5", "alpha_diameter"]


def test_filtrations_end_to_end():
    for name, params in (("rips", {}), ("dtm", {"k": 5}), ("alpha", {"scale": "radius"})):
        diagram = FILTRATIONS[name](_points(), **params)
        assert set(diagram) == {0, 1}
        assert all(np.isfinite(pairs).all() and pairs.shape[1] == 2 for pairs in diagram.values())


def test_dtm_matches_gudhi():
    from gudhi.dtm_rips_complex import DTMRipsComplex

    points = _points()
    st = DTMRipsComplex(points=points, k=5, q=2).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    ours = FILTRATIONS["dtm"](points, k=5)
    for dim in (0, 1):
        ref = st.persistence_intervals_in_dimension(dim)
        ref = ref[np.isfinite(ref[:, 1])]
        order = lambda a: a[np.lexsort(a.T[::-1])]
        assert np.allclose(order(ref), order(ours[dim]))


def test_feature_registry_betti_and_entropy():
    assert {"betti_curve", "persistence_entropy"} <= set(FEATURE_REGISTRY.names())
    diagram = _dtm_diagram()

    betti = FEATURE_REGISTRY.build("betti_curve", homology_dims=(0, 1), grid_size=64)
    bc = betti.compute(diagram)
    assert bc.vector().shape == (128,)

    entropy = FEATURE_REGISTRY.build("persistence_entropy", homology_dims=(0, 1))
    ent = entropy.compute(diagram)
    assert set(ent.keys()) == {0, 1}


def test_calibrated_imager():
    from cloudforger.vectorization.persistence_images.calibrated import build_calibrated_imager

    diagrams = [_dtm_diagram(i) for i in range(3)]

    imager = build_calibrated_imager(diagrams, homology_dims=(0, 1), resolution=16, verbose=False)
    images = imager.transform(diagrams[0])
    assert images[0].shape == (16, 16)
    assert images[1].shape == (16, 16)
