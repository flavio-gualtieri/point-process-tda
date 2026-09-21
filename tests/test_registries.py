# tests/test_registries.py
"""Smoke tests: the filtrations run end to end on a tiny synthetic example."""

from __future__ import annotations

import numpy as np

from cloudforger.featurization.filtrations import FILTRATIONS, tag


def _points(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((150, 2))


def _dtm_pairs(seed: int = 0) -> dict[int, np.ndarray]:
    """Finite pairs per homology dimension of a DTM diagram."""
    diagram = FILTRATIONS["dtm"](_points(seed), k=5)
    return {dim: pairs[np.isfinite(pairs).all(axis=1)] for dim, pairs in diagram.items()}


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


def test_fit_imager_on_real_diagrams():
    from cloudforger.vectorization.persistence_images import fit_imager

    diagrams = [_dtm_pairs(i) for i in range(3)]
    pairs = [d[1] for d in diagrams]
    n = np.array([len(d[0]) + 1 for d in diagrams])       # H0 has one infinite bar, dropped above

    imager = fit_imager(pairs, n, birth_axis=True, resolution=16)
    assert imager.transform(pairs[0], n[0]).shape == (16, 16)
