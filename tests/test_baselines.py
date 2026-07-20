# tests/test_baselines.py
"""Smoke tests for the classical (mincontrast/palm) and vihrs baselines,
plus a regression test for a real seed-alignment bug caught during
development."""

from __future__ import annotations

import numpy as np


def _clustered_points(rng: np.random.Generator, n_parents: int = 8, n_offspring: int = 15, scale: float = 0.03) -> np.ndarray:
    parents = rng.uniform(0, 1, size=(n_parents, 2))
    pts = np.concatenate([p + rng.normal(scale=scale, size=(n_offspring, 2)) for p in parents])
    return np.clip(pts, 0, 1)


def test_mincontrast_fit():
    from cloudforger.baselines import mincontrast

    rng = np.random.default_rng(0)
    pts = _clustered_points(rng)
    result = mincontrast.fit_multistart(pts, n_starts=3, rng=rng)
    for key in ("parent_intensity", "cluster_scale", "mean_offspring"):
        assert np.isfinite(result[key])


def test_palm_fit():
    from cloudforger.baselines import palm

    rng = np.random.default_rng(1)
    pts = _clustered_points(rng)
    result = palm.fit_multistart(pts, n_starts=3, rng=rng)
    for key in ("parent_intensity", "cluster_scale", "mean_offspring"):
        assert np.isfinite(result[key])


def test_vihrs_network_and_normalization():
    import torch

    from cloudforger.baselines import vihrs

    r_grid = vihrs.default_r_grid()
    model = vihrs.VihrsCNN(seq_len=len(r_grid), n_targets=3)
    out = model(torch.randn(2, len(r_grid)), torch.randn(2))
    assert out.shape == (2, 3)

    norm = vihrs.fit_log_zscore(np.array([[50.0, 10.0, 0.02]] * 5))
    assert norm["mean"].shape == (3,)


def test_isotropic_l_minus_r_shape():
    from cloudforger.baselines import vihrs

    rng = np.random.default_rng(2)
    pts = _clustered_points(rng)
    r_grid = vihrs.default_r_grid(r_max=0.25, n_r=64)
    lr = vihrs._isotropic_l_minus_r(pts, [0, 0], [1, 1], r_grid)
    assert lr.shape == (64,)


def test_seed_join_correctness():
    """Regression test for a real bug caught during development: seed joins
    must be index-based, not same-set boolean masks -- different arrays can
    store the same seed set in different relative orders, and a boolean
    mask silently misaligns features with the wrong labels in that case."""
    from cloudforger.core.io import align_seeds, intersect_seeds

    a = np.array([5, 3, 8, 1, 9])
    b = np.array([1, 9, 5, 3])
    a_idx, b_idx = align_seeds(a, b)
    assert np.array_equal(a[a_idx], b[b_idx])

    c = np.array([9, 5, 1, 3, 2])
    idx_per_array, common = intersect_seeds([a, b, c])
    for arr, idx in zip([a, b, c], idx_per_array):
        assert np.array_equal(arr[idx], common)
