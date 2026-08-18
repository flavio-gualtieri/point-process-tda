# tests/test_landscape_silhouette.py
"""Unit tests for the landscape/silhouette build (see the build spec this
implements: tent primitive, K/grid calibration, persistence_statistics,
pad_factor alias, seed determinism). No CNN/training here -- see
tests/test_pipeline_e2e.py's test_vec_multik_reference_pipeline for the
full generate -> featurize -> train regression, including the
vectorization=persistence_image, encoder_path=native parity check against
method: pi_multik."""

from __future__ import annotations

import numpy as np
import pytest

from cloudforger.calibration import axis_bounds_1d
from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.vectorization.landscapes.tent import (
    landscape_from_tents,
    silhouette_from_tents,
    tent,
)
from cloudforger.vectorization.landscapes.calibrated import (
    build_calibrated_landscape,
    build_calibrated_silhouette,
    choose_K,
)
from cloudforger.vectorization.scalar_features.persistence_statistics import (
    PersistenceStatistics,
    STAT_NAMES,
)


def _diagram(pairs: np.ndarray, dim: int = 0) -> PersistenceDiagram:
    return PersistenceDiagram(diagrams={dim: pairs}, generator_name="test", generator_params={})


def _random_diagrams(n: int, seed: int = 0, dim: int = 0) -> list[PersistenceDiagram]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        m = rng.integers(3, 9)
        b = rng.uniform(0.1, 1.0, m)
        d = b + rng.uniform(0.1, 1.0, m)
        out.append(_diagram(np.stack([b, d], axis=1), dim=dim))
    return out


# ---------------------------------------------------------------------------
# Tent / landscape / silhouette primitives
# ---------------------------------------------------------------------------

def test_tent_peaks_at_midpoint_with_height_half_persistence():
    b, d = 0.0, 2.0
    grid = np.linspace(0.0, 2.0, 401)  # includes the exact midpoint
    tents = tent(np.array([[b, d]]), grid)
    assert np.isclose(tents[0].max(), (d - b) / 2.0)
    assert np.isclose(grid[np.argmax(tents[0])], (b + d) / 2.0)


def test_landscape_1_is_pointwise_max():
    pairs = np.array([[0.0, 2.0], [0.5, 1.5], [0.2, 1.0]])
    grid = np.linspace(0.0, 2.0, 33)
    tents = tent(pairs, grid)
    landscape = landscape_from_tents(tents, K=3)
    assert np.allclose(landscape[0], tents.max(axis=0))


def test_landscape_layers_are_nonincreasing():
    pairs = np.array([[0.0, 2.0], [0.5, 1.5], [0.2, 1.0], [0.1, 0.6]])
    grid = np.linspace(0.0, 2.0, 33)
    tents = tent(pairs, grid)
    landscape = landscape_from_tents(tents, K=4)
    for k in range(landscape.shape[0] - 1):
        assert np.all(landscape[k] + 1e-12 >= landscape[k + 1])


def test_landscape_missing_layers_are_zero_rows():
    pairs = np.array([[0.0, 2.0], [0.5, 1.5]])  # only 2 points
    grid = np.linspace(0.0, 2.0, 17)
    tents = tent(pairs, grid)
    landscape = landscape_from_tents(tents, K=5)
    assert np.allclose(landscape[2:], 0.0)

    # Empty diagram: every layer is a zero row.
    empty_tents = tent(np.empty((0, 2)), grid)
    assert np.allclose(landscape_from_tents(empty_tents, K=5), 0.0)


def test_silhouette_p0_is_unweighted_mean():
    pairs = np.array([[0.0, 2.0], [0.5, 1.5], [0.2, 1.0]])
    grid = np.linspace(0.0, 2.0, 33)
    tents = tent(pairs, grid)
    normalized, _ = silhouette_from_tents(tents, pairs, p=0.0)
    assert np.allclose(normalized, tents.mean(axis=0))


def test_silhouette_p_inf_is_longest_tent():
    pairs = np.array([[0.0, 2.0], [0.5, 1.5], [0.2, 1.0]])  # persistences 2.0, 1.0, 0.8
    grid = np.linspace(0.0, 2.0, 33)
    tents = tent(pairs, grid)
    normalized, _ = silhouette_from_tents(tents, pairs, p=np.inf)
    assert np.allclose(normalized, tents[0])  # pair 0 has the longest persistence


def test_silhouette_unnormalized_is_the_numerator():
    pairs = np.array([[0.0, 2.0], [0.5, 1.5], [0.2, 1.0]])
    grid = np.linspace(0.0, 2.0, 33)
    tents = tent(pairs, grid)
    p = 2.0
    normalized, unnormalized = silhouette_from_tents(tents, pairs, p=p)
    weights = (pairs[:, 1] - pairs[:, 0]) ** p
    expected_numerator = (weights[:, None] * tents).sum(axis=0)
    assert np.allclose(unnormalized, expected_numerator)
    assert np.allclose(normalized, expected_numerator / weights.sum())


def test_silhouette_missing_features_is_zero_row():
    grid = np.linspace(0.0, 2.0, 17)
    empty_tents = tent(np.empty((0, 2)), grid)
    normalized, unnormalized = silhouette_from_tents(empty_tents, np.empty((0, 2)), p=1.0)
    assert np.allclose(normalized, 0.0)
    assert np.allclose(unnormalized, 0.0)


# ---------------------------------------------------------------------------
# Calibration: 1-D grid, K-by-coverage, pad_factor alias, seed determinism
# ---------------------------------------------------------------------------

def test_axis_bounds_1d_fits_nonzero_t_min_for_dtm_h0():
    # DTM H0 births are nonzero -- t_min must be fit, not assumed 0.
    diagrams = _random_diagrams(50, seed=1)
    t_min, T = axis_bounds_1d(diagrams, homology_dim=0)
    assert T > t_min
    assert t_min > 0.0


def test_axis_bounds_1d_pad_factor_alias():
    diagrams = _random_diagrams(50, seed=2)
    a = axis_bounds_1d(diagrams, homology_dim=0, pad_factor=1.2)
    b = axis_bounds_1d(diagrams, homology_dim=0, pad=1.2)
    assert a == b
    with pytest.raises(ValueError):
        axis_bounds_1d(diagrams, homology_dim=0, pad_factor=1.2, pad=1.3)


def test_choose_K_capped_and_logged():
    diagrams = _random_diagrams(50, seed=3)
    t_min, T = axis_bounds_1d(diagrams, homology_dim=0)
    grid = np.linspace(t_min, T, 64)
    K, sup_norms = choose_K(diagrams, 0, grid, coverage=0.99, cap=16)
    assert 1 <= K <= 16
    assert sup_norms.shape == (len(diagrams), 17)

    # An explicit K override is honored, but choose_K's own answer is still
    # computable/loggable independently of what the run actually used.
    landscape = build_calibrated_landscape(diagrams, homology_dims=(0,), G=32, K=8, verbose=False)
    assert landscape.params[0]["K"] == 8


def test_choose_K_reconciled_across_dims_when_K_is_none():
    """Regression test: H0 (many points) and H1 (few points) routinely need
    different K under an independent per-dim coverage rule -- but
    build_landscape_tensor stacks every dim's (K, G) landscape into one
    (C, K, G) raster's channel axis, which requires a single shared K. This
    used to crash np.stack the moment H0/H1 diverged (reproduced with this
    exact H0-heavy/H1-sparse shape); build_calibrated_landscape now
    reconciles K=None to max(per-dim chosen K) across every dim."""
    rng = np.random.default_rng(42)
    diagrams = []
    for _ in range(80):
        n0 = rng.integers(15, 30)  # H0: many features -> larger chosen K
        b0 = rng.uniform(0.1, 1.0, n0)
        d0 = b0 + rng.uniform(0.1, 1.0, n0)
        n1 = rng.integers(1, 4)  # H1: few features -> smaller chosen K
        b1 = rng.uniform(0.1, 1.0, n1)
        d1 = b1 + rng.uniform(0.05, 0.3, n1)
        diagrams.append(PersistenceDiagram(
            diagrams={0: np.stack([b0, d0], axis=1), 1: np.stack([b1, d1], axis=1)},
            generator_name="x", generator_params={},
        ))

    landscape = build_calibrated_landscape(diagrams, homology_dims=(0, 1), G=32, K=None, verbose=False)
    assert landscape.params[0]["K"] == landscape.params[1]["K"]

    out = landscape.transform(diagrams[0])
    assert out[0].shape == out[1].shape
    stacked = np.stack([out[0], out[1]])  # what build_landscape_tensor does -- must not raise
    assert stacked.shape[0] == 2


def test_calibration_is_seed_deterministic():
    diagrams = _random_diagrams(50, seed=4)
    a = build_calibrated_landscape(diagrams, homology_dims=(0,), G=32, verbose=False)
    b = build_calibrated_landscape(diagrams, homology_dims=(0,), G=32, verbose=False)
    assert a.params == b.params

    sa = build_calibrated_silhouette(diagrams, homology_dims=(0,), G=32, p=1.0, verbose=False)
    sb = build_calibrated_silhouette(diagrams, homology_dims=(0,), G=32, p=1.0, verbose=False)
    assert sa.params == sb.params


def test_calibration_leakage_train_only():
    """Fitting on two disjoint diagram samples must give different
    calibrated bounds -- the same leakage-fix regression shape as
    pi_multik.py's/betti_multik.py's own tests (see test_pipeline_e2e.py)."""
    diagrams_a = _random_diagrams(50, seed=5)
    diagrams_b = _random_diagrams(50, seed=6)
    a = build_calibrated_landscape(diagrams_a, homology_dims=(0,), G=32, verbose=False)
    b = build_calibrated_landscape(diagrams_b, homology_dims=(0,), G=32, verbose=False)
    assert a.params != b.params


def test_landscape_transform_shape_and_channel_asserts():
    diagrams = _random_diagrams(50, seed=7)
    landscape = build_calibrated_landscape(diagrams, homology_dims=(0,), G=32, K=6, verbose=False)
    out = landscape.transform(diagrams[0])
    assert out[0].shape == (6, 32)


# ---------------------------------------------------------------------------
# persistence_statistics: the floor control
# ---------------------------------------------------------------------------

def test_persistence_statistics_missing_features_is_zero_row():
    feature = PersistenceStatistics(homology_dims=(0, 1))
    empty_diagram = _diagram(np.empty((0, 2)), dim=0)  # dim 1 not present at all
    out = feature.compute(empty_diagram)
    assert np.allclose(out[0], 0.0)
    assert np.allclose(out[1], 0.0)
    assert out[0].shape == (len(STAT_NAMES),)


def test_persistence_statistics_nonempty():
    feature = PersistenceStatistics(homology_dims=(0,))
    pairs = np.array([[0.0, 2.0], [0.5, 1.5], [0.2, 1.0]])
    out = feature.compute(_diagram(pairs, dim=0))
    stats = out[0]
    assert stats[STAT_NAMES.index("count")] == 3.0
    assert np.isclose(stats[STAT_NAMES.index("total_persistence")], (pairs[:, 1] - pairs[:, 0]).sum())
