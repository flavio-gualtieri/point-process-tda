"""PersistenceImager / fit_imager: the properties the regime analysis leans on."""

from __future__ import annotations

import numpy as np
import pytest

from cloudforger.featurization.filtrations import alpha, dtm, rips
from cloudforger.simulation.sweep import DATA
from cloudforger.vectorization.persistence_images import PersistenceImager, Scaling, fit_imager

RNG = np.random.default_rng(0)


def _pairs(n, rng=RNG, lo=0.0, hi=0.5):
    b = rng.uniform(lo, hi, n)
    return np.column_stack([b, b + rng.uniform(lo, hi, n)])


def _interior_pairs(n, rng=RNG):
    """Pairs several sigma away from every edge, so no Gaussian mass falls outside the box."""
    return _pairs(n, rng, lo=0.5, hi=1.0)


def test_shapes_and_total_mass():
    pairs = _interior_pairs(20)
    flat = Scaling(coords="none", density=False)
    im2 = PersistenceImager((0.0, 2.0), (0.0, 2.0), resolution=16, scaling=flat)
    im1 = PersistenceImager((0.0, 2.0), None, resolution=16, scaling=flat)
    assert im2.transform(pairs, 100).shape == (16, 16)
    assert im1.transform(pairs, 100).shape == (16,)
    # Away from the edges the image carries the full weight (the sum of the persistences).
    total = (pairs[:, 1] - pairs[:, 0]).sum()
    assert im2.transform(pairs, 100).sum() == pytest.approx(total, rel=1e-3)
    assert im1.transform(pairs, 100).sum() == pytest.approx(total, rel=1e-3)


def test_mass_at_the_zero_persistence_edge_is_truncated():
    """Half of a point's Gaussian sits below persistence = 0 and is dropped -- but the linear
    weight makes those points nearly weightless anyway, so the loss stays small."""
    pairs = np.column_stack([np.full(5, 0.5), np.full(5, 0.5) + 1e-9])  # persistence ~ 0
    im = PersistenceImager((0.0, 2.0), (0.0, 1.0), resolution=16, scaling=Scaling("none", False))
    total = (pairs[:, 1] - pairs[:, 0]).sum()
    assert im.transform(pairs, 100).sum() == pytest.approx(total / 2, rel=1e-3)


def test_empty_diagram_is_zeros():
    im = PersistenceImager((0.0, 1.0), (0.0, 1.0), resolution=8)
    assert not im.transform(np.empty((0, 2)), 100).any()


def test_scaling_removes_the_n_dependence():
    """Two diagrams identical up to the 1/sqrt(n) spacing must give the same image."""
    pairs = _pairs(30)
    im = PersistenceImager((0.0, 6.0), (0.0, 6.0), resolution=16)
    a = im.transform(pairs / np.sqrt(100), 100)
    b = im.transform(pairs / np.sqrt(800), 800)
    assert a.sum() > 0
    assert np.allclose(a * 100, b * 800)                       # same shape, density divided by n
    off = PersistenceImager((0.0, 6.0), (0.0, 6.0), resolution=16, scaling=Scaling("none", False))
    assert not np.allclose(off.transform(pairs / np.sqrt(100), 100),
                           off.transform(pairs / np.sqrt(800), 800))


def test_fit_imager_covers_the_diagrams():
    diagrams = [_pairs(RNG.integers(10, 40)) / np.sqrt(200) for _ in range(50)]
    n = np.full(len(diagrams), 200)
    im = fit_imager(diagrams, n, birth_axis=True, resolution=16, coverage=1.0)
    scaled = [d * np.sqrt(200) for d in diagrams]
    assert im.pers_range[1] >= max((d[:, 1] - d[:, 0]).max() for d in scaled)
    assert im.birth_range[0] <= min(d[:, 0].min() for d in scaled)
    assert im.birth_range[1] >= max(d[:, 0].max() for d in scaled)


def test_fit_imager_rejects_a_degenerate_birth_axis():
    flat = [np.column_stack([np.zeros(5), np.linspace(0.1, 0.5, 5)]) for _ in range(5)]
    with pytest.raises(ValueError, match="degenerate"):
        fit_imager(flat, np.full(5, 100), birth_axis=True)
    assert fit_imager(flat, np.full(5, 100), birth_axis=False, resolution=8).shape == (8,)


@pytest.mark.skipif(not (DATA / "poisson" / "points.npz").exists(), reason="no simulated data")
def test_birth_axis_is_flat_for_rips_and_alpha_but_not_dtm():
    """Which filtration needs the 1-D H0 image is a property of the filtration, not of the data."""
    z = np.load(DATA / "poisson" / "points.npz")
    pts = z["points"][z["offsets"][0]:z["offsets"][1]]
    assert np.ptp(rips(pts)[0][:, 0]) == 0
    assert np.ptp(alpha(pts)[0][:, 0]) == 0
    assert np.ptp(dtm(pts, k=10)[0][:, 0]) > 0
