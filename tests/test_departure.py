import numpy as np
import pytest

from cloudforger.classical.lfunction import RADII
from cloudforger.departure import config, simulate
from cloudforger.departure.fit import wls
from cloudforger.departure.tables import Tables, basis, knots, r_min

CFG = config.load()


def test_grid():
    g = CFG.grid()
    assert g[0] == CFG.n_low and g[-1] == CFG.n_high and np.all(np.diff(g) > 0)


def test_r_min_expected_pairs():
    n = np.array([20, 100, 2000])
    pairs = n * (n - 1) / 2 * np.pi * r_min(n, CFG.min_pairs) ** 2
    np.testing.assert_allclose(pairs, CFG.min_pairs)


def test_streams_are_addressed():
    a = simulate.grid_curve(CFG.root, 50, 7)
    assert np.array_equal(a, simulate.grid_curve(CFG.root, 50, 7))
    assert not np.array_equal(a, simulate.grid_curve(CFG.root, 50, 8))
    assert CFG.validation < simulate._REP_MAX <= CFG.n_low * simulate._REP_MAX


def test_wls_recovers_a_spline():
    n = CFG.grid().astype(float)
    t = knots(n[0], n[-1], 6)
    b = basis(n, t)
    coef = np.random.default_rng(0).normal(size=(b.shape[1], 3))
    np.testing.assert_allclose(wls(b, b @ coef, np.ones((len(n), 3))), coef, atol=1e-8)


@pytest.fixture(scope="module")
def tables():
    if not config.TABLES.exists():
        pytest.skip("no tables; run scripts/departure.py fit")
    return Tables()


def test_statistic_is_studentised_max(tables):
    n = np.array([35.0, 240.0, 1500.0])
    m0, s0, c95, mask = tables.moments(n)
    assert np.all(s0[mask] > 0)
    curves = m0 + 2.5 * c95[:, None] * s0
    np.testing.assert_allclose(tables.statistic(curves, n), 2.5)
    np.testing.assert_allclose(tables.delta_tilde(np.zeros((3, RADII.size)), n), 0.0)


def test_out_of_range_n_raises(tables):
    with pytest.raises(ValueError):
        tables.moments(CFG.n_low - 1)
