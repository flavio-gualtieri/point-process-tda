import numpy as np
import pytest

from cloudforger.classical.lfunction import RADII
from cloudforger.departure import config, simulate
from cloudforger.departure.fit import wls
from cloudforger.departure.tables import DEFAULT, REDUCTIONS, SCALAR, Tables, basis, knots, r_min

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
    m0, s0 = tables.moments(n)
    c95 = tables.critical(n, "sup")
    assert np.all(s0[tables.window(n, "sup")] > 0)
    curves = m0 + 2.5 * c95[:, None] * s0
    np.testing.assert_allclose(tables.statistic(curves, n, "sup"), 2.5)


def _constant(tables, n, name, target):
    """A flat curve whose raw extremum makes `name`'s statistic exactly `target`."""
    m, s = tables.scalar_moments(name, n)
    v = (m + target * tables.critical(n, name) * s)
    return np.repeat((-v if name == "lo" else v)[:, None], RADII.size, axis=1)


def test_every_reduction_is_calibrated(tables):
    """Each reduction scales like its own critical value, so 1 is its own rejection boundary."""
    n = np.array([35.0, 240.0, 1500.0])
    m0, s0 = tables.moments(n)
    for name in REDUCTIONS:
        assert name in tables.coef_c, f"{name} is defined but not calibrated -- refit the tables"
    flat = m0 + 2.5 * tables.critical(n, "sup")[:, None] * s0
    np.testing.assert_allclose(tables.statistic(flat, n, "sup"), 2.5)
    for name in SCALAR:
        np.testing.assert_allclose(tables.statistic(_constant(tables, n, name, 2.5), n, name), 2.5)


def test_sup_delta_tilde_vanishes_at_csr(tables):
    """The pointwise reference for an exact model curve is 0, so a flat zero curve scores 0."""
    n = np.array([35.0, 240.0, 1500.0])
    np.testing.assert_allclose(tables.delta_tilde(np.zeros((3, RADII.size)), n, "sup"), 0.0)


def test_scalar_delta_tilde_is_negative_at_csr(tables):
    """m_lo/m_hi are the free dip, not a bias: a CSR model curve sits BELOW the noise floor, and
    stays there as a finite ordered number rather than clamping to 0."""
    n = np.array([35.0, 240.0, 1500.0])
    zero = np.zeros((3, RADII.size))
    for name in (*SCALAR, DEFAULT):
        d = tables.delta_tilde(zero, n, name)
        assert np.all(d < 0) and np.all(np.isfinite(d)), (name, d)


def test_scalar_moments_scale_like_one_over_n(tables):
    """1/n is the leading behaviour, which is why the splines are fitted to n*m and n*s. n*m is
    not constant -- n*m_hi drifts by ~70% across the grid -- only far flatter than m itself."""
    n = np.geomspace(25.0, 1900.0, 12)
    spread = lambda v: np.ptp(v) / np.mean(v)
    for name in SCALAR:
        m, s = tables.scalar_moments(name, n)
        assert np.all(m > 0) and np.all(s > 0)          # both arms are oriented positive
        assert spread(n * m) < 0.2 * spread(m)      # n*s_lo still spans 0.30-0.45 across the grid
        assert spread(n * s) < 0.2 * spread(s)


def test_ext_is_the_joint_max_of_its_arms(tables):
    """`ext`'s own critical value is the joint scale k, so ext = max(arms) / k exactly."""
    n = np.array([35.0, 240.0, 1500.0])
    curves = _constant(tables, n, "hi", 3.0)
    arms = np.maximum(tables.statistic(curves, n, "lo"), tables.statistic(curves, n, "hi"))
    np.testing.assert_allclose(tables.statistic(curves, n, "ext"),
                               arms / tables.critical(n, "ext"))


def test_lo_reads_a_hard_core_radius(tables):
    """Below a hard core no point has a neighbour, so phi is pinned to -r and the minimum is -R."""
    n = np.full(3, 300.0)
    R = np.array([0.004, 0.010, 0.020])
    curves = np.where(RADII[None, :] < R[:, None], -RADII[None, :], 0.0)
    np.testing.assert_allclose(-curves.min(axis=1), RADII[np.searchsorted(RADII, R) - 1])
    d = tables.delta_tilde(curves, n, "lo")
    assert np.all(np.diff(d) > 0)                        # deeper core => strictly larger label


def test_only_the_sup_is_windowed(tables):
    n = np.array([35.0, 240.0, 1500.0])
    assert tables.window(n, "sup").all(axis=1).sum() == 0     # every n drops some low-pair radii
    for name in REDUCTIONS:
        assert REDUCTIONS[name].windowed == (name == "sup")
        if name != "sup":
            assert tables.window(n, name).all()


def test_unknown_reduction_raises(tables):
    with pytest.raises(ValueError):
        tables.statistic(np.zeros((1, RADII.size)), 240.0, "nope")


def test_out_of_range_n_raises(tables):
    with pytest.raises(ValueError):
        tables.moments(CFG.n_low - 1)
