import numpy as np

from cloudforger.classical.lfunction import RADII, isotropic_weights, k_function, l_minus_r, l_minus_r_from_excess


def test_radii():
    assert len(RADII) == 512 and RADII[0] > 0 and RADII[-1] == 0.25


def test_weights_match_brute_force_circle_fraction():
    rng = np.random.default_rng(0)
    centres, d = rng.random((300, 2)), rng.uniform(1e-3, 0.25, 300)
    theta = np.linspace(0, 2 * np.pi, 40000, endpoint=False)
    x = centres[:, [0]] + d[:, None] * np.cos(theta)
    y = centres[:, [1]] + d[:, None] * np.sin(theta)
    inside = ((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)).mean(axis=1)
    np.testing.assert_allclose(1 / isotropic_weights(centres, d), inside, atol=1e-3)


def test_two_interior_points():
    pts = np.array([[0.5, 0.5], [0.5, 0.6]])
    k = k_function(pts)
    assert np.all(k[RADII < 0.1 - 1e-12] == 0)
    np.testing.assert_allclose(k[RADII >= 0.1], 1.0)


def test_k_unbiased_under_binomial():
    rng = np.random.default_rng(1)
    k = np.mean([k_function(rng.random((200, 2))) for _ in range(400)], axis=0)
    sel = RADII >= 0.05
    np.testing.assert_allclose(k[sel] / (np.pi * RADII[sel] ** 2), 1.0, atol=0.03)


def test_l_from_excess():
    assert np.all(l_minus_r_from_excess(np.zeros_like(RADII)) == 0)
    e = np.pi * RADII**2 * np.random.default_rng(2).uniform(-0.9, 3.0, RADII.size)
    np.testing.assert_allclose(l_minus_r_from_excess(e), np.sqrt((np.pi * RADII**2 + e) / np.pi) - RADII, atol=1e-12)
    assert np.allclose(l_minus_r(np.array([[0.2, 0.2], [0.9, 0.9]])), -RADII)
