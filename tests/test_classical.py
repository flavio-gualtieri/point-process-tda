"""Stage 2 of the classical arm: point pattern -> L, F, G, J curves -> curves.npz.

F and G are checked against their closed form under CSR, F(r) = G(r) = 1 - exp(-lambda pi r^2),
which is the only place these estimators have an exact answer to be wrong about. J is checked
against its neutral value of 1 there for the same reason.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cloudforger.classical import curves as C
from cloudforger.classical import fgj, functions
from cloudforger.classical.lfunction import RADII

FIXED = {"name": "fixed"}
SQRTN = {"name": "sqrtn", "u_max": 2.0}


def _poisson(rng, lam=300):
    return rng.random((rng.poisson(lam), 2))


# ---------------------------------------------------------------- estimators

def test_f_and_g_match_csr_closed_form():
    lam, rng = 300, np.random.default_rng(0)
    r = RADII[RADII <= 0.12]                      # border correction keeps most of the window here
    theory = 1 - np.exp(-lam * np.pi * r**2)
    f = np.mean([fgj.f_function(_poisson(rng, lam), r) for _ in range(300)], axis=0)
    g = np.mean([fgj.g_function(_poisson(rng, lam), r) for _ in range(300)], axis=0)
    np.testing.assert_allclose(f, theory, atol=0.01)
    np.testing.assert_allclose(g, theory, atol=0.01)


def test_j_is_one_under_csr():
    rng = np.random.default_rng(1)
    r = RADII[RADII <= 0.12]
    j = np.mean([fgj.j_function(fgj.f_function(p, r), fgj.g_function(p, r))
                 for p in (_poisson(rng) for _ in range(300))], axis=0)
    np.testing.assert_allclose(j, 1.0, atol=0.1)


def test_j_direction_matches_clustering_and_inhibition():
    """J < 1 for clustering, > 1 for inhibition -- the property J is in the feature set for."""
    rng = np.random.default_rng(2)
    r = RADII[RADII <= 0.08]

    def mean_j(patterns):
        return np.mean([fgj.j_function(fgj.f_function(p, r), fgj.g_function(p, r)).mean()
                        for p in patterns], axis=0)

    clustered = [np.clip(np.concatenate([c + rng.normal(scale=0.02, size=(25, 2))
                                         for c in rng.random((12, 2))]), 0, 1) for _ in range(40)]
    grid = np.linspace(0.02, 0.98, 17)
    inhibited = [np.stack(np.meshgrid(grid, grid, indexing="ij"), -1).reshape(-1, 2)
                 + rng.normal(scale=0.004, size=(289, 2)) for _ in range(40)]
    assert mean_j(clustered) < 1.0 < mean_j(inhibited)


def test_cdfs_are_monotone_and_bounded():
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = _poisson(rng)
        f, g = fgj.f_function(p, RADII), fgj.g_function(p, RADII)
        j = fgj.j_function(f, g)
        for c in (f, g):
            assert np.all(np.diff(c) >= 0) and c.min() >= 0.0 and c.max() <= 1.0
        assert np.isfinite(j).all() and j.min() >= 0.0 and j.max() <= fgj.J_CLIP


@pytest.mark.parametrize("n", [0, 1])
def test_degenerate_patterns_give_zero_curves(n):
    p = np.full((n, 2), 0.5)
    assert fgj.f_function(p, RADII).shape == RADII.shape
    assert not fgj.g_function(p, RADII).any()       # no neighbour distance exists


def test_test_grid_is_interior_and_cached():
    mesh, edge = functions.fgj.test_grid(8)
    assert mesh.shape == (64, 2) and edge.min() > 0          # no location sits on the boundary
    assert functions.fgj.test_grid(8)[0] is mesh             # lru_cache: identical object


# ------------------------------------------------------------ grid conventions

def test_fixed_grid_is_radii_for_every_pattern():
    assert functions.radii(FIXED, 50) is RADII and functions.radii(FIXED, 900) is RADII
    np.testing.assert_array_equal(functions.axis(FIXED), RADII)


def test_sqrtn_grid_scales_as_one_over_root_n():
    r100, r400 = functions.radii(SQRTN, 100), functions.radii(SQRTN, 400)
    np.testing.assert_allclose(r100 / r400, 2.0)             # sqrt(400/100)
    assert r100[-1] == pytest.approx(2.0 / 10.0)             # u_max / sqrt(n)
    np.testing.assert_allclose(functions.axis(SQRTN), r100 * 10.0)   # the axis is u = r sqrt(n)


def test_tags():
    assert functions.tag(FIXED) == "fixed" and functions.tag(SQRTN) == "sqrtn_u2"


def test_curves_returns_every_function_at_one_length():
    out = functions.curves(_poisson(np.random.default_rng(4)), FIXED)
    assert set(out) == set(functions.NAMES)
    assert all(c.shape == (functions.GRID_SIZE,) for c in out.values())


def test_sqrtn_stabilises_where_g_saturates():
    """The reason `sqrtn` exists: on a fixed axis the saturation point moves with n, in u it does not."""
    rng = np.random.default_rng(5)

    def u99(spec, lam):
        p = _poisson(rng, lam)
        g = functions.curves(p, spec)["G"]
        return functions.axis(spec)[np.argmax(g >= 0.99)] if (g >= 0.99).any() else np.nan

    sparse = np.median([u99(FIXED, 80) for _ in range(15)])
    dense = np.median([u99(FIXED, 800) for _ in range(15)])
    assert sparse / dense > 2.0                       # fixed axis: saturation moves a lot with n

    sparse = np.median([u99(SQRTN, 80) for _ in range(15)])
    dense = np.median([u99(SQRTN, 800) for _ in range(15)])
    assert 0.8 < sparse / dense < 1.25                # u axis: near n-invariant


# -------------------------------------------------------------------- the sweep

@pytest.fixture
def fake_simulation(tmp_path, monkeypatch):
    """One tiny family on disk, in the layout scripts/simulate.py writes."""
    rng = np.random.default_rng(6)
    patterns = [_poisson(rng, 60) for _ in range(7)]
    sim = tmp_path / "simulation"
    (sim / "toy").mkdir(parents=True)
    np.savez(sim / "toy" / "points.npz", points=np.concatenate(patterns),
             offsets=np.concatenate([[0], np.cumsum([len(p) for p in patterns])]))
    pd.DataFrame({"case_id": [f"toy-{i:05d}-0" for i in range(len(patterns))]}).to_csv(
        sim / "toy" / "manifest.csv", index=False)
    monkeypatch.setattr(C, "SIMULATION", sim)
    monkeypatch.setattr(C, "DATA", tmp_path / "classical")
    return patterns


def test_run_writes_curves_in_manifest_order(fake_simulation, tmp_path):
    cfg = C.Config(grids=(FIXED, SQRTN), f_grid_size=16)
    path = C.run("toy", "fixed", cfg)
    assert path == tmp_path / "classical" / "toy" / "fixed" / "curves.npz"

    z = np.load(path)
    assert set(z) == {"case_id", "axis", "spec", *functions.NAMES}
    assert list(z["case_id"]) == [f"toy-{i:05d}-0" for i in range(len(fake_simulation))]
    assert json.loads(str(z["spec"])) == FIXED
    for name in functions.NAMES:
        assert z[name].shape == (len(fake_simulation), functions.GRID_SIZE)
        assert z[name].dtype == np.float32
        # row i must be pattern i's own curve, not a neighbour's
        np.testing.assert_allclose(
            z[name][i := 3], functions.curves(fake_simulation[i], FIXED, 16)[name], rtol=1e-6)


def test_run_is_resumable_and_tags_are_separate(fake_simulation, tmp_path):
    cfg = C.Config(grids=(FIXED, SQRTN), f_grid_size=16)
    first = C.run("toy", "fixed", cfg)
    stamp = first.stat().st_mtime_ns
    assert C.run("toy", "fixed", cfg).stat().st_mtime_ns == stamp    # existing output is not rebuilt

    other = C.run("toy", "sqrtn_u2", cfg)
    assert other != first and other.exists()
    assert not np.array_equal(np.load(first)["G"], np.load(other)["G"])


def test_config_round_trips_the_shipped_file():
    cfg = C.Config.load()
    assert [functions.tag(s) for s in cfg.grids] == ["fixed", "sqrtn_u2"]
    assert cfg.spec("sqrtn_u2")["u_max"] == 2.0
    assert cfg.f_grid_size == fgj.F_GRID_SIZE
