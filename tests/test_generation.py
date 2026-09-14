# tests/test_generation.py
"""DV3 generation (docs/theory/generation.tex): seeding, prior, samplers, and
the plan -> run_shard -> merge -> regen_case pipeline on a tiny spec."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml
from scipy import stats

from cloudforger.core.io import load_pickle
from cloudforger.core.records import to_pointcloud
from cloudforger.generation import seeding
from cloudforger.generation.pipeline import (
    PipelineError, list_shards, load_plan, make_plan, merge, regen_case, run_shard,
)
from cloudforger.generation.prior import PoissonPrior, ThomasPrior, draw, fixed
from cloudforger.generation.samplers import simulate
from cloudforger.generation.seeding import PARAMS, PATTERN, case_rng
from cloudforger.generation.spec import load_spec
from cloudforger.generation.store import DV3Paths, read_csv

ROOT = 70519940839025816406438102027139626253
THOMAS = ThomasPrior(mu={"low": 0.1, "high": 30}, s={"low": 0.1, "high": 1.5},
                     constraints={"sigma_max": 0.1, "kappa_min": 5})


# --- seeding ---------------------------------------------------------------------

def _first(rng, k=8):
    return rng.random(k)


def test_same_key_same_stream():
    a = _first(case_rng(ROOT, "train", "thomas", 417, PARAMS))
    b = _first(case_rng(ROOT, "train", "thomas", 417, PARAMS))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("change", [
    dict(root=ROOT + 1), dict(set_="A"), dict(family="poisson"), dict(index=418), dict(role=PATTERN),
])
def test_every_key_component_changes_the_stream(change):
    base = dict(root=ROOT, set_="train", family="thomas", index=417, role=PARAMS)
    a = _first(case_rng(**base))
    b = _first(case_rng(**{**base, **change}))
    assert not np.array_equal(a, b)


def test_packed_indices_cannot_alias():
    assert seeding.index_B(3, 9_999) + 1 == seeding.index_B(4, 0)
    with pytest.raises(ValueError):
        seeding.index_B(3, 10_000)
    with pytest.raises(ValueError):
        seeding.index_C(0, 100, 0)


def test_case_id_round_trip_and_r_seed_range():
    cid = seeding.case_id("train", "thomas", 417)
    assert cid == "dv3-train-thomas-000417"
    assert seeding.parse_case_id(cid) == ("train", "thomas", 417)
    assert 1 <= seeding.r_seed(ROOT, "B", "strauss", 12) <= 2**31 - 1


def test_root_must_be_128_bit():
    with pytest.raises(ValueError):
        seeding.check_root(2**128)


# --- prior -----------------------------------------------------------------------

def test_worked_example_inversion():
    # generation.tex, appendix "Worked example": nbar=312, mu=4, s=0.4.
    d = fixed(THOMAS, 312.0, {"mu": 4.0, "s": 0.4})
    assert d.model["kappa"] == pytest.approx(78.0)
    assert d.model["sigma"] == pytest.approx(0.02265, abs=1e-5)
    assert d.regime["tau_K"] == pytest.approx(0.666, abs=1e-3)


def test_inversion_round_trips():
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = draw(THOMAS, rng, 100, 800)
        assert d.nbar / d.model["kappa"] == pytest.approx(d.design["mu"], rel=1e-12)
        assert d.model["sigma"] * math.sqrt(d.nbar) == pytest.approx(d.design["s"], rel=1e-12)


def test_acceptance_rate_matches_closed_form():
    # 14% rejection at nbar=125 (0.891 * 0.968), none at nbar >= 225.
    p = THOMAS.acceptance_probability(125.0)
    assert p == pytest.approx(0.863, abs=1e-3)
    assert THOMAS.acceptance_probability(225.0) == 1.0

    rng = np.random.default_rng(1)
    m = 20_000
    rate = np.mean([THOMAS.accepts(125.0, THOMAS.draw_shape(rng, 125.0)) for _ in range(m)])
    assert abs(rate - p) < 4 * math.sqrt(p * (1 - p) / m)
    assert all(THOMAS.accepts(300.0, THOMAS.draw_shape(rng, 300.0)) for _ in range(2_000))


def test_nbar_marginal_is_log_uniform_and_constraints_hold():
    draws = [draw(THOMAS, case_rng(ROOT, "pilot", "thomas", i, PARAMS), 100, 800) for i in range(4_000)]
    u = np.array([math.log(d.nbar / 100) / math.log(8) for d in draws])
    assert stats.kstest(u, "uniform").pvalue > 1e-3
    assert all(d.model["sigma"] <= 0.1 and d.model["kappa"] >= 5 for d in draws)


def test_fixed_theta_must_satisfy_constraints():
    with pytest.raises(ValueError):
        fixed(THOMAS, 125.0, {"mu": 29.0, "s": 0.4})  # kappa = 4.3 < 5


# --- samplers --------------------------------------------------------------------

def test_pattern_stream_reproduces_points():
    d = fixed(THOMAS, 312.0, {"mu": 4.0, "s": 0.4})
    a = simulate("thomas", d, case_rng(ROOT, "pilot", "thomas", 0, PATTERN))
    b = simulate("thomas", d, case_rng(ROOT, "pilot", "thomas", 0, PATTERN))
    assert np.array_equal(a.points, b.points) and a.diagnostics["sha1"] == b.diagnostics["sha1"]
    assert a.diagnostics["buffer"] == pytest.approx(3.719 * d.model["sigma"], rel=1e-3)
    assert ((a.points >= 0) & (a.points <= 1)).all()


@pytest.mark.parametrize("family,prior,shape", [
    ("poisson", PoissonPrior(), {}),
    ("thomas", THOMAS, {"mu": 4.0, "s": 0.4}),
])
def test_mean_count_is_nbar(family, prior, shape):
    # A cheap V0: the buffer makes E n = nbar exact for Thomas.
    d = fixed(prior, 300.0, shape)
    n = np.array([len(simulate(family, d, case_rng(ROOT, "validation", family, i, PATTERN)).points)
                  for i in range(600)])
    assert abs(n.mean() - 300.0) < 4 * n.std(ddof=1) / math.sqrt(len(n))


# --- pipeline --------------------------------------------------------------------

@pytest.fixture
def tiny(tmp_path):
    spec_path = tmp_path / "dv3.yaml"
    spec_path.write_text(yaml.safe_dump({
        "dataset": "DV3",
        "root": ROOT,
        "nbar": {"low": 100, "high": 800},
        "families": {
            "poisson": {},
            "thomas": {"mu": {"low": 0.1, "high": 30}, "s": {"low": 0.1, "high": 1.5},
                       "constraints": {"sigma_max": 0.1, "kappa_min": 5}},
        },
        "sets": {"train": {"size": 20, "n_val": 5}, "A": {"size": 12}},
        "shard_size": 8,
    }))
    return load_spec(spec_path), DV3Paths(tmp_path / "dv3")


def _run_all(spec, paths):
    make_plan(spec, paths)
    for s in list_shards(load_plan(spec, paths)[0]):
        run_shard(spec, paths, *s)
    return merge(spec, paths, allow_dirty=True)


def test_pipeline_end_to_end(tiny):
    spec, paths = tiny
    card = _run_all(spec, paths)

    assert set(card["outputs"]) == {"train/poisson", "train/thomas", "A/poisson", "A/thomas"}
    assert card["outputs"]["train/thomas"]["splits"] == {"train": 15, "val": 5, "test": 0}
    assert card["outputs"]["A/thomas"]["splits"]["test"] == 12

    rows = read_csv(paths.manifest("train", "thomas"))
    assert [r["index"] for r in rows] == list(range(20))
    assert all(r["sampler"] == "exact" and r["n_parents"] is not None for r in rows)

    records = load_pickle(paths.clouds("train", "thomas"))
    cloud = to_pointcloud(records[3])
    assert cloud.n_points == rows[3]["n"] and cloud.generator_params["mu"] == rows[3]["mu"]

    report = regen_case(spec, paths, "dv3-train-thomas-000011")
    assert report["theta_identical"] and report["points_identical"]


def test_plan_is_deterministic_and_shards_are_idempotent(tiny):
    spec, paths = tiny
    sha = make_plan(spec, paths)["plan_sha256"]
    assert make_plan(spec, paths)["plan_sha256"] == sha
    run_shard(spec, paths, "train", "thomas", 0)
    assert run_shard(spec, paths, "train", "thomas", 0).startswith("skipped")


def test_merge_refuses_missing_or_tampered_shards(tiny):
    spec, paths = tiny
    _run_all(spec, paths)
    npz, _, _ = paths.shard_files("A", "thomas", 1)
    npz.write_bytes(npz.read_bytes() + b"x")
    with pytest.raises(PipelineError, match="missing, incomplete or stale"):
        merge(spec, paths, allow_dirty=True)


def test_changed_spec_invalidates_the_plan(tiny):
    spec, paths = tiny
    make_plan(spec, paths)
    spec.path.write_text(spec.path.read_text().replace("shard_size: 8", "shard_size: 9"))
    with pytest.raises(PipelineError, match="changed since the plan"):
        load_plan(load_spec(spec.path), paths)
