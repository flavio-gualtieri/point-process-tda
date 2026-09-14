# tests/test_generation.py
"""DV3 generation (docs/theory/generation.tex): seeding, prior, null tables and
delta-tilde, samplers, and the plan -> run_shard -> merge -> regen_case
pipeline on a tiny spec."""

from __future__ import annotations

import math
from pathlib import Path

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
from cloudforger.data_generation.point_processes import CirculantLGCPProcess
from cloudforger.generation import nulls, validate
from cloudforger.generation.lgcp_grid import choose_grid_M, grid_bias
from cloudforger.generation.plan import complete
from cloudforger.generation.prior import PoissonPrior, ThomasPrior, build_priors, draw, fixed
from cloudforger.generation.samplers import WINDOW, simulate
from cloudforger.generation.seeding import PARAMS, PATTERN, case_rng
from cloudforger.generation.spec import load_spec
from cloudforger.generation.store import DV3Paths, read_csv

ROOT = 70519940839025816406438102027139626253
THOMAS = ThomasPrior(mu={"low": 0.1, "high": 30}, s={"low": 0.1, "high": 1.5},
                     constraints={"sigma_max": 0.1, "kappa_min": 5})
REAL_SPEC = Path(__file__).resolve().parents[1] / "configs" / "generation" / "dv3.yaml"
SPEC = load_spec(REAL_SPEC)
PRIORS = build_priors(SPEC)
TABS = nulls.load_tables(SPEC.null_tables_path)
MID = {  # one mid-prior shape per family
    "poisson": {}, "thomas": {"mu": 4.0, "s": 0.4},
    "nested": {"mu1": 3.0, "mu2": 2.0, "s2": 0.2, "rho": 6.0},
    "matern2": {"tau": 0.3}, "lgcp": {"sigma2": 1.0, "sp": 1.0},
}
AMP = {"thomas": "mu", "nested": "mu2", "matern2": "tau", "lgcp": "sigma2"}


def _mid(family, nbar=250.0, **change):
    prior = PRIORS[family]
    return prior, complete(prior, fixed(prior, nbar, {**MID[family], **change}), TABS)


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


@pytest.mark.parametrize("family", ["poisson", "thomas", "nested", "matern2", "lgcp"])
def test_mean_count_is_nbar(family):
    # A cheap V0: buffers (Neyman-Scott), the inversion (Matern II) and the
    # mu = ln nbar - sigma2 / 2 shift (LGCP) all make E n = nbar.
    _, d = _mid(family, nbar=300.0)
    n = np.array([len(simulate(family, d, case_rng(ROOT, "validation", family, 900_000 + i, PATTERN)).points)
                  for i in range(400)])
    assert abs(n.mean() - 300.0) < 4 * n.std(ddof=1) / math.sqrt(len(n))


def test_matern2_hard_core_holds_up_to_the_edge():
    # V3, and the edge fix: primaries outside W thin points inside it.
    _, d = _mid("matern2", tau=0.5)
    for i in range(50):
        pts = simulate("matern2", d, case_rng(ROOT, "validation", "matern2", 800_000 + i, PATTERN)).points
        nn = np.sort(np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)), axis=1)[:, 1]
        assert nn.min() >= d.model["R"]


def test_lgcp_field_has_the_target_covariance():
    # V7 in miniature: exact embedding, variance sigma2, correlation exp(-lag/s) along an axis.
    proc = CirculantLGCPProcess(mu=0.0, sigma2=1.5, s=0.05, grid_M=64)
    lam, P, ratio = proc.embedding(1.0)
    assert P == 2 and ratio >= -1e-10
    rng = np.random.default_rng(3)
    import scipy.fft
    N = P * 64
    fields = [scipy.fft.fft2(np.sqrt(lam / N**2) * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))))
              .real[:64, :64] for _ in range(300)]
    Y = np.stack(fields)
    assert Y.var() == pytest.approx(1.5, rel=0.03)
    lag = 3  # cells of 1/64
    cov = np.mean(Y[:, :-lag, :] * Y[:, lag:, :])
    assert cov == pytest.approx(1.5 * math.exp(-lag / 64 / 0.05), rel=0.05)


# --- closed forms, null tables, delta-tilde ------------------------------------------

def test_null_tables_are_sane():
    assert np.all(TABS["L"]["c95_se"] <= 0.05)                  # V9
    assert np.all((TABS["L"]["c95"] > 2.5) & (TABS["L"]["c95"] < 4.0))
    assert np.array_equal(TABS["r_grid"], nulls.R_GRID)


def test_tiny_null_table_build_is_deterministic():
    a = nulls.build_tables(ROOT, [50, 80], 20, log=lambda *_: None)
    b = nulls.build_tables(ROOT, [50, 80], 20, log=lambda *_: None)
    assert np.array_equal(a["L"]["m0"], b["L"]["m0"]) and a["L"]["c95"][0] == b["L"]["c95"][0]


def test_delta_tilde_is_zero_at_csr_and_grows_with_the_amplitude():
    assert _mid("poisson")[1].regime["delta_tilde"] == 0.0
    for family, amp in AMP.items():
        lo, hi = {"thomas": (0.5, 8.0), "nested": (0.2, 4.0), "matern2": (0.1, 0.4), "lgcp": (0.1, 2.0)}[family]
        d_lo = _mid(family, **{amp: lo})[1].regime["delta_tilde"]
        d_hi = _mid(family, **{amp: hi})[1].regime["delta_tilde"]
        assert 0 < d_lo < d_hi, family


def test_delta_tilde_grows_like_sqrt_nbar_at_fixed_shape():
    # Principle P2: fixed shape, 4x the points -> about 2x the departure (rule of thumb within ~20%).
    d1 = _mid("thomas", nbar=150.0)[1].regime["delta_tilde"]
    d4 = _mid("thomas", nbar=600.0)[1].regime["delta_tilde"]
    assert 1.6 < d4 / d1 < 2.4


@pytest.mark.parametrize("family", ["thomas", "nested", "lgcp"])
def test_closed_form_K_matches_quadrature_of_the_pcf(family):
    # K(r) = 2 pi int_0^r g(t) t dt, with g from finite differences of K, checked at large r:
    # equivalently K(r) - pi r^2 approaches the known limit (the cluster/field excess).
    prior, d = _mid(family)
    r = np.array([2.0])
    excess = float(prior.excess(r, d.nbar, d.design)[0])
    if family == "thomas":
        limit = 1.0 / d.model["kappa"]
    elif family == "nested":
        limit = (1.0 / (d.model["kappa"] * d.design["mu1"])) + 1.0 / d.model["kappa"]
    else:
        s = d.model["s_abs"]
        k = np.arange(1, 81)
        limit = 2 * np.pi * s**2 * np.sum(np.exp(k * np.log(d.design["sigma2"]) - np.cumsum(np.log(k))) / k**2)
    assert excess == pytest.approx(limit, rel=1e-6)


def test_matern2_K_is_zero_inside_R_and_pi_r2_shifted_beyond_2R():
    prior, d = _mid("matern2")
    R = d.model["R"]
    K = prior.K(np.array([0.5 * R, 3 * R, 4 * R]), d.nbar, d.design)
    assert K[0] == 0.0
    assert K[2] - K[1] == pytest.approx(np.pi * (16 - 9) * R**2, rel=1e-9)


def test_tau_conversions_match_the_doc():
    assert PRIORS["matern2"].regime(250.0, {"tau": 0.1})["tau_K"] / 0.1 == pytest.approx(0.71, abs=0.01)
    assert PRIORS["lgcp"].regime(250.0, {"sigma2": 1.0, "sp": 1.0})["tau_K"] == pytest.approx(1.53, abs=0.01)


def test_lgcp_grid_is_the_smallest_passing_power_of_two():
    s, nbar = 0.5 / math.sqrt(800), 800.0
    M, ratio = choose_grid_M(3.0, s, nbar, TABS)
    assert M > 256 and ratio <= 1.0
    tol = 0.1 * 2 * np.pi * nulls.R_GRID * nulls.s0_at(nbar, "L", TABS)
    sel = nulls.R_GRID >= nulls.L_RMIN
    assert np.max(np.abs(grid_bias(3.0, s, M // 2)[sel]) / tol[sel]) > 1.0
    assert choose_grid_M(0.05, 2.0 / math.sqrt(125), 125.0, TABS)[0] == 256


def test_v1_k_matches_closed_form_for_every_family():
    # The check that ties delta-tilde to the samplers (small batch; the CLI runs 4000).
    report = validate.run(str(REAL_SPEC), reps=300, jobs=1, log=lambda *_: None)
    assert report["pass"], {f: v for f, v in report.items() if isinstance(v, dict)}


# --- pipeline --------------------------------------------------------------------

@pytest.fixture
def tiny(tmp_path):
    spec_path = tmp_path / "dv3.yaml"
    spec_path.write_text(yaml.safe_dump({
        "dataset": "DV3",
        "root": ROOT,
        "nbar": {"low": 100, "high": 800},
        "families": SPEC.families,
        "null_tables": {"path": str(SPEC.null_tables_path), "n_grid": list(SPEC.null_n_grid), "reps": SPEC.null_reps},
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

    assert set(card["outputs"]) == {f"{s}/{f}" for s in ("train", "A") for f in MID}
    assert card["outputs"]["train/thomas"]["splits"] == {"train": 15, "val": 5, "test": 0}
    assert card["outputs"]["A/thomas"]["splits"]["test"] == 12
    assert card["outputs"]["train/matern2"]["V3"]["pass"] and card["outputs"]["A/lgcp"]["V7"]["pass"]

    lgcp = read_csv(paths.manifest("train", "lgcp"))
    assert all(r["grid_M"] >= 256 and r["pad_P"] >= 2 and r["delta_tilde"] > 0 for r in lgcp)
    assert all(r["delta_tilde"] == 0.0 for r in read_csv(paths.manifest("A", "poisson")))

    rows = read_csv(paths.manifest("train", "thomas"))
    assert [r["index"] for r in rows] == list(range(20))
    assert all(r["sampler"] == "exact" and r["n_parents"] is not None for r in rows)

    records = load_pickle(paths.clouds("train", "thomas"))
    cloud = to_pointcloud(records[3])
    assert cloud.n_points == rows[3]["n"] and cloud.generator_params["mu"] == rows[3]["mu"]

    for cid in ("dv3-train-thomas-000011", "dv3-A-nested-000003", "dv3-train-lgcp-000017"):
        report = regen_case(spec, paths, cid)
        assert report["theta_identical"] and report["points_identical"], cid


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
