# tests/test_evaluation.py
"""The DV3 evaluation layer: split, regime grouping, detection calibration,
the classical envelope test's size, prediction bundles, config wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cloudforger.config import RunConfig
from cloudforger.core.splits import resolve_split, train_val_test_indices
from cloudforger.evaluation import classical, regimes as R
from cloudforger.evaluation.dv3 import (
    DELTA_EDGES, Regimes, band_index, case_id, split_indices_from_records,
)
from cloudforger.evaluation.predictions import (
    load_predictions, save_classify_predictions, save_params_predictions, save_score_predictions,
)
from cloudforger.experiments.common import invert_label_norm, record_eval_set
from cloudforger.generation import nulls

NULL_TABLES = Path(__file__).resolve().parents[1] / "configs" / "generation" / "null_tables.npz"


def _regimes(**cols) -> Regimes:
    n = len(next(iter(cols.values())))
    base = {
        "case_id": np.array([f"c{i}" for i in range(n)], dtype=object),
        "set": np.array(["C"] * n, dtype=object), "family": np.array(["thomas"] * n, dtype=object),
        "index": np.arange(n), "cell_id": np.full(n, -1), "level_id": np.full(n, -1), "rep": np.arange(n),
        "nbar": np.full(n, 125.0), "delta_tilde": np.ones(n), "tau_K": np.ones(n), "scale": np.ones(n),
        "amplitude": np.ones(n), "n": np.full(n, 125), "split": np.array(["test"] * n, dtype=object),
    }
    base.update(cols)
    return Regimes(base)


# --- split ------------------------------------------------------------------------

def test_dv3_split_is_read_not_drawn():
    recs = [{"split": "train"}] * 7 + [{"split": "val"}] * 3
    tr, va = split_indices_from_records(recs)
    assert tr.tolist() == list(range(7)) and va.tolist() == [7, 8, 9]
    # identical for every seed -- that is the point
    a = resolve_split(10, 1, [r["split"] for r in recs])
    b = resolve_split(10, 999, [r["split"] for r in recs])
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and len(a[2]) == 0


def test_dv3_split_refuses_test_rows_in_training_pool():
    with pytest.raises(ValueError, match="test"):
        split_indices_from_records([{"split": "train"}, {"split": "val"}, {"split": "test"}])


def test_legacy_split_unchanged():
    for x, y in zip(resolve_split(100, 7), train_val_test_indices(100, 7)):
        assert np.array_equal(x, y)


def test_reshuffle_stays_inside_pool():
    tr, va, te = resolve_split(200, 0, ["train"] * 170 + ["val"] * 30, reshuffle_seed=3)
    assert len(te) == 0 and len(tr) + len(va) == 200 and not set(tr) & set(va)


# --- regimes ----------------------------------------------------------------------

def test_band_index_marks_out_of_range():
    idx = band_index(np.array([0.0, 0.49, 0.5, 7.9, 100.0, np.nan, -1.0]), DELTA_EDGES)
    assert idx.tolist() == [0, 0, 1, 4, 5, -1, -1]


def test_case_id_matches_generator():
    assert case_id("C", "poisson", 1000000) == "dv3-C-poisson-1000000"
    assert case_id("A", "thomas", 17) == "dv3-A-thomas-000017"


def test_groupings_c_ladders_and_anchors():
    reg = _regimes(
        family=np.array(["thomas"] * 4 + ["poisson"] * 2, dtype=object),
        cell_id=np.array([0, 0, 1, 1, 0, 1]), level_id=np.array([0, 1, 0, 1, 0, 0]),
        nbar=np.array([125, 125, 500, 500, 125, 250.0]),
    )
    g = R.groupings(reg, "C")
    assert set(g) == {"C/thomas/ladder=0/level=0", "C/thomas/ladder=0/level=1", "C/thomas/ladder=1/level=0",
                      "C/thomas/ladder=1/level=1", "C/poisson/nbar=125", "C/poisson/nbar=250"}


def test_params_metrics_count_failures_and_bias():
    truth = np.zeros((4, 2))
    pred = np.array([[1.0, 0.0], [1.0, 0.0], [np.nan, 0.0], [1.0, 0.0]])
    m = R.params_metrics(pred, truth, ["a", "b"])
    assert m["fail_rate"] == 0.25 and m["n_ok"] == 3
    assert m["bias_std"]["a"] == pytest.approx(1.0) and m["loss"] == pytest.approx(0.5)


# --- detection ------------------------------------------------------------------

def test_threshold_calibration_holds_size():
    rng = np.random.default_rng(0)
    n = 4000
    reg = _regimes(family=np.array(["poisson"] * n, dtype=object), rep=np.arange(n),
                   nbar=np.where(np.arange(n) < n // 2, 125.0, 500.0))
    thr = R.calibrate_threshold(rng.normal(size=n), reg, alpha=0.05)
    for p in thr.size.values():
        assert abs(p - 0.05) < 0.02
    assert thr(np.array([250.0]))[0] == pytest.approx(np.interp(np.log(250), np.log([125, 500]), thr.value))


@pytest.mark.parametrize("n", [137, 300])
def test_envelope_F_has_nominal_size_off_grid(n):
    """Regression: nulls.departure's statistic interpolation gives F size ~1
    at off-grid n; the moment interpolation must keep it near 5%."""
    tabs = nulls.load_tables(NULL_TABLES)
    curves = np.stack([nulls.null_curves(4242, n, 95000 + r) for r in range(300)])
    for k, func in enumerate(nulls.FUNCS):
        size = (classical.studentised_departure(curves[:, k], np.full(300, n), func, tabs) > 1).mean()
        assert size < 0.11, (func, size)


# --- bundles --------------------------------------------------------------------

def test_prediction_bundles_roundtrip(tmp_path):
    ids, fam = ["x", "y"], ["thomas", "thomas"]
    save_params_predictions(tmp_path, "A", case_id=ids, family=fam, label_names=["kappa"],
                            pred_std=np.array([[0.1], [np.nan]]), truth_std=np.zeros((2, 1)))
    b = load_predictions(tmp_path / "predictions_A.npz")
    assert b["task"] == "params" and list(b["case_id"]) == ids and np.isnan(b["pred_std"][1, 0])
    save_classify_predictions(tmp_path, "B", case_id=ids, family=fam, class_names=["poisson", "thomas"],
                              proba=np.array([[0.9, 0.1], [0.2, 0.8]]), truth=np.array([1, 1]))
    assert load_predictions(tmp_path / "predictions_B.npz")["task"] == "classify"
    save_score_predictions(tmp_path, "C", case_id=ids, family=fam, score=np.array([0.3, 2.0]),
                           score_name="S_L", meta={"nominal_threshold": 1.0})
    c = load_predictions(tmp_path / "predictions_C.npz")
    assert c["task"] == "detect" and c["meta"]["nominal_threshold"] == 1.0


def test_record_eval_set_params_loss_and_raw(tmp_path):
    norm = {"mean": np.array([0.0]), "std": np.array([1.0])}
    s = record_eval_set(tmp_path, "A", task="params", outputs=np.array([[1.0], [0.0]]),
                        truth=np.array([[0.0], [0.0]]), case_id=np.array(["a", "b"]),
                        family=np.array(["thomas"] * 2), names=["kappa"], label_norm=norm,
                        truth_raw=np.ones((2, 1)))
    assert s["loss"] == pytest.approx(0.5)
    b = load_predictions(tmp_path / "predictions_A.npz")
    assert b["pred_raw"][0, 0] == pytest.approx(np.e)
    assert invert_label_norm(np.zeros((1, 1)), norm)[0, 0] == pytest.approx(1.0)


# --- config -----------------------------------------------------------------------

def test_data_config_defaults_to_legacy():
    cfg = RunConfig.from_dict({"process": {"name": "thomas"}})
    assert not cfg.data.is_dv3
    cfg = RunConfig.from_dict({"process": {"name": "dv3_thomas"},
                               "data": {"source": "dv3", "group": "thomas", "eval_sets": ["A", "C"]}})
    assert cfg.data.is_dv3 and cfg.data.eval_sets == ["A", "C"]
    with pytest.raises(ValueError):
        RunConfig.from_dict({"process": {"name": "x"}, "data": {"source": "dv4"}})


def test_every_dv3_config_parses():
    from cloudforger.config import load_config

    paths = sorted((Path(__file__).resolve().parents[1] / "configs" / "runs" / "dv3").rglob("*.yaml"))
    assert paths
    for p in paths:
        cfg = load_config(p)
        assert cfg.data.is_dv3 and cfg.method is not None and cfg.target_label_names, p
