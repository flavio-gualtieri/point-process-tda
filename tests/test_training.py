"""End-to-end: fake diagrams on disk -> scripts/train.py -> predictions.npz.

Small and synthetic, so it runs anywhere; what it checks is the wiring, not the science: that rips H0
becomes a 1-D image and DTM H0 a 2-D one, that imagers are fitted on train rows only, that the split
reaching the model is split_of's, and that predictions come back for exactly the test patterns.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cloudforger.training import data as D

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["poisson", "thomas"]
THETAS = {"train": [0, 1, 2, 3, 4, 5], "val": [7000, 7001], "test": [8000, 8001]}


def _train_module():
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_script"] = module
    spec.loader.exec_module(module)
    return module


def _write_family(root: Path, featurization: Path, family: str, tags: list[str], rng) -> int:
    rows = []
    for split_thetas in THETAS.values():
        for theta in split_thetas:
            for rep in (0, 1):
                n = int(rng.integers(100, 800))
                rows.append({"case_id": f"{family}-{theta:05d}-{rep}", "family": family, "theta": theta,
                             "rep": rep, "nbar": float(n), "delta": 0.5, "n": n,
                             "kappa": 20.0 + n / 100, "mu": 4.0 + rep, "sigma": 0.01 + n / 50000})
    manifest = pd.DataFrame(rows)
    (root / family).mkdir(parents=True, exist_ok=True)
    manifest.to_csv(root / family / "manifest.csv", index=False)

    for tag in tags:
        pairs = {0: [], 1: []}
        for dim in (0, 1):
            for _ in range(len(manifest)):
                m = int(rng.integers(3, 12))
                birth = np.zeros(m) if (dim == 0 and not tag.startswith("dtm")) else rng.uniform(0, 0.05, m)
                pairs[dim].append(np.column_stack([birth, birth + rng.uniform(0.01, 0.08, m)]))
        out = {"case_id": manifest.case_id.to_numpy(str)}
        for dim in (0, 1):
            out[f"h{dim}"] = np.concatenate(pairs[dim])
            out[f"h{dim}_offsets"] = np.concatenate([[0], np.cumsum([len(p) for p in pairs[dim]])])
        (featurization / family / tag).mkdir(parents=True, exist_ok=True)
        np.savez(featurization / family / tag / "diagrams.npz", **out)
    return len(manifest)


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    simulation, featurization = tmp_path / "simulation", tmp_path / "featurization"
    n_rows = sum(_write_family(simulation, featurization, f, ["rips", "dtm_k10"], rng) for f in FAMILIES)
    monkeypatch.setattr(D, "SIMULATION", simulation)
    monkeypatch.setattr(D, "FEATURIZATION", featurization)
    monkeypatch.setattr(D, "FAMILIES", tuple(FAMILIES))
    return n_rows


def test_image_rank_follows_the_filtration(fake_data):
    rips = D.build(FAMILIES, ["rips"], [0, 1], resolution=8, verbose=False)
    assert rips.images[0].shape[1:] == (1, 8)        # H0 births all 0 -> 1-D
    assert rips.images[1].shape[1:] == (1, 8, 8)
    dtm = D.build(FAMILIES, ["dtm_k10"], [0], resolution=8, verbose=False)
    assert dtm.images[0].shape[1:] == (1, 8, 8)      # DTM H0 births are a density -> 2-D


def test_split_and_channel_stacking(fake_data):
    dataset = D.build(FAMILIES, ["rips", "dtm_k10"], [1], resolution=8, verbose=False)
    assert dataset.images[1].shape[1] == 2           # one channel per filtration
    counts = {s: len(dataset.index(s)) for s in ("train", "val", "test")}
    assert counts == {"train": 24, "val": 8, "test": 8}   # 2 families x 2 reps x thetas
    train_thetas = set(dataset.manifest.theta[dataset.index("train")])
    assert train_thetas == set(THETAS["train"])


def test_train_script_writes_test_predictions(fake_data, tmp_path, monkeypatch):
    train = _train_module()
    monkeypatch.setattr(train.D, "SIMULATION", D.SIMULATION)
    monkeypatch.setattr(train.D, "FEATURIZATION", D.FEATURIZATION)
    monkeypatch.setattr(train.D, "FAMILIES", tuple(FAMILIES))
    out = tmp_path / "run"

    train.main(["--task", "classify", "--filtration", "rips", "--dims", "0,1", "--seed", "1",
                "--resolution", "8", "--epochs", "2", "--batch-size", "8", "--out", str(out)])

    z = np.load(out / "predictions.npz")
    assert len(z["case_id"]) == 8                      # the test patterns, and only those
    assert set(z["y_true"]) <= {0, 1}
    assert z["posterior"].shape == (8, len(FAMILIES))
    assert np.allclose(z["posterior"].sum(axis=1), 1.0, atol=1e-5)
    assert torch.load(out / "model.pt", weights_only=True)
    run = __import__("json").loads((out / "run.json").read_text())
    assert run["imagers"]["rips_h0"]["birth_range"] is None
    assert run["labels"] == FAMILIES


def test_params_task_predicts_in_parameter_units(fake_data, tmp_path, monkeypatch):
    train = _train_module()
    monkeypatch.setattr(train.D, "SIMULATION", D.SIMULATION)
    monkeypatch.setattr(train.D, "FEATURIZATION", D.FEATURIZATION)
    out = tmp_path / "params"

    train.main(["--task", "params", "--family", "thomas", "--filtration", "dtm_k10", "--dims", "0",
                "--seed", "1", "--resolution", "8", "--epochs", "2", "--batch-size", "8",
                "--out", str(out)])

    z = np.load(out / "predictions.npz")
    manifest = pd.read_csv(D.SIMULATION / "thomas" / "manifest.csv").set_index("case_id")
    truth = manifest.loc[list(z["case_id"]), D.TARGETS["thomas"]].to_numpy()
    assert np.allclose(z["y_true"], truth, rtol=1e-5)   # y_true is in parameter units, not standardized
    assert (z["y_pred"] > 0).all()                      # log targets come back positive
