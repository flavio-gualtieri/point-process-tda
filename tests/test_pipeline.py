"""End to end on a tiny fake bank: tables -> train (a tree and a network unit) -> compare.

The scripts run as subprocesses with CLOUDFORGER_DATA / CLOUDFORGER_RESULTS pointing into tmp_path,
exactly as the SLURM jobs run them, so this checks the wiring between stages and the output
contracts, not the science.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["poisson", "thomas"]
THETAS = {"train": list(range(8)), "val": [7000, 7001], "test": [8000, 8001, 8002]}


def _bank(data: Path, rng) -> None:
    for family in FAMILIES:
        rows, points = [], []
        for thetas in THETAS.values():
            for theta in thetas:
                kappa, mu, sigma = 10 + theta % 7, 3 + theta % 3, 0.02 + 0.01 * (theta % 4)
                for rep in (0, 1):
                    n = int(rng.integers(40, 80))
                    points.append(rng.random((n, 2)))
                    rows.append({"case_id": f"{family}-{theta:05d}-{rep}", "family": family, "theta": theta,
                                 "rep": rep, "nbar": float(n), "n": n,
                                 **({"kappa": kappa, "mu": mu, "sigma": sigma} if family == "thomas" else {})})
        m = pd.DataFrame(rows)
        (data / "bank" / family).mkdir(parents=True)
        m.to_csv(data / "bank" / family / "manifest.csv", index=False)
        sizes = [len(p) for p in points]
        np.savez(data / "bank" / family / "points.npz", points=np.concatenate(points).astype(np.float32),
                 offsets=np.concatenate([[0], np.cumsum(sizes)]))
        # rips diagrams: H0 births 0, H1 anywhere
        out = {"case_id": m.case_id.to_numpy(str)}
        for d in (0, 1):
            pairs = []
            for _ in range(len(m)):
                k = int(rng.integers(3, 10))
                b = np.zeros(k) if d == 0 else rng.uniform(0, 0.05, k)
                pairs.append(np.column_stack([b, b + rng.uniform(0.01, 0.08, k)]))
            out[f"h{d}"] = np.concatenate(pairs)
            out[f"h{d}_offsets"] = np.concatenate([[0], np.cumsum([len(p) for p in pairs])])
        (data / "featurization" / family / "rips").mkdir(parents=True)
        np.savez(data / "featurization" / family / "rips" / "diagrams.npz", **out)
        # summary-function curves on both grids
        for grid in ("fixed", "sqrtn_u2"):
            c = {"case_id": m.case_id.to_numpy(str), "axis": np.linspace(0, 1, 32)}
            c |= {f: rng.random((len(m), 32)).astype(np.float32) for f in ("L", "F", "G", "J")}
            (data / "classical" / family / grid).mkdir(parents=True)
            np.savez(data / "classical" / family / grid / "curves.npz", **c)


CONFIG = {
    "name": "test", "families": FAMILIES, "prior": "equal",
    "inputs": {"classical": {"source": "classical"}, "ph_rips": {"source": "ph", "filtration": "rips"}},
    "models": {"hgb_classical": {"learner": "hgb", "inputs": ["classical"], "params": {"max_iter": 10}},
               "hgb_classical_ph": {"learner": "hgb", "inputs": ["classical", "ph_rips"], "params": {"max_iter": 10}},
               "nn_curves": {"learner": "nn", "curves": "L@fixed,F@fixed,G@fixed,J@fixed"}},
    "nn": {"seed": 1, "epochs": 2, "patience": 1, "batch_size": 8, "lr": 0.001},
    "classify": ["hgb_classical", "hgb_classical_ph", "nn_curves"],
    "estimate": ["hgb_classical", "nn_curves"],
    "crossfit_folds": 3, "clip_to_train_range": True,
    "regime": {"classifier": "hgb_classical", "event": "detected", "coordinates": {"thomas": "sigma"},
               "taus": [0.5, 0.9]},
    "compare": {"baseline": "hgb_classical", "bootstrap": 10, "seed": 0, "classifiers": "all",
                "estimators": ["best", "hgb_classical"], "poisson_threshold": None},
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline")
    _bank(tmp / "data", np.random.default_rng(0))
    config = tmp / "config.yaml"
    config.write_text(yaml.safe_dump(CONFIG))
    env = {**os.environ, "CLOUDFORGER_DATA": str(tmp / "data"), "CLOUDFORGER_RESULTS": str(tmp / "results")}

    def script(*args):
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / args[0]), "--config", str(config), *args[1:]],
                             env=env, cwd=ROOT, capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr
        return out.stdout

    return tmp, script


def test_tables_are_built_and_checked(run):
    tmp, script = run
    script("tables.py", "classical", "--workers", "2")
    script("tables.py", "ph", "--workers", "2")
    for source in ("classical", "ph_rips"):
        z = np.load(tmp / "data" / "tables" / source / "thomas.npz")
        manifest = pd.read_csv(tmp / "data" / "bank" / "thomas" / "manifest.csv")
        assert list(z["case_id"]) == list(manifest.case_id)            # manifest order
        assert z["X"].shape == (len(manifest), len(z["names"]))
    assert "all inputs present" in script("tables.py", "check")


def test_units_train_and_honour_the_contract(run):
    tmp, script = run
    listed = script("train.py", "list").splitlines()
    assert len(listed) == 3 + 2 * 1                                     # 3 classifiers, 2 estimators x thomas
    for line in listed:
        _, task, model, *family = line.split()
        script("train.py", task, "--model", model, *(["--family", family[0]] if family else []))
    assert script("train.py", "list", "--todo").strip() == ""           # everything done

    res = tmp / "results" / "test"
    clf = np.load(res / "classify" / "hgb_classical" / "predictions.npz")
    assert clf["posterior"].shape[1] == len(FAMILIES)
    assert set(clf["split"]) == {"train", "val", "test"}                # the reference classifier crossfits
    assert np.allclose(clf["posterior"].sum(1), 1, atol=1e-5)
    net = np.load(res / "classify" / "nn_curves" / "predictions.npz")
    assert set(net["split"]) == {"val", "test"}
    est = np.load(res / "estimate" / "thomas" / "nn_curves" / "predictions.npz")
    assert list(est["targets"]) == ["kappa", "mu", "sigma"] and (est["theta_hat"] > 0).all()
    report = json.loads((res / "estimate" / "thomas" / "hgb_classical" / "report.json").read_text())
    assert "git_commit" in report["provenance"]


def test_compare_scores_every_unit(run):
    tmp, script = run
    script("compare.py")
    out = tmp / "results" / "test" / "compare"
    rep = json.loads((out / "report.json").read_text())
    assert rep["missing"] == []
    assert set(rep["classifiers"]) == set(CONFIG["classify"])
    assert set(rep["estimators"]["thomas"]) == set(CONFIG["estimate"])
    assert len(rep["pipelines"]) == 3 * 2                               # classifiers x estimator assignments
    assert (out / "summary.md").read_text().startswith("# Comparison: test")
