"""scripts/regimes.py: predictions + manifest -> one tidy per-regime table.

Predictions are synthesised with a known regime dependence (accuracy rising with delta), so the
table can be checked against something, not just for shape.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["poisson", "thomas", "nested"]
SEEDS = [1, 2]


def _regimes_module():
    spec = importlib.util.spec_from_file_location("regimes_script", ROOT / "scripts" / "regimes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["regimes_script"] = module
    spec.loader.exec_module(module)
    return module


def _manifests(simulation: Path, thetas: list[int], rng) -> pd.DataFrame:
    frames = []
    for family in FAMILIES:
        rows = []
        for theta in thetas:
            delta = 0.0 if family == "poisson" else float(np.exp(rng.uniform(np.log(0.11), np.log(3.9))))
            nbar = float(np.exp(rng.uniform(np.log(110), np.log(790))))
            for rep in (0, 1):
                rows.append({"case_id": f"{family}-{theta:05d}-{rep}", "family": family, "theta": theta,
                             "rep": rep, "nbar": nbar, "delta": delta, "n": int(nbar),
                             "kappa": 20.0, "mu": 5.0, "sigma": 0.02})
        frame = pd.DataFrame(rows)
        (simulation / family).mkdir(parents=True, exist_ok=True)
        frame.to_csv(simulation / family / "manifest.csv", index=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _write_classify_run(results: Path, filtration: str, rows: pd.DataFrame, rng, strength: float):
    """A classifier whose per-pattern accuracy rises with delta, at `strength`."""
    for seed in SEEDS:
        out = results / "classify" / "all" / filtration / "h01" / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        truth = rows.family.map(FAMILIES.index).to_numpy()
        p_correct = np.clip(0.2 + strength * rows.delta.to_numpy(), 0.05, 0.98)
        correct = rng.random(len(rows)) < p_correct
        pred = np.where(correct, truth, (truth + 1) % len(FAMILIES))
        posterior = np.full((len(rows), len(FAMILIES)), 0.1)
        posterior[np.arange(len(rows)), pred] = 1.0 - 0.1 * (len(FAMILIES) - 1)
        np.savez(out / "predictions.npz", case_id=rows.case_id.to_numpy(str), y_true=truth,
                 y_pred=pred, posterior=posterior)
        (out / "run.json").write_text(json.dumps({"args": {"task": "classify"}, "labels": FAMILIES}))


@pytest.fixture
def fake_results(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    regimes = _regimes_module()
    simulation, results = tmp_path / "simulation", tmp_path / "results"
    thetas = [8000 + i for i in range(120)]
    rows = _manifests(simulation, thetas, rng)
    monkeypatch.setattr(regimes, "SIMULATION", simulation)
    _write_classify_run(results, "dtm_k10", rows, rng, strength=0.20)
    _write_classify_run(results, "rips", rows, rng, strength=0.05)
    return regimes, results, tmp_path


def test_table_shape_and_marginals(fake_results):
    regimes, results, tmp_path = fake_results
    regimes.main(["--results", str(results), "--n-boot", "200"])
    table = pd.read_csv(results / "regimes.csv")

    assert set(table.columns) == {"task", "group", "features", "variant", "target", "metric", "family",
                                  "nbar_bin", "delta_bin", "n_thetas", "estimate", "lo", "hi",
                                  "seed_sd", "reference", "n_seeds"}
    assert set(table.metric) == {"accuracy", "nll"}
    assert set(table.family) == {*FAMILIES, "all"}
    assert (table.n_seeds == len(SEEDS)).all()

    # The fully marginal row is the overall estimate, and its interval brackets it.
    overall = table[(table.features == "dtm_k10") & (table.family == "all")
                    & table.nbar_bin.isna() & table.delta_bin.isna() & (table.metric == "accuracy")]
    assert len(overall) == 1
    assert overall.lo.iloc[0] <= overall.estimate.iloc[0] <= overall.hi.iloc[0]
    assert overall.n_thetas.iloc[0] == 360          # 3 families x 120 thetas, both replicates pooled


def test_poisson_has_no_delta_bin_but_keeps_the_marginals(fake_results):
    regimes, results, _ = fake_results
    regimes.main(["--results", str(results), "--n-boot", "200"])
    table = pd.read_csv(results / "regimes.csv")

    poisson = table[table.family == "poisson"]
    assert poisson.delta_bin.isna().all()           # delta = 0 falls in no bin
    assert poisson.nbar_bin.notna().any()           # but it still appears in the nbar cells
    assert table[(table.family == "thomas")].delta_bin.notna().any()


def test_accuracy_rises_with_delta(fake_results):
    regimes, results, _ = fake_results
    regimes.main(["--results", str(results), "--n-boot", "200"])
    table = pd.read_csv(results / "regimes.csv")

    curve = table[(table.features == "dtm_k10") & (table.family == "thomas")
                  & (table.metric == "accuracy") & table.delta_bin.notna()
                  & table.nbar_bin.isna()].sort_values("delta_bin")
    assert len(curve) >= 5
    assert curve.estimate.iloc[-1] > curve.estimate.iloc[0]      # the planted regime dependence


def test_paired_reference_rows(fake_results):
    regimes, results, _ = fake_results
    regimes.main(["--results", str(results), "--n-boot", "200", "--reference", "rips/h01"])
    table = pd.read_csv(results / "regimes.csv")

    paired = table[(table.reference == "rips/h01") & (table.features == "dtm_k10")
                   & (table.metric == "accuracy") & (table.family == "all")
                   & table.delta_bin.isna() & table.nbar_bin.isna()]
    assert len(paired) == 1
    assert paired.estimate.iloc[0] > 0              # the stronger run beats the reference
    assert paired.lo.iloc[0] < paired.estimate.iloc[0] < paired.hi.iloc[0]
    # A run is never paired against itself.
    assert table[(table.reference == "rips/h01") & (table.features == "rips")].empty
