# tests/test_pipeline_e2e.py
"""End-to-end smoke test: generate -> featurize -> train -> evaluate,
driven through the actual CLI scripts (subprocess, exactly as a user would
invoke them) on a tiny synthetic config, with isolated data/results roots.
This is the same verification run manually while building the refactored
pipeline, kept here as a regression test.

Two cases: the original single-k CNN path (betti_cnn_01), and pi_multik
-- the multi-k (k=5,10) reference pipeline docs/architecture.md is
organized around, which had no dedicated regression test before the
pipeline-housekeeping refactor added this second case.

test_pi_multik_reference_pipeline also regression-tests the train/test
calibration-leakage fix (experiments/pi_multik/pi_multik.py's module
docstring): persistence-image calibration is now fit fresh per seed, on
that seed's train split only, so two different seeds trained against the
IDENTICAL featurized diagrams should end up with DIFFERENT calibrated
axis bounds. The test asserts exactly that (results.json's
config.imager_params differs between seed 1 and seed 2) -- if a future
change accidentally reintroduces a single shared calibration, this
assertion is what would catch it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

CONFIG = {
    "process": {
        "name": "thomas",
        "seed": 3,
        "design": {
            "mode": "random",
            "reps": 1,
            "random": {
                "n_param_vectors": 16,
                "ranges": {
                    "parent_intensity": {"low": 8.0, "high": 15.0, "scale": "log"},
                    "mean_offspring": {"low": 2.0, "high": 4.0, "scale": "log"},
                    "cluster_scale": {"low": 0.01, "high": 0.05, "scale": "log"},
                },
            },
        },
        "adversarial": {"enabled": True, "fraction": 0.25},
    },
    "filtration": [{"name": "dtm", "params": {"k": 5, "maxdim": 1}}],
    "features": [{"name": "betti_curve", "params": {"grid_size": 512, "homology_dims": [0, 1]}}],
    "method": {
        "name": "betti_cnn_01",
        "params": {"batch_size": 4, "n_epochs": 2, "lr": 0.001, "embedding_dim": 8},
    },
    "seeds": [1, 2],
    "target_label_names": ["parent_intensity", "mean_offspring", "cluster_scale"],
    "log_label_names": ["parent_intensity", "mean_offspring", "cluster_scale"],
}

PI_MULTIK_CONFIG = {
    "process": {
        "name": "thomas",
        "seed": 7,
        "design": {
            "mode": "random",
            "reps": 1,
            "random": {
                "n_param_vectors": 24,
                "ranges": {
                    "parent_intensity": {"low": 8.0, "high": 15.0, "scale": "log"},
                    "mean_offspring": {"low": 2.0, "high": 4.0, "scale": "log"},
                    "cluster_scale": {"low": 0.01, "high": 0.05, "scale": "log"},
                },
            },
        },
        "adversarial": {"enabled": True, "fraction": 0.25},
    },
    # Two k's is enough to exercise the multi-source seed-intersection/
    # channel-stacking path (load_multik_split/build_pi_tensor) without
    # paying for the full k=5,10,15 reference grid in a unit test.
    "filtration": [
        {"name": "dtm", "params": {"k": 5, "maxdim": 1}},
        {"name": "dtm", "params": {"k": 10, "maxdim": 1}},
    ],
    "features": [
        {"name": "persistence_image", "params": {
            "resolution": 8, "sigma_pixels": 2.0, "homology_dims": [0, 1], "pd_calibration_coverage": 0.95,
        }},
        {"name": "persistence_entropy", "params": {"homology_dims": [0, 1]}},
    ],
    "method": {
        "name": "pi_multik",
        "params": {
            "homology_dims": [0, 1], "include_entropy": True,
            # Persistence-image calibration -- fit here, per seed, on train
            # rows only (mirrors the features.persistence_image.params
            # above, which scripts/featurize.py still uses to precompute
            # persistence_image.pkl/persistence_entropy.pkl, now unused by
            # pi_multik itself but left in this config to also exercise
            # that codepath).
            "resolution": 8, "sigma_pixels": 2.0, "pd_calibration_coverage": 0.95,
            "embedding_dim": 8, "conv_channels": [4, 8], "dropout": 0.1, "pool_type": "max",
            "head_hidden_dims": [8], "head_dropout": 0.1,
            "batch_size": 4, "n_epochs": 3, "lr": 0.001, "weight_decay": 0.0001,
        },
    },
    "seeds": [1, 2],
    "target_label_names": ["parent_intensity", "mean_offspring", "cluster_scale"],
    "log_label_names": ["parent_intensity", "mean_offspring", "cluster_scale"],
    "use_adversarial": True,
}


def _run(script: str, config_path: Path, data_root: Path, results_root: Path, *extra_args: str) -> None:
    cmd = [
        sys.executable, str(SCRIPTS / script), str(config_path),
        "--set", f"data_root={data_root}",
        "--set", f"results_root={results_root}",
        *extra_args,
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"{script} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


@pytest.mark.slow
def test_full_pipeline_generate_featurize_train_evaluate(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"

    _run("generate.py", config_path, data_root, results_root)
    assert (data_root / "thomas" / "clouds.pkl").exists()
    assert (data_root / "thomas" / "adversarial_clouds.pkl").exists()

    _run("featurize.py", config_path, data_root, results_root)
    assert (data_root / "thomas" / "dtm_k5" / "betti_curve.pkl").exists()

    _run("train.py", config_path, data_root, results_root)
    for seed in (1, 2):
        result_dir = results_root / "thomas" / "dtm_k5" / "betti_cnn_01" / f"seed_{seed}"
        assert (result_dir / "results.json").exists()
        assert (result_dir / "results.pt").exists()
        assert (result_dir / "model.pt").exists()

    _run("evaluate.py", config_path, data_root, results_root, "--methods", "betti_cnn_01")
    assert (results_root / "thomas" / "dtm_k5" / "_compare" / "compare" / "summary.json").exists()


@pytest.mark.slow
def test_pi_multik_reference_pipeline(tmp_path):
    """generate -> featurize -> train for pi_multik specifically -- the
    reference k=5,10,15 pipeline (docs/architecture.md), here with a
    2-k (5, 10) grid to keep the test fast."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(PI_MULTIK_CONFIG))
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"

    _run("generate.py", config_path, data_root, results_root)
    assert (data_root / "thomas" / "clouds.pkl").exists()

    _run("featurize.py", config_path, data_root, results_root)
    for k in (5, 10):
        # diagrams.pkl is what pi_multik itself now reads (see module
        # docstring); persistence_image.pkl/persistence_entropy.pkl are
        # still produced too (features: requests them) but are unused by
        # pi_multik -- kept in this config to exercise that codepath too.
        assert (data_root / "thomas" / f"dtm_k{k}" / "diagrams.pkl").exists()
        assert (data_root / "thomas" / f"dtm_k{k}" / "persistence_image.pkl").exists()
        assert (data_root / "thomas" / f"dtm_k{k}" / "persistence_entropy.pkl").exists()

    _run("train.py", config_path, data_root, results_root)
    results_by_seed = {}
    for seed in (1, 2):
        result_dir = results_root / "thomas" / "dtm_k5+10" / "pi_multik" / f"seed_{seed}"
        assert (result_dir / "results.json").exists()
        assert (result_dir / "model.pt").exists()
        result = json.loads((result_dir / "results.json").read_text())
        assert result["method"] == "pi_multik"
        assert result["config"]["k_values"] == [5, 10]
        assert set(result["test_loss_per_target"]) == {"parent_intensity", "mean_offspring", "cluster_scale"}
        assert result["test_loss"] > 0
        assert "imager_params" in result["config"]
        results_by_seed[seed] = result

    # The actual leakage-fix regression check: seeds 1 and 2 draw different
    # train/val/test splits of the SAME featurized diagrams, so their
    # per-seed-fit calibration (birth_range/pers_range per k) must differ.
    # Equal would mean calibration silently fell back to being shared
    # again.
    assert results_by_seed[1]["config"]["imager_params"] != results_by_seed[2]["config"]["imager_params"]
