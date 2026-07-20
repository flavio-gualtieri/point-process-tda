# tests/test_pipeline_e2e.py
"""End-to-end smoke test: generate -> featurize -> train -> evaluate,
driven through the actual CLI scripts (subprocess, exactly as a user would
invoke them) on a tiny synthetic config, with isolated data/results roots.
This is the same verification run manually while building the refactored
pipeline, kept here as a regression test."""

from __future__ import annotations

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
