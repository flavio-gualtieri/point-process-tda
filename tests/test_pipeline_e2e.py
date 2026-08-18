# tests/test_pipeline_e2e.py
"""End-to-end smoke test: generate -> featurize -> train -> evaluate,
driven through the actual CLI scripts (subprocess, exactly as a user would
invoke them) on a tiny synthetic config, with isolated data/results roots.
This is the same verification run manually while building the refactored
pipeline, kept here as a regression test.

Three cases: pi_multik and betti_multik -- the multi-k (k=5,10) late-fusion
reference pipelines docs/architecture.md is organized around (persistence
images and Betti curves/Euler characteristic respectively) -- and
betti_cnn, the single-filtration/single-dimension sibling
(experiments/betti_cnn.py). Each gets its own dedicated regression test.

All three test_*_reference_pipeline cases also regression-test their train/test
calibration-leakage fix (see pi_multik.py's and betti_multik.py's module
docstrings): calibration is fit fresh per seed, on that seed's train split
only, so two different seeds trained against the IDENTICAL featurized
diagrams should end up with DIFFERENT calibrated axis bounds. The test
asserts exactly that (results.json's config.imager_params /
config.betti_params differs between seed 1 and seed 2) -- if a future
change accidentally reintroduces a single shared calibration, this
assertion is what would catch it."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

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

BETTI_MULTIK_CONFIG = {
    "process": {
        "name": "thomas",
        "seed": 11,
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
    # Same two-k grid as PI_MULTIK_CONFIG, for the same reason.
    "filtration": [
        {"name": "dtm", "params": {"k": 5, "maxdim": 1}},
        {"name": "dtm", "params": {"k": 10, "maxdim": 1}},
    ],
    # Deliberately empty: betti_multik needs no precomputed features file --
    # it computes Betti/Euler curves on the fly from diagrams.pkl alone
    # (see betti_multik.py's module docstring), unlike pi_multik above,
    # which still exercises the (now-unused-by-pi_multik) persistence_image
    # precompute codepath.
    "features": [],
    "method": {
        "name": "betti_multik",
        "params": {
            "homology_dims": [0, 1], "include_entropy": True,
            # Betti-curve calibration -- fit here, per seed, on train rows
            # only (see betti_multik.py's module docstring). grid_size=64
            # (not the 128 default) and n_filters=4 (not 64) keep this fast;
            # 64 is comfortably above the ~46-sample floor the default
            # kernel_size=7/pool_size=2 conv stack needs to not collapse
            # (see sequence_bank.py's docstring).
            "grid_size": 64, "pd_calibration_coverage": 0.95,
            "embedding_dim": 8, "n_filters": 4, "dropout": 0.1,
            "head_hidden_dims": [8], "head_dropout": 0.1,
            "batch_size": 4, "n_epochs": 3, "lr": 0.001, "weight_decay": 0.0001,
        },
    },
    "seeds": [1, 2],
    "target_label_names": ["parent_intensity", "mean_offspring", "cluster_scale"],
    "log_label_names": ["parent_intensity", "mean_offspring", "cluster_scale"],
    "use_adversarial": True,
}

BETTI_CNN_CONFIG = {
    "process": {
        "name": "thomas",
        "seed": 13,
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
    # A single filtration -- betti_cnn reads exactly one, unlike
    # pi_multik/betti_multik's k-grid.
    "filtration": [{"name": "dtm", "params": {"k": 5, "maxdim": 1}}],
    # Deliberately empty -- betti_cnn computes its Betti curve on the fly
    # from diagrams.pkl alone (see betti_cnn.py's module docstring).
    "features": [],
    "method": {
        "name": "betti_cnn",
        "params": {
            "hom_dim": 1, "include_entropy": True,
            # Betti-curve calibration -- fit here, per seed, on train rows
            # only (see betti_cnn.py's module docstring). grid_size=64 (not
            # the 128 default) keeps this fast, comfortably above the
            # ~46-sample floor the default kernel_size=7/pool_size=2 conv
            # stack needs to not collapse (see sequence_bank.py's
            # docstring, which SequenceCNNEncoder's own default pool_size=2
            # also satisfies).
            "grid_size": 64, "pd_calibration_coverage": 0.95,
            "embedding_dim": 8, "n_filters": 4, "dropout": 0.1,
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


@pytest.mark.slow
def test_betti_multik_reference_pipeline(tmp_path):
    """generate -> featurize -> train for betti_multik specifically -- the
    Betti-curve/Euler-characteristic sibling of pi_multik (see
    betti_multik.py's module docstring), here with a 2-k (5, 10) grid to
    keep the test fast."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(BETTI_MULTIK_CONFIG))
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"

    _run("generate.py", config_path, data_root, results_root)
    assert (data_root / "thomas" / "clouds.pkl").exists()

    _run("featurize.py", config_path, data_root, results_root)
    for k in (5, 10):
        # betti_multik reads diagrams.pkl directly (features: is empty in
        # this config) -- no betti_curve.pkl is produced or expected.
        assert (data_root / "thomas" / f"dtm_k{k}" / "diagrams.pkl").exists()
        assert not (data_root / "thomas" / f"dtm_k{k}" / "betti_curve.pkl").exists()

    _run("train.py", config_path, data_root, results_root)
    results_by_seed = {}
    for seed in (1, 2):
        result_dir = results_root / "thomas" / "dtm_k5+10" / "betti_multik" / f"seed_{seed}"
        assert (result_dir / "results.json").exists()
        assert (result_dir / "model.pt").exists()
        result = json.loads((result_dir / "results.json").read_text())
        assert result["method"] == "betti_multik"
        assert result["config"]["k_values"] == [5, 10]
        assert set(result["test_loss_per_target"]) == {"parent_intensity", "mean_offspring", "cluster_scale"}
        assert result["test_loss"] > 0
        assert "betti_params" in result["config"]
        results_by_seed[seed] = result

    # The same leakage-fix regression check as pi_multik's: seeds 1 and 2
    # draw different train/val/test splits of the SAME featurized diagrams,
    # so their per-seed-fit calibration (grid_range per k) must differ.
    assert results_by_seed[1]["config"]["betti_params"] != results_by_seed[2]["config"]["betti_params"]


@pytest.mark.slow
def test_vec_multik_persistence_image_native_matches_pi_multik(tmp_path):
    """The image-path regression required by the landscape/silhouette build
    spec: method: vec_multik with vectorization=persistence_image,
    encoder_path=native must delegate to (and therefore be byte-identical
    to) method: pi_multik -- see vectorized_multik.py's module docstring.
    Generates/featurizes once, trains both methods against the identical
    featurized data with the same seeds, and compares test losses."""
    pi_config_path = tmp_path / "pi.yaml"
    pi_config_path.write_text(yaml.safe_dump(PI_MULTIK_CONFIG))
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"

    _run("generate.py", pi_config_path, data_root, results_root)
    _run("featurize.py", pi_config_path, data_root, results_root)
    _run("train.py", pi_config_path, data_root, results_root)

    vec_config = copy.deepcopy(PI_MULTIK_CONFIG)
    vec_config["method"] = {
        "name": "vec_multik",
        "params": {
            **PI_MULTIK_CONFIG["method"]["params"],
            "vectorization": "persistence_image",
            "encoder_path": "native",
        },
    }
    vec_config_path = tmp_path / "vec.yaml"
    vec_config_path.write_text(yaml.safe_dump(vec_config))
    _run("train.py", vec_config_path, data_root, results_root)

    for seed in (1, 2):
        pi_result = json.loads(
            (results_root / "thomas" / "dtm_k5+10" / "pi_multik" / f"seed_{seed}" / "results.json").read_text()
        )
        vec_result = json.loads(
            (results_root / "thomas" / "dtm_k5+10" / "vec_multik_persistence_image_native" / f"seed_{seed}" / "results.json").read_text()
        )
        assert abs(pi_result["test_loss"] - vec_result["test_loss"]) < 1e-6
        for name in pi_result["test_loss_per_target"]:
            assert abs(pi_result["test_loss_per_target"][name] - vec_result["test_loss_per_target"][name]) < 1e-6


@pytest.mark.slow
def test_vec_multik_landscape_silhouette_stats(tmp_path):
    """The non-image vectorizations: landscape+native, silhouette+mlp,
    persistence_statistics+mlp (mlp-only, see vectorized_multik.py's
    module docstring), on the same featurized diagrams as PI_MULTIK_CONFIG.
    Also the landscape arm's calibration-leakage regression check, same
    shape as pi_multik's/betti_multik's own tests above."""
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"

    gen_config_path = tmp_path / "gen.yaml"
    gen_config_path.write_text(yaml.safe_dump(PI_MULTIK_CONFIG))
    _run("generate.py", gen_config_path, data_root, results_root)
    _run("featurize.py", gen_config_path, data_root, results_root)

    common_params = {
        "homology_dims": [0, 1], "include_entropy": True,
        "pd_calibration_coverage": 0.95,
        "embedding_dim": 8, "conv_channels": [4, 8], "dropout": 0.1,
        "head_hidden_dims": [8], "head_dropout": 0.1,
        "batch_size": 4, "n_epochs": 3, "lr": 0.001, "weight_decay": 0.0001,
    }
    arms = {
        "landscape_native": (
            "vec_multik_landscape_native",
            {"vectorization": "landscape", "encoder_path": "native", "G": 16, "K": 4},
        ),
        "silhouette_mlp": (
            "vec_multik_silhouette_mlp_p1",
            {"vectorization": "silhouette", "encoder_path": "mlp", "G": 16, "p": 1.0},
        ),
        "stats_mlp": (
            "vec_multik_persistence_statistics_mlp",
            {"vectorization": "persistence_statistics", "encoder_path": "mlp"},
        ),
    }

    results_by_arm: dict[str, dict[int, dict]] = {}
    for arm_name, (subdir, extra_params) in arms.items():
        config = copy.deepcopy(PI_MULTIK_CONFIG)
        config["method"] = {"name": "vec_multik", "params": {**common_params, **extra_params}}
        config_path = tmp_path / f"{arm_name}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        _run("train.py", config_path, data_root, results_root)

        results_by_seed = {}
        for seed in (1, 2):
            result_dir = results_root / "thomas" / "dtm_k5+10" / subdir / f"seed_{seed}"
            assert (result_dir / "results.json").exists()
            result = json.loads((result_dir / "results.json").read_text())
            assert result["method"] == "vec_multik"
            assert set(result["test_loss_per_target"]) == {"parent_intensity", "mean_offspring", "cluster_scale"}
            assert result["test_loss"] > 0
            assert result["config"]["feature_dim"] > 0
            results_by_seed[seed] = result
        results_by_arm[arm_name] = results_by_seed

    a = results_by_arm["landscape_native"][1]["config"]["vectorizer_params"]
    b = results_by_arm["landscape_native"][2]["config"]["vectorizer_params"]
    assert a != b


@pytest.mark.slow
def test_betti_cnn_reference_pipeline(tmp_path):
    """generate -> featurize -> train for betti_cnn specifically -- the
    single-filtration/single-homology-dimension sibling of betti_multik
    (see betti_cnn.py's module docstring)."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(BETTI_CNN_CONFIG))
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"

    _run("generate.py", config_path, data_root, results_root)
    assert (data_root / "thomas" / "clouds.pkl").exists()

    _run("featurize.py", config_path, data_root, results_root)
    # betti_cnn reads diagrams.pkl directly (features: is empty in this
    # config) -- no betti_curve.pkl is produced or expected.
    assert (data_root / "thomas" / "dtm_k5" / "diagrams.pkl").exists()
    assert not (data_root / "thomas" / "dtm_k5" / "betti_curve.pkl").exists()

    _run("train.py", config_path, data_root, results_root)
    results_by_seed = {}
    for seed in (1, 2):
        result_dir = results_root / "thomas" / "dtm_k5" / "betti_cnn_1" / f"seed_{seed}"
        assert (result_dir / "results.json").exists()
        assert (result_dir / "model.pt").exists()
        result = json.loads((result_dir / "results.json").read_text())
        assert result["method"] == "betti_cnn"
        assert set(result["test_loss_per_target"]) == {"parent_intensity", "mean_offspring", "cluster_scale"}
        assert result["test_loss"] > 0
        assert "betti_params" in result["config"]
        results_by_seed[seed] = result

    # The actual leakage-fix regression check: seeds 1 and 2 draw different
    # train/val/test splits of the SAME featurized diagrams, so their
    # per-seed-fit calibration (grid_range) must differ. Equal would mean
    # calibration silently fell back to being shared (i.e. fit on the whole
    # population) again -- exactly the bug this file's module docstring
    # describes fixing.
    assert results_by_seed[1]["config"]["betti_params"] != results_by_seed[2]["config"]["betti_params"]
