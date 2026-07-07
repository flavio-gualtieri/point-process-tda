# scripts/processing/params/generate_inhom_thomas.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline_lib.clouds import (  # noqa: E402
    build_region,
    generate_clouds_for_design,
    repeat_vectors,
    save_clouds,
    write_cloud_manifest,
)

# ---------------------------------------------------------------------------
# Configuration  (PLACEHOLDERS — align with your homogeneous setup)
# ---------------------------------------------------------------------------
N_COVARIATES = 6          # a separate model is needed only when this changes
DIMENSION = 2
BASE_SEED = 20240601
ADVERSARIAL_SEED_OFFSET = 500_000
OUTPUT_FORMAT = "dict"    # dict -> records via cloud_to_record (carries covariates)
DATA_ROOT = Path("data/params/2d/inhom_thomas")

SWEEP = {
    "n_param_vectors": 729,
    "train_reps": 8,
    "adversarial_reps": 8,
    "adversarial_random_fraction": 0.15,   # representative middle-of-range holdout
    "adversarial_corners": 36,             # most-extreme combos forced into holdout
    "ranges": {
        # match these two to thomas_cloudgen.yaml:
        "parent_intensity": (5.0, 200.0),        # log-uniform (kappa)
        "cluster_scale": (0.005, 0.10),           # log-uniform (omega)
        # nuisance / count and covariate-field controls:
        "expected_points": (300.0, 3500.0),      # log-uniform, spans BCI counts
        "length_scale": (0.12, 0.50),            # uniform, covariate smoothness
        "covariate_correlation": (0.0, 0.6),     # uniform, cross-covariate corr
        # per-component beta ~ U[-beta_abs_max, beta_abs_max] (per covariate SD):
        "beta_abs_max": 1.25,
    },
    "region": {
        "low": [0.0, 0.0],
        "high": [1.0, 0.5],
        }
}


# ---------------------------------------------------------------------------
# Sweep + split (inhom-specific)
# ---------------------------------------------------------------------------
def _loguniform(rng, lo, hi, size):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def build_inhom_param_vectors(cfg, n_covariates, rng):
    n = cfg["n_param_vectors"]
    r = cfg["ranges"]
    kappa = _loguniform(rng, *r["parent_intensity"], n)
    omega = _loguniform(rng, *r["cluster_scale"], n)
    npts = _loguniform(rng, *r["expected_points"], n)
    lscale = rng.uniform(*r["length_scale"], n)
    corr = rng.uniform(*r["covariate_correlation"], n)
    beta = rng.uniform(-r["beta_abs_max"], r["beta_abs_max"], size=(n, n_covariates))
    return [
        {
            "parent_intensity": float(kappa[i]),
            "cluster_scale": float(omega[i]),
            "beta": beta[i].tolist(),
            "expected_points": float(npts[i]),
            "length_scale": float(lscale[i]),
            "covariate_correlation": float(corr[i]),
        }
        for i in range(n)
    ]


def _extremeness(pv, ranges):
    def center_half(a, b):
        return (np.log(a) + np.log(b)) / 2.0, (np.log(b) - np.log(a)) / 2.0

    ck, hk = center_half(*ranges["parent_intensity"])
    cw, hw = center_half(*ranges["cluster_scale"])
    cn, hn = center_half(*ranges["expected_points"])
    bnorm = np.linalg.norm(pv["beta"]) / (ranges["beta_abs_max"] * np.sqrt(len(pv["beta"])))
    return (
        abs((np.log(pv["parent_intensity"]) - ck) / hk)
        + abs((np.log(pv["cluster_scale"]) - cw) / hw)
        + abs((np.log(pv["expected_points"]) - cn) / hn)
        + 2.0 * bnorm
    )


def split_inhom_adversarial(param_vectors, cfg, seed):
    """Hold out the most-extreme combos plus a random representative slice, so the
    adversarial set spans the full parameter range with no train/test overlap."""
    rng = np.random.default_rng(seed)
    ranges = cfg["ranges"]
    n = len(param_vectors)

    scores = np.array([_extremeness(p, ranges) for p in param_vectors])
    corners = set(np.argsort(scores)[-cfg["adversarial_corners"] :].tolist())

    rest = [i for i in range(n) if i not in corners]
    n_rand = int(round(cfg["adversarial_random_fraction"] * n))
    rand = set(rng.choice(rest, size=n_rand, replace=False).tolist()) if n_rand else set()

    adv_idx = sorted(corners | rand)
    adv_set = set(adv_idx)
    tt_vectors = [p for i, p in enumerate(param_vectors) if i not in adv_set]
    adv_vectors = [param_vectors[i] for i in adv_idx]
    return tt_vectors, adv_vectors, adv_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    region = build_region(SWEEP, DIMENSION)
    design_rng = np.random.default_rng(BASE_SEED)

    param_vectors = build_inhom_param_vectors(SWEEP, N_COVARIATES, design_rng)
    tt_vectors, adv_vectors, adv_idx = split_inhom_adversarial(param_vectors, SWEEP, BASE_SEED)

    tt_design = repeat_vectors(tt_vectors, SWEEP["train_reps"])
    adv_design = repeat_vectors(adv_vectors, SWEEP["adversarial_reps"])

    tt_clouds = generate_clouds_for_design("inhom_thomas", region, tt_design, BASE_SEED, n_hint=0)
    adv_clouds = generate_clouds_for_design(
        "inhom_thomas", region, adv_design, BASE_SEED + ADVERSARIAL_SEED_OFFSET, n_hint=0
    )

    out = DATA_ROOT
    out.mkdir(parents=True, exist_ok=True)
    save_clouds(out / "clouds.pkl", tt_clouds, OUTPUT_FORMAT)
    save_clouds(out / "adversarial_clouds.pkl", adv_clouds, OUTPUT_FORMAT)
    write_cloud_manifest(
        out / "cloud_manifest.yaml",
        {"process": "inhom_thomas", "dimension": DIMENSION, "n_covariates": N_COVARIATES, "sweep": SWEEP},
        tt_vectors,
        adv_vectors,
        adv_idx,
        tt_clouds,
        adv_clouds,
    )

    print(f"train/test: {len(tt_vectors)} param vectors x {SWEEP['train_reps']} -> {len(tt_clouds)} clouds")
    print(f"adversarial: {len(adv_vectors)} param vectors x {SWEEP['adversarial_reps']} -> {len(adv_clouds)} clouds")
    print(f"written under {out}")


if __name__ == "__main__":
    main()