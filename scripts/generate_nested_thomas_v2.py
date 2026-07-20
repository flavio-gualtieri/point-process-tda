#!/usr/bin/env python3
# scripts/generate_nested_thomas_v2.py
"""Regenerate data/nested_thomas/{clouds.pkl, adversarial_clouds.pkl,
cloud_generation_manifest.yaml} with a parameter design that:

  1. targets ~7000 train/test clouds,
  2. carves out a fixed ~1000-cloud held-out set (the "adversarial" split;
     same mechanism scripts/generate.py uses for every other process) that
     never changes across reruns of downstream experiments,
  3. hard-enforces a floor of FLOOR_POINTS points per cloud by re-rolling
     the RNG seed (never the parameters) up to MAX_RESAMPLE_ATTEMPTS times,
  4. keeps meta_cluster_scale >= 5x cluster_scale everywhere in RANGES so
     the two cluster scales stay resolvable instead of collapsing into CSR,
  5. keeps the product of meta_parent_intensity x meta_offspring x
     mean_offspring (the expected-point-count driver, since region volume
     is 1) inside roughly [100, 700] across the whole sweep so the 100-500
     working band holds for the bulk of clouds without truncating the
     sampled distribution.

See docs/nested_thomas_v2_data_report.md for the full rationale and the
resulting dataset statistics.

Usage:
    python scripts/generate_nested_thomas_v2.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.core.design import (
    cloud_stats,
    default_region,
    repeat_vectors,
    sample_param_vector,
    split_adversarial_vectors,
)
from cloudforger.core.io import dump_pickle
from cloudforger.core.records import cloud_to_record
from cloudforger.paths import DEFAULT_DATA_ROOT, DataPaths
from cloudforger.processes.nested_thomas import NestedThomasProcess

PROCESS_NAME = "nested_thomas"
SEED = 0
ADVERSARIAL_SEED_OFFSET = 100_000

# 2000 distinct parameter draws x 4 noise realizations each = 8000 clouds
# total; splitting 1750/250 of the *raw* draws before repetition puts
# exactly 7000 in train/test and 1000 in the frozen held-out set.
N_PARAM_VECTORS = 2000
REPS = 4
ADVERSARIAL_FRACTION = 0.125

RANGES = {
    "meta_parent_intensity": {"low": 8.0, "high": 13.0, "scale": "log"},
    "meta_offspring": {"low": 2.5, "high": 4.0, "scale": "log"},
    "meta_cluster_scale": {"low": 0.05, "high": 0.15, "scale": "log"},
    "mean_offspring": {"low": 7.0, "high": 12.0, "scale": "log"},
    "cluster_scale": {"low": 0.003, "high": 0.008, "scale": "log"},
}

FLOOR_POINTS = 75
MAX_RESAMPLE_ATTEMPTS = 50
# Spacing between per-attempt seeds; must exceed MAX_RESAMPLE_ATTEMPTS so
# retries of cloud i never collide with the base seed of cloud i+1.
SEED_STRIDE = 1000


def sample_cloud_with_floor(params: dict[str, float], region, base_seed: int) -> tuple:
    proc = NestedThomasProcess(**params)
    cloud = None
    for attempt in range(MAX_RESAMPLE_ATTEMPTS):
        cloud = proc.sample(region=region, seed=base_seed * SEED_STRIDE + attempt)
        if cloud.n_points >= FLOOR_POINTS:
            return cloud, attempt
    return cloud, MAX_RESAMPLE_ATTEMPTS - 1


def generate(param_vectors: list[dict[str, float]], base_seed: int, region) -> tuple[list, list[int]]:
    clouds, retries = [], []
    for offset, params in enumerate(param_vectors):
        cloud, attempt = sample_cloud_with_floor(params, region, base_seed + offset)
        clouds.append(cloud)
        retries.append(attempt)
    return clouds, retries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="regenerate even if clouds.pkl already exists")
    args = parser.parse_args()

    data_paths = DataPaths(PROCESS_NAME, root=DEFAULT_DATA_ROOT)
    out_clouds = data_paths.clouds()
    if out_clouds.exists() and not args.force:
        print(f"{out_clouds} already exists; pass --force to regenerate.")
        return

    design_rng = np.random.default_rng(SEED)
    param_vectors = [sample_param_vector(RANGES, design_rng) for _ in range(N_PARAM_VECTORS)]

    train_test_vectors, adversarial_vectors, adversarial_indices = split_adversarial_vectors(
        param_vectors,
        {"enabled": True, "fraction": ADVERSARIAL_FRACTION, "seed_offset": ADVERSARIAL_SEED_OFFSET},
        SEED,
    )
    train_test_design = repeat_vectors(train_test_vectors, REPS)
    adversarial_design = repeat_vectors(adversarial_vectors, REPS)

    region = default_region(2)

    print(f"Generating {len(train_test_design)} train/test clouds ...")
    train_clouds, train_retries = generate(train_test_design, SEED, region)
    print(f"Generating {len(adversarial_design)} frozen held-out clouds ...")
    adv_clouds, adv_retries = generate(adversarial_design, SEED + ADVERSARIAL_SEED_OFFSET, region)

    dump_pickle(out_clouds, [cloud_to_record(c) for c in train_clouds])
    dump_pickle(data_paths.clouds(adversarial=True), [cloud_to_record(c) for c in adv_clouds])

    manifest = {
        "process": PROCESS_NAME,
        "seed": SEED,
        "n_param_vectors": N_PARAM_VECTORS,
        "reps": REPS,
        "ranges": RANGES,
        "floor_points": FLOOR_POINTS,
        "max_resample_attempts": MAX_RESAMPLE_ATTEMPTS,
        "adversarial_fraction": ADVERSARIAL_FRACTION,
        "adversarial_seed_offset": ADVERSARIAL_SEED_OFFSET,
        "n_train_test_param_vectors": len(train_test_vectors),
        "n_adversarial_param_vectors": len(adversarial_vectors),
        "adversarial_param_indices": adversarial_indices,
        "train_test_stats": cloud_stats(train_clouds),
        "adversarial_stats": cloud_stats(adv_clouds),
        "train_test_resample_attempts_max": int(max(train_retries)),
        "train_test_resample_attempts_mean": float(np.mean(train_retries)),
        "train_test_clouds_needing_resample": int(sum(1 for r in train_retries if r > 0)),
        "adversarial_resample_attempts_max": int(max(adv_retries)),
        "adversarial_resample_attempts_mean": float(np.mean(adv_retries)),
        "adversarial_clouds_needing_resample": int(sum(1 for r in adv_retries if r > 0)),
    }
    manifest_path = data_paths.manifest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"Saved {len(train_clouds)} train/test clouds -> {out_clouds}")
    print(f"Saved {len(adv_clouds)} frozen held-out clouds -> {data_paths.clouds(adversarial=True)}")
    print(f"Saved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
