#!/usr/bin/env python3
# scripts/merge_error_floor_shards.py
"""Merges shard JSONs written by scripts/estimate_error_floor.py
(--group-start/--group-end) into one full-dataset floor_L number.
Concatenates per_group_L across shards by group index (deduplicating and
sorting), checks every shard used the identical label_norm (they must,
per estimate_error_floor.py's "fit on the FULL dataset" rule -- a mismatch
here means shards came from inconsistent runs and should not be merged),
and errors if any theta group in [0, n_groups_total) is missing or
covered by more than one shard.

Usage:
    python scripts/merge_error_floor_shards.py data/error_floor/thomas/error_floor_mincontrast_groups*.json \
        --out data/error_floor/thomas/error_floor_mincontrast.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shards", type=Path, nargs="+", help="shard JSONs from estimate_error_floor.py")
    parser.add_argument("--out", type=Path, required=True, help="path for the merged full-dataset JSON")
    args = parser.parse_args(argv)

    shards = [json.load(open(p)) for p in args.shards]
    if not shards:
        raise ValueError("No shard files given.")

    estimator = shards[0]["estimator"]
    reps = shards[0]["reps"]
    n_groups_total = shards[0]["n_groups_total"]
    label_names = shards[0]["label_names"]
    label_norm = shards[0]["label_norm"]
    for s, path in zip(shards, args.shards):
        if s["estimator"] != estimator or s["reps"] != reps or s["n_groups_total"] != n_groups_total or s["label_names"] != label_names:
            raise ValueError(f"{path} disagrees with {args.shards[0]} on estimator/reps/n_groups_total/label_names -- inconsistent shard set.")
        if s["label_norm"] != label_norm:
            raise ValueError(
                f"{path}'s label_norm differs from {args.shards[0]}'s -- shards were not normalized on the "
                "same population (see estimate_error_floor.py's 'fit on the FULL dataset' note); do not merge these."
            )

    covered: dict[int, tuple[float, int, Path]] = {}
    for s, path in zip(shards, args.shards):
        for local_idx, g in enumerate(range(s["group_start"], s["group_end"])):
            if g in covered:
                raise ValueError(f"theta group {g} is covered by both {covered[g][2]} and {path} -- overlapping shards.")
            covered[g] = (s["per_group_L"][local_idx], s["per_group_n_valid"][local_idx], path)

    missing = sorted(set(range(n_groups_total)) - set(covered))
    if missing:
        raise ValueError(f"theta groups {missing} are not covered by any shard.")

    group_L = np.array([covered[g][0] for g in range(n_groups_total)])
    group_n_valid = [covered[g][1] for g in range(n_groups_total)]

    floor_L = float(np.nanmean(group_L))
    floor_L_std_across_groups = float(np.nanstd(group_L))
    print(f"Merged {len(shards)} shard(s) covering all {n_groups_total} theta groups.")
    print(f"floor L = {floor_L:.4f}  (spread across groups: {floor_L_std_across_groups:.4f})")

    out = {
        "estimator": estimator,
        "reps": reps,
        "sharded": False,
        "group_start": 0,
        "group_end": n_groups_total,
        "n_groups_total": n_groups_total,
        "n_groups": n_groups_total,
        "label_names": label_names,
        "label_norm": label_norm,
        "floor_L": floor_L,
        "floor_L_std_across_groups": floor_L_std_across_groups,
        "per_group_L": group_L.tolist(),
        "per_group_n_valid": group_n_valid,
        "merged_from": [str(p) for p in args.shards],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
