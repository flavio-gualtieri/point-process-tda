#!/usr/bin/env python3
# scripts/estimate_error_floor.py
"""Computes the irreducible error floor (writeup's sec:problem, SS4): for
each of the N parameter vectors in an error_floor.yaml-generated dataset
(scripts/generate.py), fits theta-hat on each of its >=200 independent
realizations via a classical baseline estimator (mincontrast by default),
then measures how much theta-hat disperses around the TRUE theta shared by
that group of realizations.

Reported in the SAME units as every trained method's test_loss: per-target
squared error in log-zscore-normalized space, averaged over targets then
over realizations then over theta groups -- cloudforger.experiments.
common's convention (also scripts/train.py's run_classical_baseline, which
this script's per-cloud fitting loop mirrors closely), NOT sec:problem's
inline sqrt(sum of squares) formula, which nothing in this codebase
actually computes.

Grouping: clouds.pkl row i belongs to theta-group i // --reps (see
error_floor.yaml's header -- data_generation/design.py's repeat_vectors
orders clouds in consecutive theta-blocks, not interleaved).

Normalization caveat: label_norm (the log-zscore mean/std) is fit HERE, on
this dataset's own unique theta's, by default -- consistent within this
floor estimate, but not byte-identical to any one trained seed's own
fitted stats (fit on a different, much larger, differently-drawn
population). Pass --label-norm-from a results.json path to reuse that
seed's exact stats instead, if exact ratio-comparability to one specific
run matters more than a self-contained floor estimate.

Usage:
    python scripts/estimate_error_floor.py data/error_floor/thomas/clouds.pkl --reps 200
    python scripts/estimate_error_floor.py data/error_floor/thomas/clouds.pkl --reps 200 --estimator mincontrast_g

Sharding (--group-start/--group-end, half-open theta-group index range,
e.g. for a SLURM array splitting 50 groups into 10 shards of 5 each): each
shard writes its own JSON (filename suffixed with the group range) with
its own per_group_L list over just that shard's groups; merge shards with
scripts/merge_error_floor_shards.py before reading floor_L off any single
shard's file (a shard's own floor_L/floor_L_std_across_groups are only
over ITS groups, not the full dataset).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.baselines import mincontrast, mincontrast_g, vihrs

ESTIMATORS = {"mincontrast": mincontrast, "mincontrast_g": mincontrast_g}


def load_clouds(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, list) else [data]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clouds", type=Path, help="path to an error_floor.yaml-generated clouds.pkl")
    parser.add_argument("--reps", type=int, required=True, help="realizations per theta (design.reps used to generate this file)")
    parser.add_argument("--estimator", choices=sorted(ESTIMATORS), default="mincontrast")
    parser.add_argument("--target-label-names", nargs="+", default=None, help="defaults to every key in the first cloud's params")
    parser.add_argument("--n-starts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for crop_and_rescale")
    parser.add_argument("--label-norm-from", type=Path, default=None, help="reuse an existing results.json's label_norm instead of refitting here")
    parser.add_argument("--group-start", type=int, default=None, help="first theta-group index (inclusive) to process, for sharding -- default: 0")
    parser.add_argument("--group-end", type=int, default=None, help="last theta-group index (exclusive) to process, for sharding -- default: all groups")
    parser.add_argument("--out", type=Path, default=None, help="where to write the per-group/aggregate JSON (default: alongside --clouds)")
    args = parser.parse_args(argv)

    all_clouds = load_clouds(args.clouds)
    n_total = len(all_clouds)
    if n_total % args.reps != 0:
        raise ValueError(f"{n_total} clouds is not a multiple of --reps={args.reps}; wrong file or wrong --reps.")
    n_groups_total = n_total // args.reps

    group_start = args.group_start if args.group_start is not None else 0
    group_end = args.group_end if args.group_end is not None else n_groups_total
    if not (0 <= group_start < group_end <= n_groups_total):
        raise ValueError(f"--group-start/--group-end ({group_start}, {group_end}) out of range [0, {n_groups_total}].")
    sharded = (group_start, group_end) != (0, n_groups_total)
    clouds = all_clouds[group_start * args.reps: group_end * args.reps]
    n = len(clouds)
    n_groups = group_end - group_start

    label_names = args.target_label_names or list(clouds[0]["params"].keys())
    true_targets = np.array([[c["params"][name] for name in label_names] for c in clouds], dtype=float)

    module = ESTIMATORS[args.estimator]
    rng = np.random.default_rng(args.seed)

    print(f"Fitting {args.estimator} on {n} clouds ({n_groups} theta groups x {args.reps} realizations) ...")
    raw_predictions = np.full((n, len(label_names)), np.nan)
    n_fit = 0
    for i, rec in enumerate(clouds):
        points = np.asarray(rec["points"], dtype=float)
        if len(points) < 5:
            continue
        reg = rec.get("region", {})
        low = np.asarray(reg.get("low", [0.0, 0.0]), dtype=float)
        high = np.asarray(reg.get("high", [1.0, 1.0]), dtype=float)
        points_unit = module.crop_and_rescale(points, low, high, rng=rng)
        result = module.fit_multistart(points_unit, n_starts=args.n_starts, rng=rng)
        raw_predictions[i] = [result.get(name, np.nan) for name in label_names]
        n_fit += 1
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  fit {i + 1}/{n} clouds ...")
    print(f"Fit {n_fit}/{n} clouds with {args.estimator}.")

    if args.label_norm_from is not None:
        with open(args.label_norm_from) as f:
            saved = json.load(f)["label_norm"]
        label_norm = {"mean": np.asarray(saved["mean"]), "std": np.asarray(saved["std"])}
    else:
        # Fit on the FULL dataset's targets, not just this shard's -- if
        # sharded, every shard must normalize on the same scale or merging
        # shards' squared errors afterwards would silently mix incompatible
        # units. Repetition doesn't bias mean/std here -- every one of the
        # n_groups_total unique theta's contributes exactly `reps`
        # equal-weight rows, same as fitting on the uniques directly.
        all_true_targets = np.array(
            [[c["params"][name] for name in label_names] for c in all_clouds], dtype=float,
        )
        label_norm = vihrs.fit_log_zscore(all_true_targets)

    valid = np.isfinite(raw_predictions).all(axis=1) & (raw_predictions > 0).all(axis=1)
    pred_std = np.full_like(raw_predictions, np.nan)
    pred_std[valid] = vihrs.apply_log_zscore(raw_predictions[valid], label_norm)
    truth_std = vihrs.apply_log_zscore(true_targets, label_norm)

    sq_err = (pred_std - truth_std) ** 2  # (n, n_targets), NaN where the fit failed/was invalid
    per_cloud_L = np.nanmean(sq_err, axis=1)  # (n,) -- per-target-then-averaged MSE, matches every reported test_loss

    group_ids = np.arange(n) // args.reps
    group_L = np.array([np.nanmean(per_cloud_L[group_ids == g]) for g in range(n_groups)])
    group_n_valid = np.array([int(valid[group_ids == g].sum()) for g in range(n_groups)])

    floor_L = float(np.nanmean(group_L))
    floor_L_std_across_groups = float(np.nanstd(group_L))

    label = f"groups [{group_start}, {group_end}) of {n_groups_total}" if sharded else f"all {n_groups_total} theta groups"
    print(f"\n{'Shard' if sharded else 'Irreducible error floor'} ({args.estimator}, {label} x ~{args.reps} realizations):")
    print(f"  {'shard' if sharded else 'floor'} L = {floor_L:.4f}  (spread across these groups: {floor_L_std_across_groups:.4f})")
    if sharded:
        print("  This is a SHARD, not the full floor -- merge every shard's JSON with "
              "scripts/merge_error_floor_shards.py before reading floor_L.")

    out = {
        "estimator": args.estimator,
        "clouds_path": str(args.clouds),
        "reps": args.reps,
        "sharded": sharded,
        "group_start": group_start,
        "group_end": group_end,
        "n_groups_total": n_groups_total,
        "n_groups": n_groups,
        "n_fit": n_fit,
        "n_total": n,
        "label_names": label_names,
        "label_norm": {"mean": label_norm["mean"].tolist(), "std": label_norm["std"].tolist()},
        "floor_L": floor_L,
        "floor_L_std_across_groups": floor_L_std_across_groups,
        "per_group_L": group_L.tolist(),
        "per_group_n_valid": group_n_valid.tolist(),
    }
    if args.out is not None:
        out_path = args.out
    elif sharded:
        out_path = args.clouds.parent / f"error_floor_{args.estimator}_groups{group_start}-{group_end}.json"
    else:
        out_path = args.clouds.parent / f"error_floor_{args.estimator}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
