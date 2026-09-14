#!/usr/bin/env python3
# scripts/generation/dv3.py
"""Generate the DV3 prior-drawn sets (training, A) -- docs/theory/generation.tex.

    python scripts/generation/dv3.py nulls --jobs 8         # CSR null tables (once; commit the .npz)
    python scripts/generation/dv3.py validate --jobs 8      # V1 (K vs closed form) and V4 (edges) batches
    python scripts/generation/dv3.py plan --jobs 8          # dv3.yaml + tables -> data/dv3/plan.csv
    python scripts/generation/dv3.py shards                 # list shards (= SLURM array tasks)
    python scripts/generation/dv3.py run-shard --task 7     # simulate one shard (idempotent)
    python scripts/generation/dv3.py run-shard --set train --family thomas --shard 3
    python scripts/generation/dv3.py merge                  # shards -> data/dv3/<set>/<family>/
    python scripts/generation/dv3.py regen-case dv3-train-thomas-000417
    python scripts/generation/dv3.py all --jobs 8           # plan + every shard + merge, locally

On SLURM: `sbatch --array=0-$((N-1))` where N comes from `shards`, each task
running `run-shard --task $SLURM_ARRAY_TASK_ID`; then one `merge` job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.generation.pipeline import (  # noqa: E402
    PipelineError, list_shards, load_plan, make_nulls, make_plan, merge, regen_case, run_all_shards, run_shard,
)
from cloudforger.generation.spec import load_spec  # noqa: E402
from cloudforger.generation.store import DV3Paths, write_json  # noqa: E402
from cloudforger.generation.validate import run as run_validation  # noqa: E402

DEFAULT_SPEC = ROOT / "configs" / "generation" / "dv3.yaml"
DEFAULT_OUT = ROOT / "data" / "dv3"


def cmd_nulls(spec, paths, args) -> None:
    print(f"null tables: {len(spec.null_n_grid)} sizes x {spec.null_reps} binomial replicates -> {spec.null_tables_path}")
    report = make_nulls(spec, jobs=args.jobs, force=args.force)
    print(f"V9 (Monte Carlo s.e. of c95 <= 0.05): {'PASS' if report['V9_pass'] else 'FAIL'}; "
          f"max s.e. {max(report['c95_se_L']):.3f}")


def cmd_validate(spec, paths, args) -> None:
    print(f"validation batches: {args.reps} replicates per family (set 9 streams)")
    report = run_validation(str(spec.path), reps=args.reps, jobs=args.jobs)
    write_json(paths.root / "validation.json", report)
    print(f"{'ALL PASS' if report['pass'] else 'FAILURES'} -> {paths.root / 'validation.json'}")
    if not report["pass"]:
        raise SystemExit(1)


def cmd_plan(spec, paths, args) -> None:
    meta = make_plan(spec, paths, force=args.force, jobs=args.jobs)
    print(f"plan: {meta['n_cases']} cases -> {paths.plan_csv} (sha256 {meta['plan_sha256'][:12]})")


def cmd_shards(spec, paths, args) -> None:
    shards = list_shards(load_plan(spec, paths)[0])
    for task, (set_, family, shard) in enumerate(shards):
        print(f"{task}\t{set_}\t{family}\t{shard}")
    print(f"{len(shards)} shards; SLURM: --array=0-{len(shards) - 1}", file=sys.stderr)


def cmd_run_shard(spec, paths, args) -> None:
    if args.task is not None:
        set_, family, shard = list_shards(load_plan(spec, paths)[0])[args.task]
    elif None not in (args.set, args.family, args.shard):
        set_, family, shard = args.set, args.family, args.shard
    else:
        raise SystemExit("run-shard needs --task, or all of --set/--family/--shard")
    t0 = time.perf_counter()
    status = run_shard(spec, paths, set_, family, shard, force=args.force)
    print(f"{set_}/{family}/shard {shard}: {status} ({time.perf_counter() - t0:.1f}s)")


def cmd_merge(spec, paths, args) -> None:
    card = merge(spec, paths, allow_dirty=args.allow_dirty)
    for name, out in card["outputs"].items():
        v0, dt = out["V0"], out["delta_tilde"]
        checks = "  ".join(f"{v}: {'PASS' if out[v]['pass'] else 'FAIL'}" for v in ("V0", "V3", "V7") if v in out)
        print(f"{name:16s} {out['n_cases']:6d} cases  n in [{out['n_points']['min']}, {out['n_points']['max']}]  "
              f"mean n/nbar {v0['mean_n_over_nbar']:.4f} +- {v0['se']:.4f}  "
              f"P(delta~ <= 1) {dt['frac_le_1']:.2f}  {checks}")
    print(f"dataset card -> {paths.card}")


def cmd_regen_case(spec, paths, args) -> None:
    report = regen_case(spec, paths, args.case_id)
    print(json.dumps(report, indent=2))
    if not all(v for k, v in report.items() if k.endswith("_identical")):
        raise SystemExit(1)


def cmd_all(spec, paths, args) -> None:
    cmd_plan(spec, paths, args)
    run_all_shards(spec, paths, jobs=args.jobs, force=args.force)
    cmd_merge(spec, paths, args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="generation spec (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output root (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("nulls", help="build the CSR null tables for delta-tilde")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--force", action="store_true", help="rebuild existing tables")
    p.set_defaults(func=cmd_nulls)

    p = sub.add_parser("validate", help="V1/V4 validation batches against the closed forms")
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("plan", help="enumerate every case and draw its theta")
    p.add_argument("--force", action="store_true", help="replace an existing, different plan")
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("shards", help="list shards with their SLURM array task ids")
    p.set_defaults(func=cmd_shards)

    p = sub.add_parser("run-shard", help="simulate one shard")
    p.add_argument("--task", type=int, help="position in the `shards` list (SLURM_ARRAY_TASK_ID)")
    p.add_argument("--set")
    p.add_argument("--family")
    p.add_argument("--shard", type=int)
    p.add_argument("--force", action="store_true", help="rerun even if the shard is complete")
    p.set_defaults(func=cmd_run_shard)

    p = sub.add_parser("merge", help="merge shards, write outputs and the dataset card")
    p.add_argument("--allow-dirty", action="store_true", help="allow an uncommitted tree (throwaway builds)")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("regen-case", help="rebuild one case and compare with the stored copy")
    p.add_argument("case_id")
    p.set_defaults(func=cmd_regen_case)

    p = sub.add_parser("all", help="plan + every shard + merge, locally")
    p.add_argument("--jobs", type=int, default=1, help="worker processes for plan and shards")
    p.add_argument("--force", action="store_true", help="replace the plan and rerun complete shards")
    p.add_argument("--allow-dirty", action="store_true")
    p.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    try:
        args.func(load_spec(args.spec), DV3Paths(args.out), args)
    except PipelineError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
