"""Simulate the cloud bank.

    python scripts/simulate.py run --jobs 10                       # every shard of every family (resumable)
    python scripts/simulate.py run --family lgcp --shard 3         # one shard
    python scripts/simulate.py run --task 17                       # (family, shard) no. 17 (a SLURM array task)
    python scripts/simulate.py tasks                               # how many (family, shard) tasks there are
    python scripts/simulate.py merge                               # -> <data>/bank/<family>/{points.npz, manifest.csv}
    python scripts/simulate.py check --jobs 10                     # samplers vs closed-form K
    python scripts/simulate.py compare --other data                # every pattern identical to another bank's?

Families, bank size and seeds are configs/simulation.yaml's. A shard that exists is skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from cloudforger.departure.tables import Tables
from cloudforger.simulation.check import check_theta
from cloudforger.simulation.families import FAMILIES
from cloudforger.simulation.bank import DATA, Config, run_shard



def _shard(args):
    t0 = time.time()
    run_shard(*args)
    return args[0], args[1], time.time() - t0


def _pool(f, tasks, jobs):
    if jobs == 1:
        yield from map(f, tasks)
        return
    with get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
        yield from pool.imap_unordered(f, tasks)


def all_tasks(cfg) -> list[tuple]:
    return [(f, s, cfg) for f in cfg.families for s in range(cfg.n_shards())]


def cmd_tasks(cfg, args):
    print(len(all_tasks(cfg)))


def cmd_run(cfg, args):
    if args.task is not None:
        tasks = [all_tasks(cfg)[args.task]]
    else:
        families = [args.family] if args.family else cfg.families
        shards = [args.shard] if args.shard is not None else range(cfg.n_shards())
        tasks = [(f, s, cfg) for f in families for s in shards]
    for fam, shard, secs in _pool(_shard, tasks, args.jobs):
        print(f"{fam:8s} shard {shard:3d}  {secs:6.1f}s", flush=True)


def cmd_merge(cfg, args):
    for fam in ([args.family] if args.family else cfg.families):
        paths = sorted((DATA / "shards" / fam).glob("shard_*.npz"))
        expected = cfg.n_shards()
        if len(paths) != expected:
            raise SystemExit(f"{fam}: {len(paths)} of {expected} shards")
        parts = [np.load(p) for p in paths]
        manifest = pd.DataFrame([row for z in parts for row in json.loads(str(z["manifest"]))])
        manifest.insert(0, "case_id", [f"{fam}-{t:05d}-{r}" for t, r in zip(manifest.theta, manifest.rep)])
        sizes = np.concatenate([z["sizes"] for z in parts])
        if not (np.array_equal(sizes, manifest.n) and manifest.case_id.is_unique and len(manifest) == len(cfg.indices()) * cfg.reps):
            raise SystemExit(f"{fam}: shards inconsistent")
        out = DATA / fam
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "points.npz", points=np.concatenate([z["points"] for z in parts]),
                 offsets=np.concatenate([[0], np.cumsum(sizes)]))
        manifest.to_csv(out / "manifest.csv", index=False)
        print(f"{fam:8s} {len(manifest)} patterns, n {manifest.n.min()}-{manifest.n.max()}, "
              f"max tries: draw {manifest.draw_tries.max()}, pattern {manifest.pattern_tries.max()}", flush=True)


def cmd_compare(cfg, args):
    """Every pattern of this bank against the pattern with the same case_id in <other>/bank: the
    parameters (every shared manifest column but delta-tilde, a label) and the points, exactly."""
    bad = 0
    for fam in cfg.families:
        mine = pd.read_csv(DATA / fam / "manifest.csv")
        theirs = pd.read_csv(Path(args.other) / "bank" / fam / "manifest.csv").set_index("case_id")
        missing = ~mine.case_id.isin(theirs.index)
        if missing.any():
            raise SystemExit(f"{fam}: {missing.sum()} patterns not in {args.other} (e.g. {mine.case_id[missing].iloc[0]})")
        cols = [c for c in mine.columns if c in theirs.columns and c != "case_id" and not c.startswith("delta_tilde")]
        other = theirs.loc[mine.case_id, cols].reset_index(drop=True)
        rows_differ = ~(mine[cols].eq(other) | (mine[cols].isna() & other.isna())).all(axis=1)
        a, b = np.load(DATA / fam / "points.npz"), np.load(Path(args.other) / "bank" / fam / "points.npz")
        pa, oa, pb, ob = a["points"], a["offsets"], b["points"], b["offsets"]
        at = pd.Index(theirs.index).get_indexer(mine.case_id)
        pts_differ = [not np.array_equal(pa[oa[i]:oa[i + 1]], pb[ob[j]:ob[j + 1]]) for i, j in enumerate(at)]
        n_bad = int(rows_differ.sum() + np.sum(pts_differ))
        bad += n_bad
        print(f"{fam:8s} {len(mine)} patterns: {int(rows_differ.sum())} parameter rows and "
              f"{int(np.sum(pts_differ))} point sets differ", flush=True)
    if bad:
        raise SystemExit(f"{bad} differences")
    print(f"identical to {args.other}")


def _check(args):
    return check_theta(*args, Tables())


def cmd_check(cfg, args):
    tasks = [(f, i, args.patterns, cfg) for f in cfg.families for i in range(args.thetas)]
    rows = sorted(_pool(_check, tasks, args.jobs), key=lambda r: (cfg.families.index(r["family"]), r["theta"]))
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--family", choices=list(FAMILIES))
    r.add_argument("--shard", type=int)
    r.add_argument("--task", type=int, help="index into every (family, shard) pair")
    r.add_argument("--jobs", type=int, default=1)
    r.set_defaults(func=cmd_run)
    t = sub.add_parser("tasks")
    t.set_defaults(func=cmd_tasks)
    o = sub.add_parser("compare")
    o.add_argument("--other", required=True, help="another data root, e.g. data")
    o.set_defaults(func=cmd_compare)
    m = sub.add_parser("merge")
    m.add_argument("--family", choices=list(FAMILIES))
    m.set_defaults(func=cmd_merge)
    c = sub.add_parser("check")
    c.add_argument("--thetas", type=int, default=6)
    c.add_argument("--patterns", type=int, default=400)
    c.add_argument("--jobs", type=int, default=1)
    c.set_defaults(func=cmd_check)
    args = p.parse_args()
    args.func(Config.load(), args)


if __name__ == "__main__":
    main()
