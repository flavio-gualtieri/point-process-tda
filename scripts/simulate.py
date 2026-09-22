"""Simulate the cloud bank.

    python scripts/simulate.py run --jobs 10                       # every shard of every family (resumable)
    python scripts/simulate.py run --family lgcp --shard 3         # one shard (e.g. one SLURM array task)
    python scripts/simulate.py merge --family thomas               # -> data/bank/<family>/{points.npz, manifest.csv}
    python scripts/simulate.py check --jobs 10                     # samplers vs closed-form K
"""

from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from cloudforger.departure.tables import Tables
from cloudforger.simulation.check import check_theta
from cloudforger.simulation.families import FAMILIES
from cloudforger.simulation.bank import DATA, Config, run_shard

ORDER = ("poisson", "thomas", "nested", "matern2", "lgcp")


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


def cmd_run(cfg, args):
    families = [args.family] if args.family else ORDER
    shards = range(math.ceil(cfg.thetas / cfg.shard_size))
    tasks = [(f, s, cfg) for f in families for s in ([args.shard] if args.shard is not None else shards)]
    for fam, shard, secs in _pool(_shard, tasks, args.jobs):
        print(f"{fam:8s} shard {shard:3d}  {secs:6.1f}s", flush=True)


def cmd_merge(cfg, args):
    for fam in ([args.family] if args.family else ORDER):
        paths = sorted((DATA / "shards" / fam).glob("shard_*.npz"))
        expected = math.ceil(cfg.thetas / cfg.shard_size)
        if len(paths) != expected:
            raise SystemExit(f"{fam}: {len(paths)} of {expected} shards")
        parts = [np.load(p) for p in paths]
        manifest = pd.DataFrame([row for z in parts for row in json.loads(str(z["manifest"]))])
        manifest.insert(0, "case_id", [f"{fam}-{t:05d}-{r}" for t, r in zip(manifest.theta, manifest.rep)])
        sizes = np.concatenate([z["sizes"] for z in parts])
        if not (np.array_equal(sizes, manifest.n) and manifest.case_id.is_unique and len(manifest) == cfg.thetas * cfg.reps):
            raise SystemExit(f"{fam}: shards inconsistent")
        out = DATA / fam
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "points.npz", points=np.concatenate([z["points"] for z in parts]),
                 offsets=np.concatenate([[0], np.cumsum(sizes)]))
        manifest.to_csv(out / "manifest.csv", index=False)
        print(f"{fam:8s} {len(manifest)} patterns, n {manifest.n.min()}-{manifest.n.max()}, "
              f"max tries: draw {manifest.draw_tries.max()}, pattern {manifest.pattern_tries.max()}", flush=True)


def _check(args):
    return check_theta(*args, Tables())


def cmd_check(cfg, args):
    tasks = [(f, i, args.patterns, cfg) for f in ORDER for i in range(args.thetas)]
    rows = sorted(_pool(_check, tasks, args.jobs), key=lambda r: (ORDER.index(r["family"]), r["theta"]))
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--family", choices=list(FAMILIES))
    r.add_argument("--shard", type=int)
    r.add_argument("--jobs", type=int, default=1)
    r.set_defaults(func=cmd_run)
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
