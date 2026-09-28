"""Persistence diagrams for the simulated patterns (configs/featurization.yaml).

    python scripts/featurize.py run --jobs 10                                # every shard (resumable)
    python scripts/featurize.py run --family thomas --tag dtm_k5 --shard 3   # one shard
    python scripts/featurize.py run --task 7 --jobs 8                        # every shard of (family, tag) no. 7
    python scripts/featurize.py tasks                                        # how many (family, tag) pairs there are
    python scripts/featurize.py merge                                        # -> <data>/featurization/<family>/<tag>/diagrams.npz

`run` first refuses to go on if a merged diagrams.npz does not match the current bank: shards and
merges skip what exists, so diagrams left from an earlier bank would otherwise survive in silence.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one core per worker
    os.environ.setdefault(_var, "1")

from multiprocessing import get_context

import numpy as np
import pandas as pd

from cloudforger.featurization.filtrations import tag
from cloudforger.featurization.sweep import DATA, Config, families, merge, n_shards, run_shard
from cloudforger.paths import BANK


def _shard(args):
    t0 = time.time()
    run_shard(*args)
    return args[:3], time.time() - t0


def _pool(f, tasks, jobs):
    if jobs == 1:
        yield from map(f, tasks)
        return
    with get_context("spawn").Pool(jobs, maxtasksperchild=1) as pool:
        yield from pool.imap_unordered(f, tasks)


def pairs(cfg) -> list[tuple[str, str]]:
    return [(f, tag(s)) for f in families() for s in cfg.filtrations]


def _selection(cfg, args):
    if getattr(args, "task", None) is not None:
        return [pairs(cfg)[args.task]]
    return [(f, t) for f, t in pairs(cfg) if (not args.family or f == args.family) and (not args.tag or t == args.tag)]


def preflight(cfg) -> None:
    """Every merged diagrams.npz must hold exactly the bank's patterns, in manifest order. A stale
    shard has the same name and patterns as the one that replaces it, so shards go too."""
    stale = []
    for family, tag_ in pairs(cfg):
        path = DATA / family / tag_ / "diagrams.npz"
        if not path.exists():
            continue
        case_id = pd.read_csv(BANK / family / "manifest.csv").case_id.to_numpy(str)
        have = np.load(path)["case_id"]
        if len(have) != len(case_id) or (have != case_id).any():
            stale.append(path)
    if stale:
        print("diagrams that predate the current bank -- delete them and the shards/ directory:\n  "
              + "\n  ".join(map(str, stale)), file=sys.stderr)
        raise SystemExit(1)


def cmd_tasks(cfg, args):
    print(len(pairs(cfg)))


def cmd_run(cfg, args):
    preflight(cfg)
    tasks = [(f, t, s, cfg) for f, t in _selection(cfg, args)
             for s in ([args.shard] if args.shard is not None else range(n_shards(f, cfg)))]
    for (fam, tag_, shard), secs in _pool(_shard, tasks, args.jobs):
        print(f"{fam:8s} {tag_:16s} shard {shard:3d}  {secs:6.1f}s", flush=True)


def cmd_merge(cfg, args):
    for fam, tag_ in _selection(cfg, args):
        print(f"{fam:8s} {tag_:16s} -> {merge(fam, tag_, cfg)}", flush=True)
    if not (args.family or args.tag) and (DATA / "shards").exists():
        raise SystemExit(f"shards left over in {DATA / 'shards'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    for name, func in (("run", cmd_run), ("merge", cmd_merge)):
        s = sub.add_parser(name)
        s.add_argument("--family")
        s.add_argument("--tag")
        s.set_defaults(func=func)
        if name == "run":
            s.add_argument("--shard", type=int)
            s.add_argument("--task", type=int, help="index into every (family, tag) pair")
            s.add_argument("--jobs", type=int, default=1)
    sub.add_parser("tasks").set_defaults(func=cmd_tasks)
    args = p.parse_args()
    args.func(Config.load(), args)


if __name__ == "__main__":
    main()
