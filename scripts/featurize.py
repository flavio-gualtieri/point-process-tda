"""Persistence diagrams for the simulated patterns (configs/featurization/config.yaml).

    python scripts/featurize.py run --jobs 10                                # every shard (resumable)
    python scripts/featurize.py run --family thomas --tag dtm_k5 --shard 3   # one shard (e.g. one SLURM array task)
    python scripts/featurize.py merge                                        # -> data/featurization/<family>/<tag>/diagrams.npz
"""

from __future__ import annotations

import argparse
import os
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one core per worker
    os.environ.setdefault(_var, "1")

from multiprocessing import get_context

from cloudforger.featurization.filtrations import tag
from cloudforger.featurization.sweep import Config, families, merge, n_shards, run_shard


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


def _selection(cfg, args):
    fams = [args.family] if args.family else families()
    tags = [args.tag] if args.tag else [tag(s) for s in cfg.filtrations]
    return [(f, t) for f in fams for t in tags]


def cmd_run(cfg, args):
    tasks = [(f, t, s, cfg) for f, t in _selection(cfg, args)
             for s in ([args.shard] if args.shard is not None else range(n_shards(f, cfg)))]
    for (fam, tag_, shard), secs in _pool(_shard, tasks, args.jobs):
        print(f"{fam:8s} {tag_:16s} shard {shard:3d}  {secs:6.1f}s", flush=True)


def cmd_merge(cfg, args):
    for fam, tag_ in _selection(cfg, args):
        print(f"{fam:8s} {tag_:16s} -> {merge(fam, tag_, cfg)}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    for name, func in (("run", cmd_run), ("merge", cmd_merge)):
        s = sub.add_parser(name)
        s.add_argument("--family")
        s.add_argument("--tag")
        s.set_defaults(func=func)
        if name == "run":
            s.add_argument("--shard", type=int)
            s.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()
    args.func(Config.load(), args)


if __name__ == "__main__":
    main()
