#!/usr/bin/env python3
"""Simulate the pilot families, one shard of thetas at a time, on the `family_pilot` streams.

    python pilot/generate.py run --family ring --shard 3      # one shard (one SLURM array task)
    python pilot/generate.py merge                            # -> data/pilot/bank/<family>/{points.npz, manifest.csv}

The bank's own machinery (draw_theta, sample_patterns, the rules in configs/simulation/config.yaml,
the bank's nbar and n ranges) with the stream set swapped, so a pilot theta is drawn exactly as a
bank theta would be and never collides with one. Output layout matches data/bank/<family>/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math

import numpy as np
import pandas as pd

from shared import DATA, load_config

from cloudforger.departure.tables import Tables                          # noqa: E402
from cloudforger.simulation.bank import Config, draw_theta, sample_patterns  # noqa: E402
from cloudforger.simulation.families import FAMILIES, Rules              # noqa: E402


def bank_config(cfg: dict) -> Config:
    return dataclasses.replace(Config.load(), thetas=cfg["thetas"], shard_size=cfg["shard_size"])


def run_shard(family: str, shard: int, cfg: dict):
    path = DATA / "shards" / family / f"shard_{shard:04d}.npz"
    if path.exists():
        return path
    bcfg, tables, fam = bank_config(cfg), Tables(), FAMILIES[family](Rules.load())
    points, rows = [], []
    for index in range(shard * bcfg.shard_size, min((shard + 1) * bcfg.shard_size, bcfg.thetas)):
        theta = draw_theta(fam, tables, bcfg, index, set_=cfg["set"])
        for rep, pts, tries in sample_patterns(family, theta, bcfg, index, set_=cfg["set"]):
            points.append(pts)
            rows.append({"family": family, "theta": index, "rep": rep, "nbar": theta["nbar"],
                         "cv": theta["cv"], "draw_tries": theta["draw_tries"], **theta["model"],
                         "n": len(pts), "pattern_tries": tries})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, points=np.concatenate(points), sizes=np.array([len(p) for p in points]),
             manifest=np.array(json.dumps(rows)))
    tmp.replace(path)
    return path


def merge(family: str, cfg: dict) -> None:
    bcfg = bank_config(cfg)
    paths = sorted((DATA / "shards" / family).glob("shard_*.npz"))
    expected = math.ceil(bcfg.thetas / bcfg.shard_size)
    if len(paths) != expected:
        raise SystemExit(f"{family}: {len(paths)} of {expected} shards")
    parts = [np.load(p) for p in paths]
    manifest = pd.DataFrame([row for z in parts for row in json.loads(str(z["manifest"]))])
    manifest.insert(0, "case_id", [f"{family}-{t:05d}-{r}" for t, r in zip(manifest.theta, manifest.rep)])
    sizes = np.concatenate([z["sizes"] for z in parts])
    if not (np.array_equal(sizes, manifest.n) and manifest.case_id.is_unique
            and len(manifest) == bcfg.thetas * bcfg.reps and manifest.theta.is_monotonic_increasing):
        raise SystemExit(f"{family}: shards inconsistent")
    out = DATA / "bank" / family
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "points.npz", points=np.concatenate([z["points"] for z in parts]),
             offsets=np.concatenate([[0], np.cumsum(sizes)]))
    manifest.to_csv(out / "manifest.csv", index=False)
    print(f"{family:8s} {len(manifest)} patterns, n {manifest.n.min()}-{manifest.n.max()}, "
          f"max tries: draw {manifest.draw_tries.max()}, pattern {manifest.pattern_tries.max()}", flush=True)


def main(argv=None) -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--family", choices=cfg["new_families"], required=True)
    r.add_argument("--shard", type=int, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--family", choices=cfg["new_families"])
    args = p.parse_args(argv)
    if args.cmd == "run":
        print(run_shard(args.family, args.shard, cfg), flush=True)
    else:
        for f in [args.family] if args.family else cfg["new_families"]:
            merge(f, cfg)


if __name__ == "__main__":
    main()
