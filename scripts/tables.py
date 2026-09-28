#!/usr/bin/env python3
"""Feature tables for the table learners, and a check that every input a config uses is on disk.

    python scripts/tables.py classical --workers 16      # summary-function samples, every bank family
    python scripts/tables.py ph --workers 16             # PH summaries, every `source: ph` input of the config
    python scripts/tables.py check                       # every table and network input, every family

classical  cloudforger.classical.table: 111 samples of L, F, G, J per pattern, from the points.
ph         cloudforger.vectorization.summaries of <data>/featurization/<family>/<tag>/diagrams.npz. The
           Betti-curve grid is fitted once per filtration on train rows (up to 2000 per family, pooled)
           and stored in grids.npy; later families reuse it, so a table never changes when the family
           list grows.

Output  <data>/tables/classical/<family>.npz, <data>/tables/ph_<tag>/{grids.npy, <family>.npz}
        each (case_id, X, names), rows in manifest order. Existing tables are skipped.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

from cloudforger.classical import table as classical_table
from cloudforger.paths import BANK, CURVES, DIAGRAMS
from cloudforger.pipeline.core import config_arg, load_config
from cloudforger.pipeline.inputs import SOURCES, table_path
from cloudforger.simulation.split import split_of
from cloudforger.vectorization.summaries import fit_grid, vector, vector_names

_Z = _N = _GRIDS = None


def _diagrams(family: str, tag: str):
    path = DIAGRAMS / family / tag / "diagrams.npz"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run scripts/featurize.py first")
    z = np.load(path)
    return {k: z[k] for k in ("case_id", "h0", "h0_offsets", "h1", "h1_offsets")}


def _pairs(i: int, d: int) -> np.ndarray:
    o = _Z[f"h{d}_offsets"]
    return _Z[f"h{d}"][o[i]:o[i + 1]]


def _vector(i: int) -> np.ndarray:
    return vector(_pairs(i, 0), _pairs(i, 1), int(_N[i]), _GRIDS)


def classical(cfg: dict, workers: int) -> None:
    for family in cfg["families"]:
        t0 = time.time()
        path = classical_table.run(family, workers)
        print(f"classical        {family:8s} {time.time() - t0:6.1f}s  -> {path}", flush=True)


def ph(cfg: dict, workers: int, only: str | None) -> None:
    global _Z, _N, _GRIDS
    tags = sorted({s["filtration"] for s in cfg["inputs"].values() if s["source"] == "ph"})
    for tag in [only] if only else tags:
        grid_path = table_path(f"ph_{tag}", "grids").with_suffix(".npy")
        if grid_path.exists():
            _GRIDS = list(np.load(grid_path))
        else:
            rng, vals = np.random.default_rng(0), ([], [])
            for family in cfg["families"]:
                _Z = _diagrams(family, tag)
                m = pd.read_csv(BANK / family / "manifest.csv")
                train = np.flatnonzero(split_of(m.theta.to_numpy()) == "train")
                for i in rng.choice(train, min(2000, len(train)), replace=False):
                    for d in (0, 1):
                        vals[d].append(_pairs(i, d).ravel() * np.sqrt(m.n.iloc[i]))
            _GRIDS = [fit_grid(np.concatenate(v)) for v in vals]
            grid_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(grid_path, np.stack(_GRIDS))
        for family in cfg["families"]:
            out = table_path(f"ph_{tag}", family)
            if out.exists():
                print(f"ph_{tag:13s} {family:8s} exists", flush=True)
                continue
            m = pd.read_csv(BANK / family / "manifest.csv")
            _Z = _diagrams(family, tag)
            if not np.array_equal(_Z["case_id"], m.case_id.to_numpy(str)):
                raise SystemExit(f"{family}/{tag}: diagrams are not in manifest order")
            _N = m.n.to_numpy()
            with mp.get_context("fork").Pool(workers) as pool:
                X = np.stack(pool.map(_vector, range(len(m)), chunksize=512)).astype(np.float32)
            tmp = out.with_name(out.name + ".tmp.npz")
            np.savez(tmp, case_id=m.case_id.to_numpy(str), X=X, names=np.array(vector_names(tag)))
            tmp.replace(out)
            print(f"ph_{tag:13s} {family:8s} {X.shape}", flush=True)


def check(cfg: dict) -> None:
    missing = []
    used = {i for m in {*cfg["classify"], *cfg["estimate"]} for i in cfg["models"][m].get("inputs", [])}
    for name in sorted(used):
        spec = cfg["inputs"][name]
        for family in cfg["families"]:
            try:
                SOURCES[spec["source"]](spec, family)
            except SystemExit as e:
                missing.append(f"input {name}: {e}")
    nets = [cfg["models"][m] for m in {*cfg["classify"], *cfg["estimate"]} if cfg["models"][m]["learner"] == "nn"]
    for family in cfg["families"]:
        for s in nets:
            need = []
            if s.get("curves"):
                need += [CURVES / family / g / "curves.npz" for g in ("fixed", "sqrtn_u2")
                         if g in s["curves"] or "@" not in s["curves"]]
            if s.get("filtration"):
                need += [DIAGRAMS / family / t / "diagrams.npz" for t in s["filtration"].split(",")]
            missing += [f"network input: {p}" for p in need if not p.exists()]
    for line in dict.fromkeys(missing):
        print("MISSING", line)
    print("all inputs present" if not missing else f"{len(set(missing))} missing")
    raise SystemExit(1 if missing else 0)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config_arg(p)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("classical", "ph"):
        s = sub.add_parser(name)
        s.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    sub.choices["ph"].add_argument("--filtration", help="one tag only (default: every ph input's)")
    sub.add_parser("check")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "classical":
        classical(cfg, args.workers)
    elif args.cmd == "ph":
        ph(cfg, args.workers, args.filtration)
    else:
        check(cfg)


if __name__ == "__main__":
    main()
