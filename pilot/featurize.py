#!/usr/bin/env python3
"""Features for every pilot row: classical summary functions and vectorised persistence diagrams.

    python pilot/featurize.py new --family ring --workers 16   # classical + diagrams of one new family
    python pilot/featurize.py assemble                         # every family -> data/pilot/features/

`new` computes what the bank already has for its own families: cascade/features.py's 111 classical
columns, and diagrams with the bank's exact filtration specs (configs/featurization/config.yaml).
`assemble` takes the bank families' pilot rows from data/cascade/features and data/featurization,
the new families' from `new`, and vectorises every diagram with cloudforger.vectorization.summaries
(Betti-curve grid fitted on a sample of train rows pooled over families).

Output  data/pilot/computed/<family>/{classical,<tag>}.npz     (new families; diagrams in data/featurization's format)
        data/pilot/features/<set>/<family>.npz                 case_id, X, names; set = classical | ph_<tag>
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one core per worker
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

from shared import DATA, ROOT, families, load_config, rows, source

import features as classical                                            # cascade/features.py
from cloudforger.featurization.filtrations import FILTRATIONS           # noqa: E402
from cloudforger.featurization.sweep import Config as FeatConfig        # noqa: E402
from cloudforger.vectorization.summaries import fit_grid, vector, vector_names   # noqa: E402

_POINTS = _OFFSETS = _SPEC = None


# ------------------------------------------------------------------------------- new families

def _classical(i: int) -> np.ndarray:
    return classical.featurize(_POINTS[_OFFSETS[i]:_OFFSETS[i + 1]])


def _diagram(i: int) -> dict:
    spec = {k: v for k, v in _SPEC.items() if k != "name"}
    return FILTRATIONS[_SPEC["name"]](_POINTS[_OFFSETS[i]:_OFFSETS[i + 1]], maxdim=1, **spec)


def cmd_new(cfg: dict, family: str, workers: int) -> None:
    global _POINTS, _OFFSETS, _SPEC
    manifest = pd.read_csv(source(cfg, family) / "manifest.csv")
    z = np.load(source(cfg, family) / "points.npz")
    _POINTS, _OFFSETS = z["points"], z["offsets"]
    idx = range(len(manifest))
    out = DATA / "computed" / family / "classical.npz"
    if not out.exists():
        with mp.get_context("fork").Pool(workers) as pool:
            X = np.stack(pool.map(_classical, idx, chunksize=64)).astype(np.float32)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, case_id=manifest.case_id.to_numpy(str), X=X, names=np.array(classical.names()))
        print(f"{family}: classical {X.shape}", flush=True)
    fcfg = FeatConfig.load()
    for tag in cfg["filtrations"]:
        out = DATA / "computed" / family / f"{tag}.npz"
        if out.exists():
            continue
        _SPEC = fcfg.spec(tag)
        with mp.get_context("fork").Pool(workers) as pool:
            dgms = pool.map(_diagram, idx, chunksize=16)
        packed = {"case_id": manifest.case_id.to_numpy(str), "spec": np.array(json.dumps(_SPEC))}
        for d in (0, 1):
            packed[f"h{d}"] = np.concatenate([g[d] for g in dgms]).reshape(-1, 2)
            packed[f"h{d}_offsets"] = np.concatenate([[0], np.cumsum([len(g[d]) for g in dgms])])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **packed)
        print(f"{family}: {tag} diagrams", flush=True)


# ----------------------------------------------------------------------------------- assemble

def diagrams(cfg: dict, family: str, tag: str, case_id: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """(h0, h1) per case_id, from the bank's merged diagrams or the pilot's."""
    path = (ROOT / "data" / "featurization" / family / tag / "diagrams.npz" if family in cfg["bank_families"]
            else DATA / "computed" / family / f"{tag}.npz")
    z = np.load(path)
    at = pd.Index(z["case_id"]).get_indexer(case_id)
    if (at < 0).any():
        raise SystemExit(f"{family}/{tag}: {(at < 0).sum()} rows without a diagram")
    h, o = [z["h0"], z["h1"]], [z["h0_offsets"], z["h1_offsets"]]
    return [tuple(h[d][o[d][i]:o[d][i + 1]].copy() for d in (0, 1)) for i in at]   # copies: frees the file


def cmd_assemble(cfg: dict) -> None:
    table = rows(cfg)
    rng = np.random.default_rng(cfg["seed"])
    # classical: bank families from cascade's feature files, new ones from `new`
    for f in families(cfg):
        ids = table.index[table.family == f].to_numpy(str)
        path = (ROOT / "data" / "cascade" / "features" / f"{f}.npz" if f in cfg["bank_families"]
                else DATA / "computed" / f / "classical.npz")
        z = np.load(path)
        at = pd.Index(z["case_id"]).get_indexer(ids)
        if (at < 0).any():
            raise SystemExit(f"{f}: {(at < 0).sum()} rows without classical features")
        out = DATA / "features" / "classical" / f"{f}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, case_id=ids, X=z["X"][at], names=z["names"])
    for tag in cfg["filtrations"]:
        dg = {f: diagrams(cfg, f, tag, table.index[table.family == f].to_numpy(str)) for f in families(cfg)}
        grids = []
        for d in (0, 1):
            vals = []
            for f, lst in dg.items():
                g = table[table.family == f]
                train = np.flatnonzero(g.split.to_numpy() == "train")
                for i in rng.choice(train, min(1000, len(train)), replace=False):
                    vals.append(lst[i][d].ravel() * np.sqrt(g.n.iloc[i]))
            grids.append(fit_grid(np.concatenate(vals)))
        for f, lst in dg.items():
            n = table.n[table.family == f].to_numpy()
            X = np.stack([vector(h0, h1, k, grids) for (h0, h1), k in zip(lst, n)]).astype(np.float32)
            out = DATA / "features" / f"ph_{tag}" / f"{f}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, case_id=table.index[table.family == f].to_numpy(str), X=X,
                     names=np.array(vector_names(tag)), grids=np.stack(grids))
            print(f"{tag:16s} {f:8s} {X.shape}", flush=True)


def main(argv=None) -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--family", choices=cfg["new_families"], required=True)
    n.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    sub.add_parser("assemble")
    args = p.parse_args(argv)
    if args.cmd == "new":
        cmd_new(cfg, args.family, args.workers)
    else:
        cmd_assemble(cfg)


if __name__ == "__main__":
    main()
