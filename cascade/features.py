#!/usr/bin/env python3
"""Stage-1 features: a short vector of classical summary-function samples per bank pattern.

    python cascade/features.py --workers 16         # every family, every bank theta
    python cascade/features.py --family thomas --thetas 20000     # train/val/test core only

Input   data/bank/<family>/{points.npz, manifest.csv}          (read only)
Output  data/cascade/features/<family>.npz
            case_id   (P,)       rows in manifest order, first 2*thetas patterns. The default
                                 is the whole bank, as the PH arm trains on: thetas past the
                                 test block (>= 20000) all go to train.
            X         (P, D)     float32
            names     (D,)       column names, e.g. L_fixed@0.0625, G_sqrtn@1.0, log_n

Why its own features rather than data/classical/: those curves predate the Sep 22 bank
regeneration (they cover thetas < 10000 of the OLD bank and no longer match data/bank), and they
are 4 x 512 samples, which is far more than a 3-way regime call needs.

Two grids, because the two departures from CSR live at different scales:
  L on the fixed axis (r up to 1/4 of the window)  -- large-scale clustering (LGCP, nested parents)
  L, F, G, J on the sqrt(n) axis (u = r sqrt(n) <= 2) -- small-scale repulsion and tight clusters
Each is subsampled at SAMPLES log-spaced grid positions. Resumable: an existing output is skipped
unless it covers fewer thetas than asked for.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.classical import fgj                                    # noqa: E402
from cloudforger.classical.functions import GRID_SIZE, axis, radii      # noqa: E402
from cloudforger.classical.lfunction import l_minus_r                   # noqa: E402
from cloudforger.simulation.bank import DATA as BANK, Config as BankConfig  # noqa: E402

OUT = ROOT / "data" / "cascade" / "features"
FAMILIES = ("poisson", "thomas", "nested", "lgcp", "matern2")
FIXED, SQRTN = {"name": "fixed"}, {"name": "sqrtn", "u_max": 2.0}
SAMPLES = 24
IDX = np.unique(np.geomspace(1, GRID_SIZE, SAMPLES).round().astype(int) - 1)


def names() -> list[str]:
    fx, sq = axis(FIXED)[IDX], axis(SQRTN)[IDX]
    cols = [f"L_fixed@{r:.4g}" for r in fx]
    cols += [f"{f}_sqrtn@{u:.3g}" for f in ("L", "F", "G", "J") for u in sq]
    return cols + ["log_n"]


def featurize(points: np.ndarray) -> np.ndarray:
    n = len(points)
    r_sq = radii(SQRTN, n)
    f = fgj.f_function(points, r_sq)
    g = fgj.g_function(points, r_sq)
    parts = [l_minus_r(points, radii(FIXED, n)) * np.sqrt(n),   # sqrt(n): less n-dependent
             l_minus_r(points, r_sq) * np.sqrt(n),
             f, g, fgj.j_function(f, g)]
    return np.concatenate([p[IDX] for p in parts] + [[np.log(n)]])


_POINTS = _OFFSETS = None


def _rows(bounds: tuple[int, int]) -> np.ndarray:
    lo, hi = bounds
    return np.stack([featurize(_POINTS[_OFFSETS[i]:_OFFSETS[i + 1]]) for i in range(lo, hi)])


def run(family: str, thetas: int, workers: int) -> Path:
    global _POINTS, _OFFSETS
    path = OUT / f"{family}.npz"
    if path.exists() and len(np.load(path)["case_id"]) >= 2 * thetas:
        return path
    manifest = pd.read_csv(BANK / family / "manifest.csv")
    keep = manifest.index[manifest.theta < thetas].to_numpy()
    if not np.array_equal(keep, np.arange(len(keep))):
        raise SystemExit(f"{family}: manifest is not theta-sorted")
    z = np.load(BANK / family / "points.npz")
    _POINTS, _OFFSETS = z["points"], z["offsets"]

    chunks = [(lo, min(lo + 500, len(keep))) for lo in range(0, len(keep), 500)]
    with mp.get_context("fork").Pool(workers) as pool:
        X = np.concatenate(pool.map(_rows, chunks)).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, case_id=manifest.case_id.to_numpy(str)[keep], X=X, names=np.array(names()))
    tmp.replace(path)
    return path


def load(families=FAMILIES) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(case_id, X, names) over the given families, concatenated in order."""
    zs = [np.load(OUT / f"{f}.npz") for f in families]
    return (np.concatenate([z["case_id"] for z in zs]), np.concatenate([z["X"] for z in zs]),
            list(zs[0]["names"]))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", help="default: all five")
    p.add_argument("--thetas", type=int, default=BankConfig.load().thetas, help="first N thetas (default: all)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    for family in [args.family] if args.family else FAMILIES:
        t0 = time.time()
        path = run(family, args.thetas, args.workers)
        print(f"{family:8s} {time.time() - t0:6.1f}s  -> {path}", flush=True)


if __name__ == "__main__":
    main()
