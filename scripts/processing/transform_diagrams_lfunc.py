#!/usr/bin/env python3
# scripts/processing/transform_diagrams_lfunc.py
"""Build L-reparameterized diagram bundles from ALREADY-COMPUTED ones.

(b, d) -> (L(b), L(d)), where L is the cloud's own empirical Ripley L --
see data_generation/filtration/lfunc.py for why this is a genuine filtration
reparameterization rather than a post-hoc edit.

The point of this script is that you do NOT need to recompute persistence.
data/<process>/<tag>/diagrams.pkl already exists for every process, and the
L curves are already cached by the vihrs baseline
(data/<process>/*.lfunc_cache.npz). Transforming is a per-point interpolation:
seconds, not hours. The from-scratch path (REGISTRY name l_dtm / l_rips) exists
too and produces identical output; use that for data generated in future.

Usage (from the repo root):
    python scripts/processing/transform_diagrams_lfunc.py --process thomas --tag dtm_k5
    python scripts/processing/transform_diagrams_lfunc.py --process classification --tag dtm_k5
    python scripts/processing/transform_diagrams_lfunc.py --all-processes --tag dtm_k5

Writes data/<process>/l_<tag>/{,adversarial_}diagrams.pkl with the same bundle
schema (diagrams/labels/label_names/seeds/params/process + any extra keys the
source carried), so every downstream consumer -- load_multik_split, the
per-seed calibrated imager, pi_multik -- works unchanged.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.core.io import load_pickle  # noqa: E402
from cloudforger.data_generation.filtration.lfunc import apply_l_transform, monotone_l  # noqa: E402
from cloudforger.paths import DEFAULT_DATA_ROOT  # noqa: E402

# Cache basenames the vihrs baseline writes. The parameter-estimation path uses
# "<stem>.lfunc_cache.npz"; the classification path uses
# "<stem>.lfunc_classify_cache.npz" (see baselines/vihrs.py).
CACHE_SUFFIXES = (".lfunc_cache.npz", ".lfunc_classify_cache.npz")


def find_lfunc_cache(clouds_path: Path) -> Path:
    for suffix in CACHE_SUFFIXES:
        candidate = clouds_path.parent / (clouds_path.stem + suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No L cache for {clouds_path}. Looked for "
        + ", ".join(clouds_path.stem + s for s in CACHE_SUFFIXES)
        + ". Build it by running the vihrs baseline on this process once "
          "(it caches L(r)-r as a side effect), or use the from-scratch "
          "l_dtm/l_rips filtration instead of this fast path."
    )


def load_l_curves(cache_path: Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """{seed -> monotone L on r_grid}, r_grid. The cache stores L - r (which is
    NOT monotone), so r is added back before enforcing monotonicity."""
    z = np.load(cache_path)
    r_grid = np.asarray(z["r_grid"], dtype=np.float64)
    l_minus_r = np.asarray(z["l_minus_r"], dtype=np.float64)
    seeds = np.asarray(z["cloud_seeds"], dtype=np.int64)
    curves = {int(s): monotone_l(l_minus_r[i] + r_grid) for i, s in enumerate(seeds)}
    return curves, r_grid


def transform_bundle(bundle: dict, curves: dict[int, np.ndarray], r_grid: np.ndarray, tag: str) -> dict:
    seeds = [int(s) for s in bundle["seeds"]]
    missing = [s for s in seeds if s not in curves]
    if missing:
        raise KeyError(
            f"[{tag}] {len(missing)}/{len(seeds)} diagram seeds have no cached L curve "
            f"(first few: {missing[:5]}). The L cache and the diagram bundle disagree on "
            "the seed convention -- do not transform against a mismatched join."
        )

    n_over = n_tot = 0
    r_max = float(r_grid[-1])
    out_records = []
    for record, seed in zip(bundle["diagrams"], seeds):
        l_curve = curves[seed]
        new_dgms = {}
        for dim, pairs in record["diagrams"].items():
            arr = np.asarray(pairs, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            n_tot += finite.size
            n_over += int((finite > r_max).sum())
            new_dgms[int(dim)] = apply_l_transform(arr, l_curve, r_grid)
        out_records.append({
            **record,
            "diagrams": new_dgms,
            "filtration": f"l_{record.get('filtration', '')}",
            "filtration_params": {
                **dict(record.get("filtration_params", {})),
                "l_reparameterized": True,
                "r_max": r_max,
                "n_r": int(len(r_grid)),
                "tail": "slope1",
            },
        })

    pct = 100.0 * n_over / n_tot if n_tot else 0.0
    print(f"  [{tag}] {len(out_records)} diagrams transformed | "
          f"{n_over}/{n_tot} ({pct:.2f}%) finite values above r_max={r_max} used the slope-1 tail")
    return {**bundle, "diagrams": out_records}


def process_one(data_root: Path, process: str, tag: str, force: bool) -> None:
    proc_dir = data_root / process
    src_dir, dst_dir = proc_dir / tag, proc_dir / f"l_{tag}"
    if not src_dir.exists():
        print(f"[{process}] no {src_dir} -- skipping"); return
    dst_dir.mkdir(parents=True, exist_ok=True)

    for clouds_name, dgm_name in (("clouds.pkl", "diagrams.pkl"),
                                  ("adversarial_clouds.pkl", "adversarial_diagrams.pkl")):
        src, dst = src_dir / dgm_name, dst_dir / dgm_name
        if not src.exists():
            print(f"[{process}] no {src} -- skipping"); continue
        if dst.exists() and not force:
            print(f"[{process}] {dst} exists -- skipping (use --force)"); continue

        curves, r_grid = load_l_curves(find_lfunc_cache(proc_dir / clouds_name))
        out = transform_bundle(load_pickle(src), curves, r_grid, tag=f"{process}/{dgm_name}")

        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(dst)  # atomic, so a concurrent reader never sees a half-written bundle
        print(f"[{process}] wrote {dst}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--process", action="append", default=[], help="process name; repeatable")
    p.add_argument("--all-processes", action="store_true", help="every directory under the data root")
    p.add_argument("--tag", default="dtm_k5", help="source filtration path tag (default: dtm_k5)")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--force", action="store_true", help="overwrite existing l_<tag> bundles")
    args = p.parse_args(argv)

    processes = args.process
    if args.all_processes:
        processes = sorted(d.name for d in args.data_root.iterdir() if d.is_dir())
    if not processes:
        raise SystemExit("Pass --process <name> (repeatable) or --all-processes.")

    for proc in processes:
        process_one(args.data_root, proc, args.tag, args.force)


if __name__ == "__main__":
    main()
