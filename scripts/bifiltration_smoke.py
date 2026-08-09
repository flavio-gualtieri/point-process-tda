#!/usr/bin/env python3
# scripts/check_bifiltration_smoke.py
"""Evaluate a bifiltration smoke run BEFORE committing to the full featurization.

Run scripts/featurize_bifiltration.py with --limit first, then point this at the
same config. Every check prints PASS / WARN / FAIL and says what to do about it.

The last check is the one that decides whether the full run is worth the hours:
it fits a cheap ridge from the images to the labels and reports per-target R^2.
If the coarse targets show nothing at 500 clouds, a CNN on 8000 will struggle
too, and it is much cheaper to learn that now.

Usage:
    python scripts/check_bifiltration_smoke.py configs/runs/nested_thomas/nested_thomas_mph.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import load_config
from cloudforger.core.io import load_pickle
from cloudforger.core.records import load_signed_measures
from cloudforger.filtration import BIFILTRATION_REGISTRY
from cloudforger.paths import DEFAULT_DATA_ROOT, DataPaths

FULL_N = 8000  # what the full run will process, for projecting cost


def _verdict(ok: bool, warn: bool = False) -> str:
    return "WARN" if warn else ("PASS" if ok else "FAIL")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--full-n", type=int, default=FULL_N)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    dp = DataPaths(cfg.process.name, root=cfg.data_root or DEFAULT_DATA_ROOT)
    bcfg = cfg.bifiltration[0]
    bifil = BIFILTRATION_REGISTRY.build(bcfg.name, **bcfg.params)

    measures, bundle = load_signed_measures(dp.signed_measures([bifil]))
    imgs_bundle = load_pickle(dp.feature([bifil], "mph_image"))
    images = np.asarray(imgs_bundle["image_tensors"])
    labels = np.asarray(bundle["labels"], float)
    names = list(bundle["label_names"])
    grid = bundle["grid"]
    dims = tuple(bifil.params["homology_dims"])
    n = len(measures)

    print(f"{n} clouds, images {images.shape}, degrees {dims}")
    print(f"grid: axis0 {grid[0][0]:.4f}..{grid[0][-1]:.4f} (step {grid[0][1]-grid[0][0]:.5f}), "
          f"axis1 {grid[1][0]:.4f}..{grid[1][-1]:.4f}\n")

    # ---------------------------------------------------------------- 1. grid
    print("[1] GRID SIZING")
    step = float(grid[0][1] - grid[0][0])
    if "cluster_scale" in names:
        smallest = labels[:, names.index("cluster_scale")].min()
        ok = step <= smallest
        print(f"    grid step {step:.5f} vs smallest cluster_scale {smallest:.5f} "
              f"-> {_verdict(ok)}")
        if not ok:
            print("    FIX: raise GRID_RESOLUTION_MAX, or raise c2's lower bound in the sweep.")
        print(f"    NOTE: the full {args.full_n}-cloud draw will contain smaller values than "
              f"these {n};\n          expect the auto-sizer to ask for more.")

    # ------------------------------------------------------- 2. radius clipping
    print("\n[2] RADIUS CUTOFF")
    top = float(grid[0][-1])
    # H0 always has essential classes, which multipers parks at the grid top.
    # So "max radius == grid top" is EXPECTED and proves nothing. What matters is
    # how much NON-essential mass piles up in the top row.
    for d in dims:
        top_frac, p99 = [], []
        for m in measures:
            a, w = m.atoms(d)
            if len(a) == 0:
                continue
            at_top = np.isclose(a[:, 0], top)
            top_frac.append(np.abs(w[at_top]).sum() / max(np.abs(w).sum(), 1e-12))
            interior = a[~at_top, 0]
            p99.append(np.percentile(interior, 99) if len(interior) else 0.0)
        tf, q = float(np.mean(top_frac)), float(np.mean(p99))
        warn = tf > 0.05
        print(f"    H{d}: mean |mass| in the top row {tf:.1%}, "
              f"mean 99th-pct interior radius {q:.4f} (top {top:.4f}) -> "
              f"{_verdict(not warn, warn)}")
        if warn:
            print("    FIX: raise threshold_radius; real structure is being pushed into the top row.")
        elif q < 0.5 * top:
            print("    NOTE: interior atoms stop well short of the top; threshold_radius could "
                  "be lowered,\n          which cuts required resolution and cost proportionally.")

    # ------------------------------------------------------------- 3. atoms
    print("\n[3] ATOM COUNTS")
    for d in dims:
        na = np.array([len(m.atoms(d)[0]) for m in measures])
        empty = int((na == 0).sum())
        print(f"    H{d}: median {int(np.median(na))}, min {na.min()}, max {na.max()}, "
              f"empty {empty} -> {_verdict(empty == 0)}")
        if empty:
            print("    FIX: empty measures become all-zero images the model cannot use.")

    # --------------------------------------------------------------- 4. mass
    print("\n[4] TOTAL MASS vs LABELS")
    print("    Mass is not forced to zero. If it correlates with a target it carries")
    print("    signal, and enforce_null_mass would throw that away.")
    for d in dims:
        mass = np.array([m.atoms(d)[1].sum() for m in measures], float)
        if mass.std() == 0:
            print(f"    H{d}: constant at {mass[0]:.0f} -> PASS (nothing lost either way)")
            continue
        cors = {nm: float(np.corrcoef(mass, np.log(np.abs(labels[:, i]) + 1e-12))[0, 1])
                for i, nm in enumerate(names)}
        worst = max(cors.items(), key=lambda kv: abs(kv[1]))
        print(f"    H{d}: range {mass.min():.0f}..{mass.max():.0f}; "
              f"strongest |corr| with log-labels: {worst[0]} {worst[1]:+.3f}")
        print(f"    -> {_verdict(abs(worst[1]) < 0.2, abs(worst[1]) >= 0.2)}"
              f"  (>=0.2 means mass carries signal; keep it, do not enforce null mass)")

    # ------------------------------------------------------------- 5. images
    print("\n[5] IMAGES")
    nz = float((np.abs(images) > 1e-9).mean())
    both = images.min() < 0 < images.max()
    allzero = int((np.abs(images).sum(axis=(2, 3)) == 0).sum())
    print(f"    range {images.min():.2f}..{images.max():.2f}, nonzero {nz:.0%}, "
          f"both signs {both}, all-zero images {allzero} -> "
          f"{_verdict(both and allzero == 0 and 0.01 < nz < 0.99)}")
    if nz > 0.95:
        print("    NOTE: nearly every pixel is nonzero; sigma_pixels may be too large.")
    if nz < 0.05:
        print("    NOTE: image is nearly empty; sigma_pixels may be too small.")

    # -------------------------------------------------------------- 6. timing
    print("\n[6] COST PROJECTION")
    print(f"    Take median and p99 per-cloud times from the featurize log.")
    print(f"    full run ~ median * {args.full_n} / 3600 core-hours.")
    print(f"    If p99/median > 10, a few clouds dominate: consider capping E[N] in the sweep.")

    # --------------------------------------------------------- 7. SIGNAL TEST
    print("\n[7] SIGNAL TEST  (the go/no-go check)")
    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import cross_val_predict
    except ImportError:
        print("    scikit-learn not available; skipped.")
        return

    # Coarse 8x8 pooling: a linear model on raw 64x64 would just overfit at n=500.
    k = images.shape[-1] // 8
    pooled = images.reshape(n, images.shape[1], 8, k, 8, k).mean(axis=(3, 5)).reshape(n, -1)
    X = (pooled - pooled.mean(0)) / (pooled.std(0) + 1e-9)
    print(f"    ridge on 8x8-pooled images ({X.shape[1]} features), 5-fold CV, "
          f"target = log-label z-scored")
    print(f"    {'target':<24}{'R^2':>8}{'MSE(prior units)':>20}")
    for i, nm in enumerate(names):
        y = np.log(np.abs(labels[:, i]) + 1e-12)
        y = (y - y.mean()) / (y.std() + 1e-12)
        pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 25)), X, y, cv=5)
        mse = float(np.mean((y - pred) ** 2))
        print(f"    {nm:<24}{1 - mse:>8.3f}{mse:>20.3f}")
    print("\n    MSE here is in the same units the training pipeline reports:")
    print("    1.0 = learned nothing. A linear probe is a LOWER bound on what a CNN")
    print("    can do, so treat these as 'signal is present', not as final numbers.")
    print("    Compare against the count-only baseline for this design before")
    print("    concluding anything about the coarse targets.")


if __name__ == "__main__":
    main()