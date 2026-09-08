#!/usr/bin/env python3
# scripts/collect_fusion_results.py
"""Aggregate the PH + L(r) fusion sweep and pair it against vihrs.

Reads the fusion__{Loff,raw8,pca2,pca3,pca4} run-tags written by
slurm/fusion_*.sh plus the vihrs baseline, and reports, per process:

  * mean +/- sd per arm (test loss for params, test accuracy for classify)
  * paired Wilcoxon of every L arm vs the PH-only arm (does L help?)
  * paired Wilcoxon of every arm vs vihrs (do we match/beat the baseline?)
  * a seed-collapse flag: arms whose spread is dominated by outlier seeds,
    which is how the raw collinear L columns failed on classification
    (3 seeds ~0.88, 2 seeds ~0.80) rather than degrading smoothly

Pairing is by seed. Arms with no overlapping seeds are reported unpaired and
labelled as such -- never silently compared as two means.

Usage:
    python scripts/collect_fusion_results.py
    python scripts/collect_fusion_results.py --process thomas --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.paths import DEFAULT_RESULTS_ROOT  # noqa: E402

ARMS = ["Loff", "raw8", "pca2", "pca3", "pca4"]
ARM_NOTE = {
    "Loff": "PH only",
    "raw8": "8 raw L cols (collinear)",
    "pca2": "2 PCA comps (95.4% var)",
    "pca3": "3 PCA comps (98.7% var)",
    "pca4": "4 PCA comps (99.5% var)",
}
# vihrs is filtration-independent -> the RAW tag; the classification run was
# registered under a different method dir than the parameter-estimation ones.
VIHRS_DIRS = {
    "thomas": "raw/vihrs_checkpointed",
    "nested_thomas": "raw/vihrs_checkpointed",
    "classification": "raw/vihrs",
}


def _load(base: Path) -> dict[int, dict]:
    """{seed -> results.json} for every seed_*/ directory under `base`."""
    out = {}
    if not base.is_dir():
        return out
    for jf in base.glob("seed_*/results.json"):
        try:
            seed = int(jf.parent.name.split("seed_")[1])
        except (IndexError, ValueError):
            continue
        out[seed] = json.loads(jf.read_text())
    return out


def load_process(results_root: Path, process: str) -> tuple[dict[str, dict[int, dict]], str]:
    clf = process == "classification"
    key = "test_accuracy" if clf else "test_loss"
    arms: dict[str, dict[int, dict]] = {}
    for arm in ARMS:
        d = results_root / process / "dtm_k5" / "pi_multik" / "_runs" / f"fusion__{arm}"
        arms[arm] = _load(d)
    arms["vihrs"] = _load(results_root / process / VIHRS_DIRS[process])
    return arms, key


def paired(a: dict[int, dict], b: dict[int, dict], key: str) -> tuple[np.ndarray, list[int]]:
    common = sorted(set(a) & set(b))
    return np.array([b[s][key] - a[s][key] for s in common], float), common


def collapse_flag(values: np.ndarray) -> str:
    """Flag an arm whose spread looks bimodal rather than smooth: any seed
    more than 3 robust-sigma (via MAD) from the median."""
    # n>=6: below that a MAD-based z is oversensitive on tightly-clustered
    # values and flags ordinary seed noise (it fired on vihrs at n=5).
    if len(values) < 6:
        return ""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad <= 0:
        return ""
    z = np.abs(values - med) / (1.4826 * mad)
    n_out = int((z > 3).sum())
    return f"  <-- {n_out}/{len(values)} outlier seed(s), possible collapse" if n_out else ""


def report(results_root: Path, processes: list[str]) -> dict:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    summary: dict = {}

    for process in processes:
        arms, key = load_process(results_root, process)
        clf = process == "classification"
        better_is = "higher" if clf else "lower"
        print(f"\n{'=' * 78}\n{process.upper()}   metric={key} ({better_is} is better)\n{'=' * 78}")

        summary[process] = {"metric": key, "arms": {}}
        for name in ARMS + ["vihrs"]:
            d = arms[name]
            if not d:
                print(f"  {name:6s} --"); continue
            v = np.array([d[s][key] for s in sorted(d)], float)
            note = ARM_NOTE.get(name, "baseline (radial L(r)-r CNN)")
            print(f"  {name:6s} {v.mean():.4f} ± {v.std():.4f}  n={len(v):2d}   {note}{collapse_flag(v)}")
            summary[process]["arms"][name] = {"mean": float(v.mean()), "sd": float(v.std()),
                                              "n": len(v), "seeds": sorted(d)}

        for ref in ("Loff", "vihrs"):
            if not arms.get(ref):
                continue
            label = "vs PH-only (does L help?)" if ref == "Loff" else "vs vihrs (do we match it?)"
            print(f"\n  --- {label} ---")
            for name in ARMS + ["vihrs"]:
                if name == ref or not arms.get(name):
                    continue
                diff, common = paired(arms[ref], arms[name], key)
                if len(common) == 0:
                    print(f"    {name:6s} no overlapping seeds -- NOT comparable"); continue
                improved = (diff.mean() > 0) if clf else (diff.mean() < 0)
                verdict = "better" if improved else "worse"
                if wilcoxon is not None and len(common) >= 6 and np.any(diff):
                    p = float(wilcoxon(diff).pvalue)
                    ptxt = f"p={p:.3f}" + ("  *" if p < 0.05 else "")
                else:
                    p, ptxt = None, f"(n={len(common)}<6: screen only)"
                print(f"    {name:6s} Δ={diff.mean():+.4f} [{verdict}]  paired n={len(common)}  {ptxt}")
                summary[process].setdefault("comparisons", {})[f"{name}_vs_{ref}"] = {
                    "delta": float(diff.mean()), "n": len(common), "p": p, "better": bool(improved)}
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--process", action="append", default=[],
                   help="process name; repeatable (default: all three)")
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--json", type=Path, default=None, help="also write the summary here")
    args = p.parse_args(argv)

    processes = args.process or ["thomas", "nested_thomas", "classification"]
    summary = report(args.results_root, processes)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
