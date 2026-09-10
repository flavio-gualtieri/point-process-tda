#!/usr/bin/env python3
# scripts/collect_twobranch.py
"""Read out the two-branch PH ablation (slurm/twobranch_{classify,params}.sh).

For each process and each arm, every seed has several restarts
(tb__<arm>__r<j>, init_offset j -- same train/val/test split, different
weight init / batch order). Per seed, the restart with the smallest
`best_val_loss` is kept; test metrics are only read AFTER that choice. Both
arms are selected the same way, then paired by seed:

    curves     = summary-function CNN branch only   (PH branch OFF)
    PHcurves   = the same model with the PH branch ON

so the paired difference is the value of the persistence-image branch on
top of the full-resolution L + F + G curves -- the one question the paper's
topological claim now rests on.

Selecting on validation loss (never test) is what makes restarts legitimate
rather than cherry-picking; this script refuses to fall back to test metrics
for a run that lacks best_val_loss.

Also printed per arm: how often a restart lands in a clearly worse basin
(spread of the test metric across restarts), because the training landscape
here is bimodal and the selected number alone hides that.

Usage:
    python scripts/collect_twobranch.py                         # classification
    python scripts/collect_twobranch.py --processes thomas matern_cluster nested_thomas
    python scripts/collect_twobranch.py --json twobranch.json
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

ARMS = ("curves", "PHcurves")


def _load_arm(base: Path, arm: str, n_restarts: int) -> dict[int, dict[int, dict]]:
    """{seed -> {restart -> results.json}} for one arm."""
    out: dict[int, dict[int, dict]] = {}
    for r in range(n_restarts):
        d = base / f"tb__{arm}__r{r}"
        for jf in d.glob("seed_*/results.json"):
            seed = int(jf.parent.name.split("seed_")[1])
            out.setdefault(seed, {})[r] = json.loads(jf.read_text())
    return out


def _select(runs: dict[int, dict[int, dict]]) -> dict[int, tuple[int, dict]]:
    chosen = {}
    for seed, by_r in runs.items():
        usable = {r: x for r, x in by_r.items() if x.get("best_val_loss") is not None}
        if usable:
            r = min(usable, key=lambda k: usable[k]["best_val_loss"])
            chosen[seed] = (r, usable[r])
    return chosen


def _wilcoxon(diff: np.ndarray) -> float | None:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    if len(diff) < 6 or not np.any(diff):
        return None
    return float(wilcoxon(diff).pvalue)


def report(results_root: Path, process: str, n_restarts: int) -> dict:
    base = results_root / process / "dtm_k5" / "pi_multik" / "_runs"
    is_clf = process == "classification"
    metric = "test_accuracy" if is_clf else "test_loss"
    sign = 1.0 if is_clf else -1.0            # +: PH branch is better

    print(f"\n=== {process} | metric {metric} ({'higher' if is_clf else 'lower'} is better) ===")
    arms = {a: _load_arm(base, a, n_restarts) for a in ARMS}
    chosen = {a: _select(arms[a]) for a in ARMS}
    for a in ARMS:
        if not chosen[a]:
            print(f"  {a}: no results under {base}/tb__{a}__r*")
            return {}

    for a in ARMS:
        sel = np.array([x[metric] for _, x in chosen[a].values()], float)
        spreads = [
            max(x[metric] for x in runs.values()) - min(x[metric] for x in runs.values())
            for runs in arms[a].values() if len(runs) > 1
        ]
        n_runs = sum(len(v) for v in arms[a].values())
        print(f"  {a:9s} selected {sel.mean():.4f} +/- {sel.std(ddof=1):.4f}  "
              f"(n={len(sel)} seeds, {n_runs} runs; "
              f"median restart spread {np.median(spreads) if spreads else 0:.4f}, "
              f"max {max(spreads) if spreads else 0:.4f})")

    common = sorted(set(chosen["curves"]) & set(chosen["PHcurves"]))
    a = np.array([chosen["curves"][s][1][metric] for s in common], float)
    b = np.array([chosen["PHcurves"][s][1][metric] for s in common], float)
    gain = sign * (b - a)                       # > 0 means the PH branch helps
    p = _wilcoxon(b - a)
    print(f"  PH branch effect: {gain.mean():+.4f}  PH better on {int((gain > 0).sum())}/{len(common)} seeds"
          + (f"  Wilcoxon p = {p:.4f}" if p is not None else "  (n < 6 or no scipy: no test)"))
    print(f"  {'seed':>6}  {'curves':>8}  {'PHcurves':>8}  {'gain':>8}  picked(r)")
    for s, x, y, g in zip(common, a, b, gain):
        print(f"  {s:>6}  {x:>8.4f}  {y:>8.4f}  {g:>+8.4f}  "
              f"{chosen['curves'][s][0]}/{chosen['PHcurves'][s][0]}")

    if not is_clf:
        # Where does the PH branch help or hurt? Per-target means of the
        # selected runs -- the density/count vs scale split is the mechanism
        # question (F and G reproduced PH's density/count advantage before).
        targets = list(chosen["curves"][common[0]][1].get("test_loss_per_target") or {})
        if targets:
            print(f"  {'target':>22}  {'curves':>8}  {'PHcurves':>8}")
            for t in targets:
                ta = np.mean([chosen["curves"][s][1]["test_loss_per_target"][t] for s in common])
                tb = np.mean([chosen["PHcurves"][s][1]["test_loss_per_target"][t] for s in common])
                print(f"  {t:>22}  {ta:>8.4f}  {tb:>8.4f}")

    return {"process": process, "metric": metric, "n_seeds": len(common),
            "curves": a.tolist(), "PHcurves": b.tolist(),
            "mean_gain": float(gain.mean()), "wins": int((gain > 0).sum()), "p": p}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processes", nargs="+", default=["classification"])
    ap.add_argument("--n-restarts", type=int, default=4)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    out = [report(args.results_root, p, args.n_restarts) for p in args.processes]
    if args.json:
        args.json.write_text(json.dumps([o for o in out if o], indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
