#!/usr/bin/env python3
# scripts/collect_restarts.py
"""Select one restart per seed by VALIDATION loss and pair the result
against vihrs.

Reads the restart__r<j> run-tags written by slurm/strauss_restarts.sh and,
for each seed independently, keeps the restart with the smallest
`best_val_loss` -- the minimum over that run's per-epoch validation losses,
surfaced into results.json by experiments/pi_multik/pi_multik.py. Test loss
is NEVER consulted for selection; it is only read out afterwards for the
selected run. That is what makes this restart selection rather than
cherry-picking, and it is the whole reason the script exists as a separate
step instead of a `min(test_loss)` one-liner.

Restarts of a seed share that seed's exact train/val/test partition
(init_offset perturbs only the global torch RNG; train_val_test_indices
draws from its own local generator), so "best val among restarts" is a
comparison within one fixed split.

Also reports the escape rate -- the fraction of restarts that cleared the
plateau -- because that number, not the selected loss, is what says whether
6 restarts was enough. Strauss stalls at train/val ~0.70 when it fails and
lands near 0.25 when it escapes, so the default --plateau 0.55 separates the
two modes cleanly; pass --plateau to retune for another process.

Usage:
    python scripts/collect_restarts.py
    python scripts/collect_restarts.py --process strauss --n-restarts 6
    python scripts/collect_restarts.py --json restarts.json
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

# vihrs is filtration-independent, so it lives under the "raw" tag; the
# subdir is vihrs_checkpointed whenever checkpoint_best is set (train.py's
# run_vihrs_method), which every current *_vihrs.yaml does.
VIHRS_DIR = Path("raw") / "vihrs_checkpointed"


def _load(base: Path) -> dict[int, dict]:
    """{seed -> results.json} for every seed_*/ directory under `base`."""
    out: dict[int, dict] = {}
    if not base.is_dir():
        return out
    for jf in base.glob("seed_*/results.json"):
        try:
            seed = int(jf.parent.name.split("seed_")[1])
        except (IndexError, ValueError):
            continue
        out[seed] = json.loads(jf.read_text())
    return out


def load_restarts(base: Path, n_restarts: int) -> dict[int, dict[int, dict]]:
    """{seed -> {restart index -> results.json}}."""
    by_seed: dict[int, dict[int, dict]] = {}
    for r in range(n_restarts):
        for seed, res in _load(base / "_runs" / f"restart__r{r}").items():
            by_seed.setdefault(seed, {})[r] = res
    return by_seed


def select(by_seed: dict[int, dict[int, dict]]) -> dict[int, tuple[int, dict]]:
    """{seed -> (chosen restart, its results.json)}, chosen on best_val_loss.

    A run missing best_val_loss predates the pi_multik patch that writes it;
    it is skipped with a warning rather than silently selected on something
    else, because falling back to test_loss here would quietly turn honest
    selection into cherry-picking.
    """
    chosen: dict[int, tuple[int, dict]] = {}
    for seed, runs in sorted(by_seed.items()):
        usable = {r: res for r, res in runs.items() if res.get("best_val_loss") is not None}
        if not usable:
            print(f"  seed {seed}: no run carries best_val_loss -- skipped "
                  f"(re-run under the patched pi_multik)")
            continue
        if len(usable) < len(runs):
            print(f"  seed {seed}: {len(runs) - len(usable)}/{len(runs)} restart(s) "
                  f"lack best_val_loss, ignored")
        r = min(usable, key=lambda k: usable[k]["best_val_loss"])
        chosen[seed] = (r, usable[r])
    return chosen


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--process", default="strauss")
    ap.add_argument("--filtration-tag", default="dtm_k5+10+15")
    ap.add_argument("--method", default="pi_multik")
    ap.add_argument("--n-restarts", type=int, default=6)
    ap.add_argument("--plateau", type=float, default=0.55,
                    help="test_loss above this counts as a failed escape (default 0.55)")
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    base = args.results_root / args.process / args.filtration_tag / args.method
    by_seed = load_restarts(base, args.n_restarts)
    if not by_seed:
        raise SystemExit(f"no restart__r* results under {base / '_runs'}")

    print(f"\n=== {args.process} | {args.filtration_tag} | {args.method} "
          f"| {args.n_restarts} restarts ===\n")

    all_losses = np.array([res["test_loss"] for runs in by_seed.values()
                           for res in runs.values()], float)
    n_escaped = int((all_losses < args.plateau).sum())
    print(f"escape rate: {n_escaped}/{len(all_losses)} restarts below "
          f"{args.plateau} ({100 * n_escaped / len(all_losses):.0f}%)")

    chosen = select(by_seed)
    if not chosen:
        raise SystemExit("nothing selectable")

    print(f"\n{'seed':>6}  {'restarts':>8}  {'picked':>6}  {'best_val':>9}  "
          f"{'test_loss':>9}  {'escaped/n':>9}")
    for seed, (r, res) in sorted(chosen.items()):
        runs = by_seed[seed]
        esc = sum(1 for x in runs.values() if x["test_loss"] < args.plateau)
        print(f"{seed:>6}  {len(runs):>8}  {r:>6}  {res['best_val_loss']:>9.4f}  "
              f"{res['test_loss']:>9.4f}  {esc:>4}/{len(runs)}")

    sel = np.array([res["test_loss"] for _, res in chosen.values()], float)
    print(f"\nselected: {sel.mean():.4f} +/- {sel.std(ddof=1):.4f}  (n={len(sel)}), "
          f"range [{sel.min():.4f}, {sel.max():.4f}]")

    # Naive baseline: what a single init_offset=0 run per seed would have
    # given. This is the number the restart machinery is buying you down from.
    r0 = {s: runs[0]["test_loss"] for s, runs in by_seed.items() if 0 in runs}
    if r0:
        v = np.array(list(r0.values()), float)
        print(f"single run (init_offset=0): {v.mean():.4f} +/- {v.std(ddof=1):.4f} (n={len(v)})")

    # Per-target on the selected runs -- the gap should be concentrated in the
    # interaction parameter (gamma for Strauss), not spread evenly.
    targets = sorted({k for _, res in chosen.values()
                      for k in (res.get("test_loss_per_target") or {})})
    vihrs = _load(args.results_root / args.process / VIHRS_DIR)
    common = sorted(set(chosen) & set(vihrs))

    if targets:
        print(f"\n{'target':>22}  {'selected':>9}  {'vihrs':>9}  {'delta':>8}")
        for t in targets:
            a = np.array([chosen[s][1]["test_loss_per_target"][t] for s in common], float)
            b = np.array([vihrs[s]["test_loss_per_target"][t] for s in common], float)
            if not len(a):
                continue
            print(f"{t:>22}  {a.mean():>9.4f}  {b.mean():>9.4f}  {a.mean() - b.mean():>+8.4f}")

    if not common:
        print(f"\nno vihrs seeds overlap (looked under "
              f"{args.results_root / args.process / VIHRS_DIR}) -- unpaired, not compared")
    else:
        a = np.array([chosen[s][1]["test_loss"] for s in common], float)
        b = np.array([vihrs[s]["test_loss"] for s in common], float)
        wins = int((a < b).sum())
        print(f"\nvs vihrs on {len(common)} shared seeds: "
              f"{a.mean():.4f} vs {b.mean():.4f} ({a.mean() - b.mean():+.4f}), "
              f"PH wins {wins}/{len(common)}")
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            wilcoxon = None
        diff = a - b
        if wilcoxon is not None and len(common) >= 6 and np.any(diff):
            print(f"  paired Wilcoxon p = {float(wilcoxon(diff).pvalue):.4f}")
        elif len(common) < 6:
            # n=5 floors the exact two-sided test at 2/2^5 = 0.0625.
            print(f"  n={len(common)} < 6: exact two-sided Wilcoxon cannot reach "
                  f"p<0.05 -- top vihrs up to 10 seeds before claiming anything")

    if args.json:
        args.json.write_text(json.dumps(
            {"process": args.process,
             "escape_rate": [n_escaped, len(all_losses)],
             "selected": {str(s): {"restart": r, "best_val_loss": res["best_val_loss"],
                                   "test_loss": res["test_loss"],
                                   "test_loss_per_target": res.get("test_loss_per_target")}
                          for s, (r, res) in sorted(chosen.items())}},
            indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
