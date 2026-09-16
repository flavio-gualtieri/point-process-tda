#!/usr/bin/env python
"""Score runs against the stratified test set, without re-running anything.

    python scripts/evaluate_testset.py results/dv3_classify/raw/vihrs_lfgj
    python scripts/evaluate_testset.py results/dv3_*/*/*  --json tables.json

A run directory holds seed_*/predictions_<set>.npz. Those bundles already cover
every evaluation pattern, so this pools them, keeps the rows the test set
selected, and reports per stratum. No model is re-run and no feature recomputed.
To write the scores back into results/ instead, use scripts/rescore_results.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloudforger.evaluation import testset as T  # noqa: E402
from cloudforger.evaluation.testset_scoring import score_run  # noqa: E402


def show(agg: dict) -> None:
    metric = agg["metric"]
    print(f"\n{agg['run']}")
    print(f"  task={agg['task']}  seeds={agg['n_seeds']}  rows covered={agg['n_covered']}  "
          f"metric={metric} ({'higher' if metric == 'accuracy' else 'lower'} is better)")
    o = agg["overall"]
    print(f"  overall            {o['mean']:.4f} +- {o['sd']:.4f} (seed sd), "
          f"+- {o['se_within']:.4f} (clustered)")
    for s in T.STRATUM_NAMES:
        if s in agg["strata"]:
            g = agg["strata"][s]
            print(f"    {s:<10} n={g['n']:>6}  {g['mean']:.4f} +- {g['sd']:.4f}  "
                  f"(clustered +- {g['se_within']:.4f})")
    if "recall" in agg:
        print("  per-class recall: " + "  ".join(
            f"{k}={v['mean']:.3f}" for k, v in agg["recall"].items()))
    elif agg.get("per_target"):
        print("  per-target loss:  " + "  ".join(
            f"{k}={v['mean']:.4f}" for k, v in agg["per_target"].items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--testset", type=Path, default=T.DEFAULT_PATH)
    ap.add_argument("--json", type=Path, help="write the aggregated tables here")
    args = ap.parse_args()

    ts = T.load(args.testset)
    out = []
    for d in args.runs:
        agg, _ = score_run(d, ts)
        out.append(agg)
        show(agg)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
