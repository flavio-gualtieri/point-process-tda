#!/usr/bin/env python
"""Assemble the stratified test set from the patterns that already exist.

    python scripts/build_testset.py                 # write configs/testset.csv
    python scripts/build_testset.py --per-cell 2000
    python scripts/build_testset.py --dry-run       # report balance, write nothing

Selection only: every row is a case_id whose clouds, diagrams and predictions
are already on disk. See cloudforger.evaluation.testset for the design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloudforger.evaluation import testset as T  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-cell", type=int, default=T.DEFAULT_PER_CELL,
                    help=f"cases per (family, stratum) cell (default {T.DEFAULT_PER_CELL})")
    ap.add_argument("--seed", type=int, default=T.DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=T.DEFAULT_PATH)
    ap.add_argument("--dry-run", action="store_true", help="report and exit")
    args = ap.parse_args()

    ts, warnings = T.build(per_cell=args.per_cell, seed=args.seed)

    print(f"test set: {len(ts)} cases, {args.per_cell} per cell, seed {args.seed}\n")

    strata = list(T.STRATUM_NAMES)
    counts = ts.counts()
    width = max(len(f) for f in counts)
    print("cases per cell".ljust(width + 2) + "".join(s.rjust(10) for s in strata) + "     total")
    for fam in sorted(counts):
        row = counts[fam]
        line = fam.ljust(width + 2) + "".join(str(row.get(s, 0)).rjust(10) for s in strata)
        print(line + str(sum(row.values())).rjust(10))

    print("\nnbar composition (share of each cell, by octave)")
    comp = ts.composition()
    bands = T.nbar_band_labels()
    print("".ljust(width + 12) + "".join(b.rjust(12) for b in bands))
    for fam in sorted(comp):
        for s in strata:
            cell = comp[fam].get(s)
            if not cell:
                continue
            tot = sum(cell.values())
            shares = "".join(f"{cell.get(b, 0) / tot:11.0%} " for b in bands)
            print(f"{fam}/{s}".ljust(width + 12) + shares)

    print("\nprovenance (source set of the selected rows)")
    for k, v in sorted(ts.provenance().items()):
        print(f"  {k}: {v}")

    n_theta = len({r["theta_id"] for r in ts.rows})
    print(f"\ndistinct theta: {n_theta}  ({len(ts) / n_theta:.1f} replicates per theta on average)")

    if warnings:
        print("\nunder-filled cells:")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("\nevery cell filled to quota.")

    if args.dry_run:
        return 0

    path = T.write_csv(ts, args.out)
    summary = {
        "n_cases": len(ts), "per_cell": args.per_cell, "seed": args.seed,
        "strata": strata, "counts": counts, "composition": comp,
        "provenance": ts.provenance(), "n_theta": n_theta, "warnings": warnings,
    }
    Path(path).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {path}")
    print(f"wrote {Path(path).with_suffix('.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
