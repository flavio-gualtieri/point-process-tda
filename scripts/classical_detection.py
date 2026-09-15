#!/usr/bin/env python3
# scripts/classical_detection.py
"""The classical CSR test on the standard summary functions, over DV3's
evaluation sets -- the "no learning" row of every detection table.

For every pattern of every family in --sets it computes the studentised
global envelope statistic S (cloudforger.evaluation.classical) for L - r,
G and F, and their max (LGF), then writes one per-pattern score bundle per
(test, set):

    results/<process>/raw/envelope_<test>/seed_0/predictions_<set>.npz

in the same format the learned classifiers' bundles use, so
scripts/evaluate_regimes.py compares them in one call, e.g.

    python scripts/evaluate_regimes.py --process dv3_classify \
        --methods pi_multik vihrs envelope_L envelope_LGF --seeds 9371 ... 9380

The statistic is deterministic, so it is written once, as seed 0, flagged
`deterministic`; the regime aggregation pairs it with every seed of a
learned method. Nothing here needs persistence diagrams, so it can run as
soon as DV3 is generated.

Usage:
    python scripts/classical_detection.py --jobs 16                     # A, B, C; all families
    python scripts/classical_detection.py --sets C --jobs 16            # just the ladders + CSR anchors
    python scripts/classical_detection.py --process dv3_classify --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.evaluation import classical  # noqa: E402
from cloudforger.evaluation.dv3 import DEFAULT_DV3_ROOT, EVAL_SETS, FAMILIES, data_paths  # noqa: E402
from cloudforger.evaluation.predictions import save_score_predictions  # noqa: E402
from cloudforger.paths import DEFAULT_RESULTS_ROOT, ResultsPaths  # noqa: E402

DETERMINISTIC_SEED = 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_DV3_ROOT, help="DV3 root (default: data/dv3)")
    ap.add_argument("--sets", nargs="+", default=list(EVAL_SETS), choices=list(EVAL_SETS))
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--tests", nargs="+", default=list(classical.TESTS), choices=list(classical.TESTS))
    ap.add_argument("--process", default="dv3_classify",
                    help="results/<process>/ key to write under -- match the classification configs' process.name")
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--tables", type=Path, default=classical.NULL_TABLES, help="CSR null tables (.npz)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="recompute the cached per-(set, family) statistics")
    args = ap.parse_args(argv)

    results_paths = ResultsPaths(args.process, root=args.results_root)
    for set_ in args.sets:
        parts = []
        for family in args.families:
            if not data_paths(set_, family, args.root).clouds().exists():
                continue  # e.g. B has no CSR cells
            parts.append(classical.load_or_compute(set_, family, args.root, args.tables, args.jobs, args.force))
        if not parts:
            print(f"[{set_}] no clouds for {args.families}; skipping")
            continue
        stats = {k: np.concatenate([p[k] for p in parts]) for k in ("case_id", "family", "n", *[f"S_{t}" for t in args.tests])}
        below = float((stats["n"] < 50).mean())
        for test in args.tests:
            seed_dir = results_paths.seed_dir([], f"envelope_{test}", DETERMINISTIC_SEED)
            path = save_score_predictions(
                seed_dir, set_, case_id=stats["case_id"], family=stats["family"], score=stats[f"S_{test}"],
                score_name=f"S_{test}", method=f"envelope_{test}", seed=DETERMINISTIC_SEED,
                meta={
                    "deterministic": True,
                    # The single-function tests are exact 5% Monte Carlo tests at S > 1;
                    # the max over three functions is not, so it gets no nominal level.
                    "nominal_threshold": 1.0 if test in classical.FUNCTIONS else None,
                    "frac_n_below_grid": below,
                    "null_tables": str(args.tables),
                },
            )
            rej = float((stats[f"S_{test}"] > 1.0).mean())
            print(f"[{set_}] envelope_{test}: {len(stats['case_id'])} patterns, "
                  f"S>1 on {rej:.3f} (all families pooled) -> {path}")


if __name__ == "__main__":
    main()
