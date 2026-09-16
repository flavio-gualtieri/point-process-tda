#!/usr/bin/env python
"""Re-base every result in results/ on the stratified test set.

    python scripts/rescore_results.py --dry-run
    python scripts/rescore_results.py

The old headline numbers were computed on the prior-draw set alone. This
rewrites them, in place, from the same prediction bundles -- no model is
re-run and no feature is recomputed (see cloudforger.evaluation.testset).

What it writes, per seed directory:
  results.json      headline metrics replaced by their test-set values; the
                    superseded numbers are kept under `superseded_prior_set`
                    so nothing is silently lost, and a `test_set` block adds
                    the per-stratum breakdown.
  metrics.json      per-target marginal metrics, recomputed on the test set's
                    rows for that family (same schema as before).
  testset.json      the full per-seed table, overall and per stratum.

and, once:
  results/testset_summary.json   every run, aggregated over seeds.
  results/experiments.jsonl      the ledger, re-pointed at the test set.

The prediction bundles are never touched: they are the source the test set is
selected from, and the only copy of what each model actually predicted.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloudforger.core.metrics import compute_marginal_metrics, json_default  # noqa: E402
from cloudforger.evaluation import testset as T  # noqa: E402
from cloudforger.evaluation.testset_scoring import pool_seed, score_run  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
BASIS = "stratified test set (configs/testset.csv)"


def run_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.glob("dv3_*/*/*") if d.is_dir() and any(d.glob("seed_*")))


def marginal_on_testset(seed_dir: Path, ts: T.TestSet) -> dict | None:
    """metrics.json's per-target numbers, recomputed on the test set's rows."""
    bundle = pool_seed(seed_dir)
    if bundle is None or bundle["task"] == "classify":
        return None
    if "pred_raw" not in bundle or "truth_raw" not in bundle:
        return None
    pos = {str(c): i for i, c in enumerate(bundle["case_id"])}
    idx = np.array([pos[r["case_id"]] for r in ts.rows if r["case_id"] in pos], dtype=np.int64)
    if idx.size == 0:
        return None
    truth, pred = bundle["truth_raw"][idx], bundle["pred_raw"][idx]
    return {name: compute_marginal_metrics(truth[:, j], pred[:, j])
            for j, name in enumerate(bundle["label_names"])}


def rewrite_results_json(path: Path, table: dict) -> None:
    """Replace the headline metrics with their test-set values, keeping the
    superseded ones under one key rather than destroying them."""
    doc = json.loads(path.read_text()) if path.exists() else {}
    overall = table["overall"]

    superseded = {k: doc[k] for k in
                  ("test_loss", "test_loss_per_target", "test_accuracy", "best_val_accuracy")
                  if k in doc}
    if superseded and "superseded_prior_set" not in doc:
        doc["superseded_prior_set"] = {
            **superseded,
            "basis": "prior-draw set A only (superseded by the stratified test set)",
        }

    if table["task"] == "classify":
        doc["test_accuracy"] = overall["accuracy"]
        doc["test_loss"] = overall["log_loss"]
        doc["test_recall"] = overall["recall"]
    else:
        doc["test_loss"] = overall["value"]
        doc["test_loss_per_target"] = overall.get("loss_std")

    doc["evaluation_basis"] = BASIS
    doc["test_set"] = {
        "n_covered": table["n_covered"],
        "n_testset": table["n_testset"],
        "families": table["families"],
        "metric": overall["metric"],
        "overall": overall["value"],
        "clustered_se": overall["se"],
        "strata": {s: {"n": g["n"], overall["metric"]: g["value"], "clustered_se": g["se"]}
                   for s, g in table["strata"].items()},
    }
    path.write_text(json.dumps(doc, indent=2, default=json_default))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--testset", type=Path, default=T.DEFAULT_PATH)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ts = T.load(args.testset)
    runs = run_dirs(args.results)
    print(f"test set: {len(ts)} cases, {ts.per_cell} per cell")
    print(f"runs to re-base: {len(runs)}\n")

    summary, ledger = [], []
    n_results = n_metrics = n_tables = 0

    for run in runs:
        agg, tables = score_run(run, ts)
        summary.append(agg)
        for seed_dir, table in tables:
            if not args.dry_run:
                (seed_dir / "testset.json").write_text(
                    json.dumps(table, indent=2, default=json_default))
                rewrite_results_json(seed_dir / "results.json", table)
            n_tables += 1
            n_results += 1
            marg = marginal_on_testset(seed_dir, ts)
            if marg is not None:
                if not args.dry_run:
                    (seed_dir / "metrics.json").write_text(
                        json.dumps(marg, indent=2, default=json_default))
                n_metrics += 1
            doc = json.loads((seed_dir / "results.json").read_text())
            ledger.append({
                "process": run.parent.parent.name,
                "filtration_tag": run.parent.name,
                "method": doc.get("method"),
                "run": run.name,
                "seed": doc.get("seed"),
                "task": agg["task"],
                "metric": agg["metric"],
                "value": table["overall"]["value"],
                "clustered_se": table["overall"]["se"],
                "strata": {s: g["value"] for s, g in table["strata"].items()},
                "evaluation_basis": BASIS,
                "git_commit": doc.get("git_commit"),
                "output_dir": str(seed_dir),
            })
        rel = run.relative_to(args.results)
        bar = "  ".join(f"{s}={agg['strata'][s]['mean']:.3f}"
                        for s in T.STRATUM_NAMES if s in agg["strata"])
        print(f"{str(rel):<52} {agg['metric']:<8} {agg['overall']['mean']:.4f}   {bar}")

    if args.dry_run:
        print(f"\ndry run: would rewrite {n_results} results.json, "
              f"{n_metrics} metrics.json, {n_tables} testset.json")
        return 0

    (args.results / "testset_summary.json").write_text(json.dumps({
        "basis": BASIS,
        "testset": str(args.testset),
        "n_cases": len(ts),
        "per_cell": ts.per_cell,
        "strata": list(T.STRATUM_NAMES),
        "generated": datetime.now(timezone.utc).isoformat(),
        "runs": summary,
    }, indent=2, default=json_default))
    with (args.results / "experiments.jsonl").open("w") as fh:
        for row in ledger:
            fh.write(json.dumps(row, default=json_default) + "\n")

    print(f"\nrewrote {n_results} results.json, {n_metrics} metrics.json, "
          f"{n_tables} testset.json")
    print(f"wrote {args.results / 'testset_summary.json'}")
    print(f"wrote {args.results / 'experiments.jsonl'} ({len(ledger)} rows)")
    print("\nprediction bundles untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
