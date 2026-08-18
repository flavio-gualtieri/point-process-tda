#!/usr/bin/env python3
# scripts/collect_vectorization_results.py
"""Tidy results table for the landscape/silhouette vectorization sweep: one
row per (filtration, vectorization, p, encoder_path, seed), with columns
for total test loss, per-parameter test loss (kappa/mu/sigma -- the
standard Thomas-process names for parent_intensity/mean_offspring/
cluster_scale, see PALETTE_RENAME below), and feature_dim.

Reads results.json under <results_root>/<process>/<filtration_tag>/
vec_multik_<vectorization>_<encoder_path>[_p<value>]/seed_<seed>/ -- i.e.
every method: vec_multik run (see experiments/pi_multik/vectorized_multik.py),
including vectorization=persistence_image/encoder_path=native runs, which
delegate to (and are byte-identical to) plain method: pi_multik but still
get filed under their own vec_multik_persistence_image_native subdir (see
that module's docstring) -- so this script's rows are exactly the
"vectorization=persistence_image must behave identically" arm alongside
every other arm, not a separate baseline that needs merging in by hand.

vec_multik's own save_results call (see VectorizedMultiKExperiment.run())
already writes vectorization/encoder_path/p/feature_dim into results.json's
config, so this script only has to walk the directory tree and flatten
that -- no re-parsing of subdir names.

Usage:
    python scripts/collect_vectorization_results.py \\
        --results-root results --process thomas --out results/vec_multik_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.paths import DEFAULT_RESULTS_ROOT

PARAMETER_ALIASES = {
    "parent_intensity": "kappa",
    "mean_offspring": "mu",
    "cluster_scale": "sigma",
}


def _process_and_filtration_tag(result_dir: Path, results_root: Path) -> tuple[str | None, str | None]:
    try:
        parts = result_dir.resolve().relative_to(results_root.resolve()).parts
    except ValueError:
        return None, None
    process = parts[0] if len(parts) > 0 else None
    filtration_tag = parts[1] if len(parts) > 1 else None
    return process, filtration_tag


def collect_rows(results_root: Path, process: str | None = None) -> list[dict]:
    # Two layouts: results/<process>/<filtration_tag>/vec_multik_.../seed_<seed>/,
    # and, for runs launched with --run-tag (e.g. the "matched" feature-budget
    # arms, which must use --run-tag to avoid colliding with the "native
    # budget" arm sharing the same (vectorization, encoder_path) subdir name
    # -- see the vec_multik_*_matched.yaml configs' own comments),
    # .../vec_multik_.../_runs/<run_tag>/seed_<seed>/ (paths.py's
    # RUN_ARCHIVE_DIR_NAME). Both glob patterns below are searched.
    patterns = ["*/*/vec_multik_*/seed_*/results.json", "*/*/vec_multik_*/_runs/*/seed_*/results.json"]
    result_paths = sorted({p for pattern in patterns for p in results_root.glob(pattern)})

    rows: list[dict] = []
    for results_json in result_paths:
        seed_dir = results_json.parent
        result = json.loads(results_json.read_text())
        cfg = result.get("config", {})

        this_process, filtration_tag = _process_and_filtration_tag(seed_dir, results_root)
        if process is not None and this_process != process:
            continue

        row = {
            "process": this_process,
            "filtration_tag": filtration_tag,
            "vectorization": cfg.get("vectorization"),
            "encoder_path": cfg.get("encoder_path"),
            "p": cfg.get("p"),
            "run_tag": result.get("run_tag"),  # disambiguates e.g. "matched" vs native-budget arms sharing a subdir
            "seed": result.get("seed"),
            "feature_dim": cfg.get("feature_dim"),
            "test_loss": result.get("test_loss"),
        }

        per_target = result.get("test_loss_per_target") or {}
        for name, value in per_target.items():
            column = f"test_loss_{PARAMETER_ALIASES.get(name, name)}"
            row[column] = value

        rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        raise SystemExit("No vec_multik results.json files found -- nothing to write.")
    fixed_columns = ["process", "filtration_tag", "vectorization", "p", "encoder_path", "run_tag", "seed", "feature_dim", "test_loss"]
    per_target_columns = sorted({k for row in rows for k in row if k.startswith("test_loss_") and k != "test_loss"})
    columns = fixed_columns + per_target_columns

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--process", type=str, default=None, help="Restrict to one process (e.g. thomas). Default: every process.")
    parser.add_argument("--out", type=Path, default=None, help="CSV output path. Default: <results-root>/vec_multik_table.csv")
    args = parser.parse_args(argv)

    out_path = args.out or (args.results_root / "vec_multik_table.csv")
    rows = collect_rows(args.results_root, process=args.process)
    write_csv(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
