#!/usr/bin/env python3
"""Per-regime tables from the saved predictions: results/**/predictions.npz -> results/regimes.csv.

    python scripts/regimes.py                                    # every run under results/
    python scripts/regimes.py --task classify --reference dtm_k10/h01

Nothing about regimes exists during training. A run's predictions carry `case_id`, which joins to
data/simulation/<family>/manifest.csv for the regime coordinates (nbar, delta), so the binning here
can change without retraining anything.

One long row per (run, cell, metric), so a figure or a table is one filter:

    task group filtration dims target metric family nbar_bin delta_bin
    n_thetas estimate lo hi seed_sd reference n_seeds

`nbar_bin`/`delta_bin` hold the bin's left edge and are empty on a marginal row; the marginals are
rows like any other, so "overall accuracy" is not a separate output. `family` is a single family or
`all`; for a single family, accuracy IS that family's recall. Poisson has delta = 0, so it lands in
no delta bin and appears only in the marginal and nbar rows -- the overall baseline, not part of the
regime curves.

Metrics: accuracy and nll (mean negative log posterior of the true family) for classification; for
parameters, rmse_log per target and over all targets, in LOG units, because the parameters span
orders of magnitude and were trained on a log scale.

Uncertainty is a bootstrap over TEST THETAS, not patterns: the two replicates of a theta share their
parameters, so they are resampled together. Seeds are averaged per pattern before aggregating, and
`seed_sd` reports the spread of the per-seed cell estimate alongside. With --reference, extra rows
carry the PAIRED difference against that run, on the same theta resamples -- much tighter than
comparing two independent intervals, and the comparison the paper makes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.simulation.sweep import Config, DATA as SIMULATION   # noqa: E402

RESULTS = ROOT / "results"
N_DELTA_BINS, N_NBAR_BINS, N_BOOT = 8, 3, 1000


def manifest(families: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(SIMULATION / f / "manifest.csv") for f in families]
    return pd.concat(frames, ignore_index=True).set_index("case_id")[["family", "theta", "nbar", "delta"]]


def edges(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Fixed log-spaced edges from the sweep's own prior, so every family and method shares cells."""
    return (np.geomspace(*cfg.delta, N_DELTA_BINS + 1), np.geomspace(*cfg.nbar, N_NBAR_BINS + 1))


def bin_left(values: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Left edge of each value's bin, nan outside the range (delta = 0 for Poisson)."""
    i = np.searchsorted(edge, values, side="right") - 1
    i = np.where((i >= 0) & (i < len(edge) - 1), i, -1)
    return np.where(i < 0, np.nan, edge[i])


def find_runs(root: Path, task: str | None) -> dict[tuple, list[Path]]:
    """seed directories grouped by (task, group, filtration, dims)."""
    runs: dict[tuple, list[Path]] = {}
    for predictions in sorted(root.glob("*/*/*/*/seed_*/predictions.npz")):
        seed_dir = predictions.parent
        key = tuple(seed_dir.parts[-5:-1])          # task, group, filtration, dims
        if task and key[0] != task:
            continue
        runs.setdefault(key, []).append(seed_dir)
    return runs


def per_pattern(seed_dirs: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    """Per-pattern scores, one column per metric, averaged over seeds; plus the per-seed columns."""
    frames, case_id = [], None
    for seed_dir in seed_dirs:
        z = np.load(seed_dir / "predictions.npz")
        run = json.loads((seed_dir / "run.json").read_text())
        if case_id is None:
            case_id = z["case_id"]
        elif not np.array_equal(case_id, z["case_id"]):
            raise SystemExit(f"{seed_dir}: test patterns differ from the other seeds of this run")

        scores = {}
        if run["args"]["task"] == "classify":
            posterior = z["posterior"]
            scores["accuracy"] = (z["y_pred"] == z["y_true"]).astype(float)
            scores["nll"] = -np.log(np.clip(posterior[np.arange(len(z["y_true"])), z["y_true"]], 1e-12, None))
        else:
            # Squared error in log units: y_true/y_pred are in parameter units, all positive.
            square = (np.log(z["y_pred"]) - np.log(z["y_true"])) ** 2
            for j, column in enumerate(run["targets"]["columns"]):
                scores[f"rmse_log:{column}"] = square[:, j]
            scores["rmse_log"] = square.mean(axis=1)
        frames.append(pd.DataFrame(scores, index=case_id))

    metrics = list(frames[0].columns)
    mean = sum(frames) / len(frames)
    for i, frame in enumerate(frames):
        mean[[f"{m}__seed{i}" for m in metrics]] = frame.to_numpy()
    return mean, metrics


def per_theta(scores: pd.DataFrame, rows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Collapse replicates: one row per theta, which is the unit the bootstrap resamples."""
    delta_edge, nbar_edge = edges(cfg)
    joined = scores.join(rows, how="inner")
    if len(joined) != len(scores):
        raise SystemExit("some predicted case_ids are missing from the manifest")
    joined["delta_bin"] = bin_left(joined["delta"].to_numpy(), delta_edge)
    joined["nbar_bin"] = bin_left(joined["nbar"].to_numpy(), nbar_edge)
    keys = ["family", "theta", "delta_bin", "nbar_bin"]
    return joined.groupby(keys, dropna=False, observed=True).mean(numeric_only=True).reset_index()


def estimate(metric: str, means: np.ndarray) -> np.ndarray:
    """Cell estimate from per-theta column means; sqrt for the squared-error metrics."""
    return np.sqrt(means) if metric.startswith("rmse_log") else means


def bootstrap(values: np.ndarray, metric: str, rng, n_boot: int) -> tuple[float, float, float]:
    """Resample thetas (rows), recompute the estimate, take percentiles. Paired when values has two
    columns: the difference is taken on the SAME resample."""
    point = estimate(metric, values.mean(axis=0))
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    draws = estimate(metric, values[idx].mean(axis=1))
    if values.shape[1] == 2:                     # paired: the difference on each resample
        point, draws = point[0] - point[1], draws[:, 0] - draws[:, 1]
    else:
        point, draws = point[0], draws[:, 0]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def cells(table: pd.DataFrame, families: list[str]) -> list[tuple]:
    """(family, delta_bin, nbar_bin) cells, including the marginals, where `None` means "any"."""
    deltas = [None] + sorted(table.delta_bin.dropna().unique())
    nbars = [None] + sorted(table.nbar_bin.dropna().unique())
    groups = ([*families, "all"] if len(families) > 1 else families)
    return [(f, d, n) for f in groups for d in deltas for n in nbars]


def select(table: pd.DataFrame, family, delta_bin, nbar_bin) -> np.ndarray:
    mask = np.ones(len(table), dtype=bool)
    if family != "all":
        mask &= (table.family == family).to_numpy()
    if delta_bin is not None:
        mask &= (table.delta_bin == delta_bin).to_numpy()
    if nbar_bin is not None:
        mask &= (table.nbar_bin == nbar_bin).to_numpy()
    return mask


def rows_for_run(key, table, metrics, n_seeds, reference, reference_id, rng, n_boot, min_thetas) -> list[dict]:
    task, group, filtration, dims = key
    families = sorted(table.family.unique())
    out = []
    for family, delta_bin, nbar_bin in cells(table, families):
        mask = select(table, family, delta_bin, nbar_bin)
        if mask.sum() < min_thetas:      # too few thetas for an interval worth reading
            continue
        subset = table[mask]
        for metric in metrics:
            values = subset[metric].to_numpy()[:, None]
            if reference is not None:
                pair = reference[select(reference, family, delta_bin, nbar_bin)]
                if len(pair) != len(subset) or not np.array_equal(pair.theta.to_numpy(), subset.theta.to_numpy()):
                    raise SystemExit(f"{key}: reference run covers different thetas")
                values = np.column_stack([values[:, 0], pair[metric].to_numpy()])
            point, lo, hi = bootstrap(values, metric, rng, n_boot)
            seeds = [estimate(metric, subset[f"{metric}__seed{i}"].to_numpy().mean()) for i in range(n_seeds)]
            out.append({
                "task": task, "group": group, "filtration": filtration, "dims": dims,
                "target": metric.split(":")[1] if ":" in metric else ("all" if task == "params" else ""),
                "metric": metric.split(":")[0], "family": family,
                "nbar_bin": nbar_bin, "delta_bin": delta_bin, "n_thetas": len(subset),
                "estimate": point, "lo": lo, "hi": hi, "seed_sd": float(np.std(seeds, ddof=1)) if n_seeds > 1 else np.nan,
                "reference": reference_id, "n_seeds": n_seeds,
            })
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--task", choices=["classify", "params"])
    p.add_argument("--reference", help="'<filtration>/h<dims>' to compare every other run against")
    p.add_argument("--out", type=Path, help="default: <results>/regimes.csv")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--min-thetas", type=int, default=10, help="skip cells with fewer test thetas")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = Config.load()
    runs = find_runs(args.results, args.task)
    if not runs:
        raise SystemExit(f"no runs under {args.results}")
    rng = np.random.default_rng(args.seed)

    tables, metrics_of, n_seeds_of = {}, {}, {}
    rows_cache: dict[tuple, pd.DataFrame] = {}
    for key, seed_dirs in runs.items():
        scores, metrics = per_pattern(seed_dirs)
        families = sorted({c.rsplit("-", 2)[0] for c in scores.index})
        if tuple(families) not in rows_cache:
            rows_cache[tuple(families)] = manifest(families)
        tables[key] = per_theta(scores, rows_cache[tuple(families)], cfg)
        metrics_of[key], n_seeds_of[key] = metrics, len(seed_dirs)
        print(f"[regimes] {'/'.join(key)}: {len(seed_dirs)} seeds, {len(tables[key])} test thetas", flush=True)

    out_rows = []
    for key, table in tables.items():
        out_rows += rows_for_run(key, table, metrics_of[key], n_seeds_of[key], None, "", rng, args.n_boot,
                                 args.min_thetas)
        if args.reference:
            ref_key = (key[0], key[1], *args.reference.split("/"))
            if ref_key in tables and ref_key != key:
                out_rows += rows_for_run(key, table, metrics_of[key], n_seeds_of[key],
                                         tables[ref_key], args.reference, rng, args.n_boot,
                                         args.min_thetas)

    out = args.out or args.results / "regimes.csv"
    frame = pd.DataFrame(out_rows)
    frame.to_csv(out, index=False)
    print(f"[regimes] {len(frame)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
