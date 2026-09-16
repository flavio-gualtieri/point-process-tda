# src/cloudforger/evaluation/testset_scoring.py
"""Score an existing run against the stratified test set.

Nothing here touches a model, a cloud or a diagram. A run's per-set prediction
bundles already cover every evaluation pattern, so scoring on the test set is a
join on case_id followed by the same metric functions every other table uses
(`regimes.params_metrics`, `regimes.classify_metrics`). That is why the test-set
redesign costs no compute: the expensive stages all sit upstream of the join.

Uncertainty is clustered on theta_id. The fixed-theta sources contribute
hundreds of replicates at one parameter vector, so a standard error taken over
rows would treat replicates as independent draws and come out too small; the
cluster-robust version averages within a parameter vector first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import testset as T
from .predictions import load_predictions
from .regimes import classify_metrics, params_metrics

SOURCE_SETS: tuple[str, ...] = ("A", "B", "C")

_STACK_KEYS = ("case_id", "family", "proba", "truth", "pred_std", "truth_std",
               "pred_raw", "truth_raw")


def pool_seed(seed_dir: Path | str) -> dict[str, Any] | None:
    """One seed's per-set bundles concatenated into a single bundle, or None
    if the seed wrote no predictions at all."""
    seed_dir = Path(seed_dir)
    parts = [load_predictions(seed_dir / f"predictions_{s}.npz")
             for s in SOURCE_SETS if (seed_dir / f"predictions_{s}.npz").exists()]
    if not parts:
        return None
    out: dict[str, Any] = {"task": parts[0]["task"]}
    for key in ("label_names", "class_names"):
        if key in parts[0]:
            out[key] = parts[0][key]
    for key in _STACK_KEYS:
        if key in parts[0]:
            out[key] = np.concatenate([p[key] for p in parts], axis=0)
    return out


def clustered_se(values: np.ndarray, clusters: np.ndarray) -> float:
    """S.e. of the mean of `values`, clustering on `clusters`: average within
    each cluster, then take the s.e. over cluster means. Reduces to the plain
    s.e. when every cluster is a singleton."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    _, inv = np.unique(clusters, return_inverse=True)
    k = int(inv.max()) + 1
    if k < 2:
        return float("nan")
    means = np.bincount(inv, weights=values, minlength=k) / np.bincount(inv, minlength=k)
    return float(means.std(ddof=1) / np.sqrt(k))


def _rows_for(bundle: dict[str, Any], ts: T.TestSet) -> tuple[list[dict], np.ndarray]:
    """The test-set rows this bundle actually predicts, and where they sit in
    it. A parameter run covers only its own family; a classification run covers
    all five. Rows the bundle does not carry are simply absent, and the caller
    reports the coverage rather than assuming it is complete."""
    pos = {str(c): i for i, c in enumerate(bundle["case_id"])}
    rows = [r for r in ts.rows if r["case_id"] in pos]
    idx = np.array([pos[r["case_id"]] for r in rows], dtype=np.int64)
    return rows, idx


def score_bundle(bundle: dict[str, Any], ts: T.TestSet) -> dict[str, Any]:
    """Overall and per-stratum metrics for one seed."""
    rows, idx = _rows_for(bundle, ts)
    if not rows:
        raise ValueError("no test-set case_id appears in these predictions")
    strata = np.array([r["stratum"] for r in rows], dtype=object)
    theta = np.array([r["theta_id"] for r in rows], dtype=object)
    task = bundle["task"]

    def metrics(mask: np.ndarray) -> dict[str, Any]:
        sel = idx[mask]
        if task == "classify":
            out = classify_metrics(bundle["proba"][sel], bundle["truth"][sel],
                                   list(bundle["class_names"]))
            per_row = (bundle["proba"][sel].argmax(1) == bundle["truth"][sel]).astype(float)
            out["metric"], out["value"] = "accuracy", out["accuracy"]
        else:
            out = params_metrics(
                bundle["pred_std"][sel], bundle["truth_std"][sel],
                list(bundle["label_names"]),
                bundle["pred_raw"][sel] if "pred_raw" in bundle else None,
                bundle["truth_raw"][sel] if "truth_raw" in bundle else None,
            )
            per_row = ((bundle["pred_std"][sel] - bundle["truth_std"][sel]) ** 2).mean(axis=1)
            out["metric"], out["value"] = "loss", out.get("loss", float("nan"))
        out["se"] = clustered_se(per_row, theta[mask])
        return out

    result: dict[str, Any] = {
        "task": task,
        "n_covered": len(rows),
        "n_testset": len(ts),
        "families": sorted({r["family"] for r in rows}),
        "overall": metrics(np.ones(len(rows), dtype=bool)),
        "strata": {},
    }
    for s in T.STRATUM_NAMES:
        m = strata == s
        if m.any():
            result["strata"][s] = metrics(m)
    return result


def seed_dirs(run: Path | str) -> list[Path]:
    return sorted(p for p in Path(run).glob("seed_*") if p.is_dir())


def score_run(run: Path | str, ts: T.TestSet) -> tuple[dict[str, Any], list[tuple[Path, dict]]]:
    """Aggregate a run over its seeds, and hand back the per-seed tables too so
    a caller can write them next to the seed they came from.

    Two spreads are reported and they answer different questions: `sd` is the
    spread across seeds (how much the fit depends on initialisation) and
    `se_within` is the cluster-robust error of one seed's own estimate (how
    much the test set itself pins the number down)."""
    tables: list[tuple[Path, dict]] = []
    for sd in seed_dirs(run):
        bundle = pool_seed(sd)
        if bundle is not None:
            tables.append((sd, score_bundle(bundle, ts)))
    if not tables:
        raise ValueError(f"{run}: no prediction bundles")
    per_seed = [t for _, t in tables]

    def across(getter: Callable[[dict], float]) -> dict[str, float]:
        v = np.array([getter(r) for r in per_seed], dtype=float)
        return {"mean": float(np.nanmean(v)),
                "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0}

    agg: dict[str, Any] = {
        "run": str(run),
        "task": per_seed[0]["task"],
        "n_seeds": len(per_seed),
        "n_covered": per_seed[0]["n_covered"],
        "families": per_seed[0]["families"],
        "metric": per_seed[0]["overall"]["metric"],
        "overall": across(lambda r: r["overall"]["value"]),
        "strata": {},
    }
    agg["overall"]["se_within"] = float(np.nanmean([r["overall"]["se"] for r in per_seed]))
    for s in T.STRATUM_NAMES:
        if s in per_seed[0]["strata"]:
            agg["strata"][s] = across(lambda r, s=s: r["strata"][s]["value"])
            agg["strata"][s]["n"] = per_seed[0]["strata"][s]["n"]
            agg["strata"][s]["se_within"] = float(
                np.nanmean([r["strata"][s]["se"] for r in per_seed]))
    if agg["task"] == "classify":
        agg["recall"] = {c: across(lambda r, c=c: r["overall"]["recall"][c])
                         for c in per_seed[0]["overall"]["recall"]}
    else:
        agg["per_target"] = {
            name: across(lambda r, name=name: r["overall"]["loss_std"][name])
            for name in per_seed[0]["overall"].get("loss_std", {})
        }
    return agg, tables
