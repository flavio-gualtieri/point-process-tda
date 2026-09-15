# src/cloudforger/evaluation/predictions.py
"""The per-pattern prediction bundle: the one artefact every method writes
and every regime analysis reads.

WHY. "Performance across regimes" is dozens of numbers per run (per cell of
B, per rung of C, per delta/nbar band of A, per target, per class), and they
must be computable for a CNN, for a minimum-contrast fit and for a classical
envelope test alike. Making each method compute them itself would mean
re-implementing the same grouping four times and re-running training every
time a band edge moves.

So each method instead writes what only it can produce -- its prediction for
every pattern of every evaluation set, tagged with the pattern's DV3
case_id -- and cloudforger.evaluation.regimes turns predictions + manifest
into every table afterwards, offline and identically for all of them. A
bundle is self-describing and small (a few MB for the whole of A/B/C), so
re-cutting the analysis costs seconds instead of GPU-hours.

Layout, one file per (method, seed, evaluation set):

    results/<process>/<tag>/<method>/seed_<seed>/predictions_<set>.npz

Parameter estimation carries predictions in BOTH scales:
  *_std   log + z-score, with the normalisation fit on the DV3 training
          rows -- the scale every loss in this project is quoted in, so
          MSE here is directly the normalised loss L
  *_raw   the natural parameter, which is what a bias/coverage statement at
          a fixed theta has to be made in
Classification carries the full class posterior, so accuracy, per-class
recall, and any CSR-vs-rest detection threshold are all recoverable without
re-running the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA_VERSION = 1


def predictions_path(seed_dir: Path | str, set_: str) -> Path:
    return Path(seed_dir) / f"predictions_{set_}.npz"


def _as_str_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([str(v) for v in values], dtype=np.str_)


def save_params_predictions(
    seed_dir: Path | str,
    set_: str,
    *,
    case_id: Sequence[str],
    family: Sequence[str],
    label_names: Sequence[str],
    pred_std: np.ndarray,
    truth_std: np.ndarray,
    pred_raw: np.ndarray | None = None,
    truth_raw: np.ndarray | None = None,
    method: str = "",
    seed: int | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write one evaluation set's parameter-estimation predictions.

    Rows may carry NaN (a classical fit that did not converge is a real,
    reportable outcome, not a row to drop): the regime analysis counts them
    as failures per cell rather than quietly averaging over the survivors.
    """
    pred_std = np.asarray(pred_std, dtype=np.float64)
    truth_std = np.asarray(truth_std, dtype=np.float64)
    if pred_std.shape != truth_std.shape:
        raise ValueError(f"pred_std {pred_std.shape} and truth_std {truth_std.shape} differ")
    if pred_std.ndim != 2 or pred_std.shape[1] != len(label_names):
        raise ValueError(f"pred_std must be (N, {len(label_names)}), got {pred_std.shape}")
    if len(case_id) != len(pred_std):
        raise ValueError(f"{len(case_id)} case_ids for {len(pred_std)} rows")

    payload: dict[str, Any] = {
        "case_id": _as_str_array(case_id),
        "family": _as_str_array(family),
        "label_names": _as_str_array(label_names),
        "pred_std": pred_std,
        "truth_std": truth_std,
        "meta": np.asarray(json.dumps({
            "schema": SCHEMA_VERSION, "task": "params", "set": set_,
            "method": method, "seed": seed, **(meta or {}),
        })),
    }
    if pred_raw is not None:
        payload["pred_raw"] = np.asarray(pred_raw, dtype=np.float64)
    if truth_raw is not None:
        payload["truth_raw"] = np.asarray(truth_raw, dtype=np.float64)

    path = predictions_path(seed_dir, set_)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def save_classify_predictions(
    seed_dir: Path | str,
    set_: str,
    *,
    case_id: Sequence[str],
    family: Sequence[str],
    class_names: Sequence[str],
    proba: np.ndarray,
    truth: np.ndarray,
    method: str = "",
    seed: int | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write one evaluation set's class posteriors (N, n_classes) and the true
    class index per pattern."""
    proba = np.asarray(proba, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    if proba.ndim != 2 or proba.shape[1] != len(class_names):
        raise ValueError(f"proba must be (N, {len(class_names)}), got {proba.shape}")
    if len(truth) != len(proba) or len(case_id) != len(proba):
        raise ValueError("case_id / proba / truth lengths disagree")

    path = predictions_path(seed_dir, set_)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        case_id=_as_str_array(case_id),
        family=_as_str_array(family),
        class_names=_as_str_array(class_names),
        proba=proba,
        truth=truth,
        meta=np.asarray(json.dumps({
            "schema": SCHEMA_VERSION, "task": "classify", "set": set_,
            "method": method, "seed": seed, **(meta or {}),
        })),
    )
    return path


def save_score_predictions(
    seed_dir: Path | str,
    set_: str,
    *,
    case_id: Sequence[str],
    family: Sequence[str],
    score: np.ndarray,
    score_name: str,
    method: str = "",
    seed: int | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write a bare per-pattern detection score (larger = more evidence
    against CSR). This is the shape a classical test has -- it estimates no
    parameter and predicts no class, it only rejects or not -- so it gets its
    own task rather than being forced into the classifier schema."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if len(score) != len(case_id):
        raise ValueError(f"{len(case_id)} case_ids for {len(score)} scores")
    path = predictions_path(seed_dir, set_)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        case_id=_as_str_array(case_id),
        family=_as_str_array(family),
        score=score,
        meta=np.asarray(json.dumps({
            "schema": SCHEMA_VERSION, "task": "detect", "set": set_, "score_name": score_name,
            "method": method, "seed": seed, **(meta or {}),
        })),
    )
    return path


def load_predictions(path: Path | str) -> dict[str, Any]:
    """Read any of the three bundle shapes back into a plain dict, with
    `meta` decoded and `task` promoted to a top-level key."""
    with np.load(path, allow_pickle=False) as z:
        out: dict[str, Any] = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    out["meta"] = meta
    out["task"] = meta.get("task", "params")
    out["set"] = meta.get("set")
    out["method"] = meta.get("method")
    out["seed"] = meta.get("seed")
    for key in ("case_id", "family", "label_names", "class_names"):
        if key in out:
            out[key] = np.asarray(out[key], dtype=object)
    return out


def find_predictions(seed_dir: Path | str, sets: Sequence[str]) -> dict[str, Path]:
    """{set -> path} for the bundles that actually exist under one seed dir."""
    seed_dir = Path(seed_dir)
    return {s: p for s in sets if (p := predictions_path(seed_dir, s)).exists()}
