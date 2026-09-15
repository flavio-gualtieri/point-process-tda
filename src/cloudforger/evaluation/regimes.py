# src/cloudforger/evaluation/regimes.py
"""Turn one prediction bundle + the DV3 manifests into regime tables.

Every function here is pure (arrays in, JSON-able dicts out) and knows
nothing about which method produced the predictions -- a CNN, a
minimum-contrast fit and a classical envelope test go through exactly the
same code, which is what makes their regime tables comparable row by row.

Row keys are strings, stable across methods and seeds, so the across-seed
aggregation (scripts/evaluate_regimes.py) can line rows up by key alone:

    A/<family>/all                         whole set, one family
    A/<family>/delta=<band>                marginal over nbar
    A/<family>/nbar=<band>                 marginal over delta
    A/<family>/delta=<band>/nbar=<band>    the joint grid
    B/<family>/cell=<id>                   one fixed-theta cell
    C/<family>/ladder=<id>/level=<id>      one rung of one ladder
    C/poisson/nbar=<nbar>                  a CSR anchor

What is measured where (docs/generation_procedure.tex, Table "The four data
products"):

  params     A: normalised loss L (MSE in train-fit log+z units), per target,
             and its robust twin medae (median |error|, same units)
             B, C: the same, plus bias and s.d. in those units and the
             relative bias / log-RMSE of the natural parameter -- at a fixed
             theta, "how wrong on average" and "how noisy" are separate
             questions and a pooled MSE hides which one a method fails.
  classify   accuracy and per-class recall everywhere; on B and C every cell
             is one family, so its accuracy IS that family's recall there.
  detect     power = P(score > threshold) with the threshold calibrated on
             half the exact-CSR anchors of C at the matching nbar, and the
             realised size checked on the other half -- every method is
             compared at the same empirical 5% false-alarm rate, not at
             whatever its own nominal threshold happens to deliver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .dv3 import DELTA_EDGES, NBAR_EDGES, Regimes, band_labels

DELTA_LABELS = band_labels(DELTA_EDGES)
NBAR_LABELS = band_labels(NBAR_EDGES, fmt="{:.0f}")

# A cell with fewer patterns than this is reported but flagged: its per-cell
# numbers are too noisy to rank methods on.
MIN_ROWS = 20


# ---------------------------------------------------------------------------
# Row groupings
# ---------------------------------------------------------------------------

def _coords(regimes: Regimes, rows: np.ndarray) -> dict[str, Any]:
    """Descriptive coordinates of a group: the constant ones exactly, the
    varying ones as their median, so a B cell reports its design theta and an
    A band reports where its mass actually sits."""
    out: dict[str, Any] = {"n_rows": int(len(rows))}
    for key in ("nbar", "delta_tilde", "scale", "amplitude", "tau_K"):
        vals = np.asarray(regimes[key][rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        out[key] = float(vals[0]) if np.ptp(vals) == 0 else float(np.median(vals))
    return out


def groupings(regimes: Regimes, set_: str) -> dict[str, np.ndarray]:
    """{row_key -> row indices} for every regime grouping that makes sense on
    `set_`. Indices address the rows of `regimes` (already aligned to the
    prediction bundle by Regimes.select)."""
    groups: dict[str, np.ndarray] = {}
    families = [str(f) for f in dict.fromkeys(regimes["family"])]
    for family in families:
        fam_rows = np.flatnonzero(regimes["family"] == family)
        if set_ in ("A", "train"):
            groups[f"{set_}/{family}/all"] = fam_rows
            if family == "poisson":
                # delta is identically 0 for CSR: only the nbar split carries information.
                nb = regimes.nbar_band()[fam_rows]
                for b, label in enumerate(NBAR_LABELS):
                    groups[f"{set_}/{family}/nbar={label}"] = fam_rows[nb == b]
                continue
            db = regimes.delta_band()[fam_rows]
            nb = regimes.nbar_band()[fam_rows]
            for b, label in enumerate(DELTA_LABELS):
                groups[f"{set_}/{family}/delta={label}"] = fam_rows[db == b]
            for b, label in enumerate(NBAR_LABELS):
                groups[f"{set_}/{family}/nbar={label}"] = fam_rows[nb == b]
            for bd, dlabel in enumerate(DELTA_LABELS):
                for bn, nlabel in enumerate(NBAR_LABELS):
                    groups[f"{set_}/{family}/delta={dlabel}/nbar={nlabel}"] = fam_rows[(db == bd) & (nb == bn)]
        elif set_ == "B":
            for cell in np.unique(regimes["cell_id"][fam_rows]):
                groups[f"B/{family}/cell={int(cell)}"] = fam_rows[regimes["cell_id"][fam_rows] == cell]
        elif set_ == "C":
            if family == "poisson":
                for nbar in np.unique(regimes["nbar"][fam_rows]):
                    groups[f"C/poisson/nbar={nbar:g}"] = fam_rows[regimes["nbar"][fam_rows] == nbar]
                continue
            cells = regimes["cell_id"][fam_rows]
            levels = regimes["level_id"][fam_rows]
            for cell in np.unique(cells):
                for level in np.unique(levels[cells == cell]):
                    groups[f"C/{family}/ladder={int(cell)}/level={int(level)}"] = (
                        fam_rows[(cells == cell) & (levels == level)]
                    )
        else:
            raise ValueError(f"unknown DV3 set {set_!r}")
    return {k: v for k, v in groups.items() if len(v)}


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------

def params_metrics(
    pred_std: np.ndarray,
    truth_std: np.ndarray,
    label_names: Sequence[str],
    pred_raw: np.ndarray | None = None,
    truth_raw: np.ndarray | None = None,
) -> dict[str, Any]:
    """Every per-group number for parameter estimation. Rows with any
    non-finite prediction count as failures (fail_rate) and are excluded
    from the error moments -- a method that silently fails on the hardest
    cells must not look better there for it."""
    ok = np.isfinite(pred_std).all(axis=1)
    n, n_ok = len(ok), int(ok.sum())
    out: dict[str, Any] = {"n": int(n), "n_ok": n_ok, "fail_rate": float(1 - n_ok / n) if n else float("nan")}
    if n_ok == 0:
        out["loss"] = float("nan")
        return out

    err = pred_std[ok] - truth_std[ok]
    mse_t = (err ** 2).mean(axis=0)
    out["loss"] = float(mse_t.mean())
    out["loss_per_target"] = dict(zip(label_names, mse_t.tolist()))
    out["bias_std"] = dict(zip(label_names, err.mean(axis=0).tolist()))
    # Robust companion to the MSE: a classical fit that diverges near CSR
    # (kappa -> 0, sigma -> 1e7) dominates a mean of squares on its own, which
    # hides whether the method is otherwise accurate there. The median
    # absolute error (same units) is insensitive to those few rows.
    out["medae_std"] = dict(zip(label_names, np.median(np.abs(err), axis=0).tolist()))
    out["medae"] = float(np.median(np.abs(err), axis=0).mean())
    out["sd_std"] = dict(zip(label_names, (pred_std[ok].std(axis=0, ddof=1) if n_ok > 1
                                           else np.full(err.shape[1], np.nan)).tolist()))

    if pred_raw is not None and truth_raw is not None:
        pr, tr = pred_raw[ok], truth_raw[ok]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = pr / tr
        good = np.isfinite(ratio) & (ratio > 0)
        rel_bias, log_rmse = [], []
        for j in range(ratio.shape[1]):
            r = ratio[good[:, j], j]
            rel_bias.append(float(np.median(r) - 1.0) if len(r) else float("nan"))
            log_rmse.append(float(np.sqrt(np.mean(np.log(r) ** 2))) if len(r) else float("nan"))
        out["rel_bias"] = dict(zip(label_names, rel_bias))
        out["log_rmse"] = dict(zip(label_names, log_rmse))
    return out


def params_tables(bundle: dict[str, Any], regimes: Regimes) -> dict[str, dict[str, Any]]:
    set_ = bundle["set"]
    names = [str(x) for x in bundle["label_names"]]
    pred_std, truth_std = bundle["pred_std"], bundle["truth_std"]
    pred_raw, truth_raw = bundle.get("pred_raw"), bundle.get("truth_raw")
    tables: dict[str, dict[str, Any]] = {}
    for key, rows in groupings(regimes, set_).items():
        metrics = params_metrics(
            pred_std[rows], truth_std[rows], names,
            None if pred_raw is None else pred_raw[rows],
            None if truth_raw is None else truth_raw[rows],
        )
        tables[key] = {"coords": _coords(regimes, rows), "metrics": metrics,
                       "small": bool(len(rows) < MIN_ROWS)}
    return tables


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_metrics(proba: np.ndarray, truth: np.ndarray, class_names: Sequence[str]) -> dict[str, Any]:
    pred = proba.argmax(axis=1)
    n = len(truth)
    out: dict[str, Any] = {"n": int(n), "accuracy": float((pred == truth).mean()) if n else float("nan")}
    recall = {}
    for c, name in enumerate(class_names):
        m = truth == c
        if m.any():
            recall[name] = float((pred[m] == c).mean())
    out["recall"] = recall
    out["pred_share"] = {name: float((pred == c).mean()) for c, name in enumerate(class_names)} if n else {}
    # mean log-loss of the true class: a calibration-sensitive companion to
    # accuracy, and the quantity a proper-scoring comparison would rank on.
    p_true = np.clip(proba[np.arange(n), truth], 1e-12, 1.0) if n else np.array([])
    out["log_loss"] = float(-np.log(p_true).mean()) if n else float("nan")
    if "poisson" in class_names:
        # "Called non-CSR at the argmax" -- the classifier's own, uncalibrated
        # detection decision. Calibrated power lives in detection_tables.
        csr = list(class_names).index("poisson")
        out["called_structured"] = float((pred != csr).mean()) if n else float("nan")
    return out


def confusion(proba: np.ndarray, truth: np.ndarray, n_classes: int) -> list[list[int]]:
    pred = proba.argmax(axis=1)
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(mat, (truth, pred), 1)
    return mat.tolist()


def classify_tables(bundle: dict[str, Any], regimes: Regimes) -> dict[str, dict[str, Any]]:
    set_ = bundle["set"]
    names = [str(x) for x in bundle["class_names"]]
    proba, truth = bundle["proba"], bundle["truth"].astype(np.int64)
    tables: dict[str, dict[str, Any]] = {}
    for key, rows in groupings(regimes, set_).items():
        tables[key] = {"coords": _coords(regimes, rows), "metrics": classify_metrics(proba[rows], truth[rows], names),
                       "small": bool(len(rows) < MIN_ROWS)}
    # Whole-set rows pooling all families -- the only place a multi-family
    # "overall accuracy" is meaningful (A is balanced, 5,000 per family, so
    # this is the headline number; on B/C the family mix is a design artefact).
    all_rows = np.arange(len(truth))
    tables[f"{set_}/_all"] = {
        "coords": _coords(regimes, all_rows),
        "metrics": {**classify_metrics(proba, truth, names), "confusion": confusion(proba, truth, len(names))},
        "small": False,
    }
    return tables


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@dataclass
class Threshold:
    """A per-nbar (1 - alpha) threshold, linear in log nbar between the CSR
    anchor sizes and held flat outside them (the anchors are 125/250/500;
    A reaches 100-800, so its extremes are a clamped extrapolation, recorded
    in `extrapolated` rather than hidden)."""

    nbar: np.ndarray
    value: np.ndarray
    alpha: float
    size: dict[str, float]            # realised size on the held-out anchor half, per nbar
    size_se: dict[str, float]
    n_calib: dict[str, int]

    def __call__(self, nbar: np.ndarray) -> np.ndarray:
        nbar = np.asarray(nbar, dtype=float)
        return np.interp(np.log(nbar), np.log(self.nbar), self.value)

    def extrapolated(self, nbar: np.ndarray) -> np.ndarray:
        nbar = np.asarray(nbar, dtype=float)
        return (nbar < self.nbar.min()) | (nbar > self.nbar.max())

    def to_json(self) -> dict[str, Any]:
        return {"nbar": self.nbar.tolist(), "value": self.value.tolist(), "alpha": self.alpha,
                "size_heldout": self.size, "size_heldout_se": self.size_se, "n_calib": self.n_calib}


def calibrate_threshold(
    score: np.ndarray, regimes: Regimes, alpha: float = 0.05,
) -> Threshold:
    """Calibrate on the exact-CSR anchors (C/poisson): even replicates set the
    (1 - alpha) quantile at each anchor nbar, odd replicates measure the size
    that threshold actually delivers. Same fit/held-out halving nulls.py uses
    for c95, for the same reason: a threshold and its size check drawn from
    the same patterns would report exactly alpha by construction."""
    csr = regimes["family"] == "poisson"
    if not csr.any():
        raise ValueError("no CSR anchors (C/poisson) among these rows -- cannot calibrate a detector")
    nbars = np.unique(regimes["nbar"][csr])
    values, size, size_se, n_calib = [], {}, {}, {}
    for nb in nbars:
        at = csr & (regimes["nbar"] == nb)
        reps = regimes["rep"][at]
        s = score[at]
        fit, held = s[reps % 2 == 0], s[reps % 2 == 1]
        fit, held = fit[np.isfinite(fit)], held[np.isfinite(held)]
        # "higher" quantile: the smallest observed score with at most alpha
        # of the calibration half strictly above it (conservative, no interpolation).
        thr = float(np.quantile(fit, 1 - alpha, method="higher"))
        values.append(thr)
        p = float((held > thr).mean())
        size[f"{nb:g}"] = p
        size_se[f"{nb:g}"] = float(np.sqrt(p * (1 - p) / max(len(held), 1)))
        n_calib[f"{nb:g}"] = int(len(fit))
    return Threshold(nbar=nbars.astype(float), value=np.asarray(values), alpha=alpha,
                     size=size, size_se=size_se, n_calib=n_calib)


def detection_metrics(
    score: np.ndarray, nbar: np.ndarray, threshold: Threshold, nominal: float | None = None,
) -> dict[str, Any]:
    ok = np.isfinite(score)
    n = len(score)
    out: dict[str, Any] = {"n": int(n), "fail_rate": float(1 - ok.mean()) if n else float("nan")}
    if not ok.any():
        out["power"] = float("nan")
        return out
    # A non-finite score is "did not reject": a test that cannot be computed
    # on a pattern has no power on it.
    reject = ok & (score > threshold(nbar))
    p = float(reject.mean())
    out["power"] = p
    out["power_se"] = float(np.sqrt(p * (1 - p) / n))
    out["extrapolated_threshold"] = float(threshold.extrapolated(nbar).mean())
    out["score_median"] = float(np.median(score[ok]))
    if nominal is not None:
        out["power_nominal"] = float((ok & (score > nominal)).mean())
    return out


def detection_tables(
    score: np.ndarray, regimes: Regimes, set_: str, threshold: Threshold, nominal: float | None = None,
) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for key, rows in groupings(regimes, set_).items():
        if set_ == "C" and key.startswith("C/poisson/"):
            # On the anchors, report the held-out size, never the in-sample
            # rejection rate (which is alpha by construction on the fit half).
            rows_held = rows[regimes["rep"][rows] % 2 == 1]
            m = detection_metrics(score[rows_held], regimes["nbar"][rows_held], threshold, nominal)
            m["is_size"] = True
        else:
            m = detection_metrics(score[rows], regimes["nbar"][rows], threshold, nominal)
        tables[key] = {"coords": _coords(regimes, rows), "metrics": m, "small": bool(len(rows) < MIN_ROWS)}
    return tables


def classifier_detection_score(proba: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    """Evidence against CSR from a family classifier: 1 - P(poisson).
    Monotone in the posterior odds of "structured", so thresholding it at a
    calibrated level is the classifier's most powerful CSR-vs-rest test for
    the prior it was trained on."""
    names = list(class_names)
    if "poisson" not in names:
        raise ValueError("classifier has no 'poisson' class -- it cannot be used as a CSR detector")
    return 1.0 - np.asarray(proba)[:, names.index("poisson")]


# ---------------------------------------------------------------------------
# One entry point
# ---------------------------------------------------------------------------

def tables_for_bundle(
    bundle: dict[str, Any],
    regimes: Regimes,
    threshold: Threshold | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """{"params"|"classify"|"detect": {row_key: {...}}} for one bundle.
    `threshold` enables detection tables (for classifiers and raw scores);
    it must have been calibrated with the same method+seed's C anchors."""
    task = bundle["task"]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    if task == "params":
        out["params"] = params_tables(bundle, regimes)
    elif task == "classify":
        out["classify"] = classify_tables(bundle, regimes)
        if threshold is not None:
            score = classifier_detection_score(bundle["proba"], bundle["class_names"])
            out["detect"] = detection_tables(score, regimes, bundle["set"], threshold)
    elif task == "detect":
        if threshold is None:
            raise ValueError("a detect bundle needs a threshold (calibrate on the same method's C anchors)")
        nominal = bundle["meta"].get("nominal_threshold")
        out["detect"] = detection_tables(bundle["score"], regimes, bundle["set"], threshold, nominal)
    else:
        raise ValueError(f"unknown prediction task {task!r}")
    return out


def detection_score_of(bundle: dict[str, Any]) -> np.ndarray | None:
    if bundle["task"] == "classify":
        names = [str(x) for x in bundle["class_names"]]
        return classifier_detection_score(bundle["proba"], names) if "poisson" in names else None
    if bundle["task"] == "detect":
        return bundle["score"]
    return None
