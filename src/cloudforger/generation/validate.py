# src/cloudforger/generation/validate.py
"""Validation batches (generation.tex, "Validation protocol"; set 9 streams).

    V1  mean K-hat (Ripley isotropic, TRUE lambda^2 = nbar^2) vs the closed-form
        K that delta-tilde uses, on R = [0.01, 0.25]. The studentised sup
        max_r |mean - K| / se is compared with its 99% point under a Gaussian
        with the replicates' correlation (a Gaussian bootstrap), so the test
        accounts for the 490 correlated radii.
    V4  intensity in the band within w = max(R, 2 sigma) of the window edge vs
        the interior: |ratio - 1| <= max(3 s.e., 1%). Catches a missing buffer
        (Thomas, nested) or thinning done inside W only (Matern II).

Simulation cost is small; the point is that a wrong closed form (which would
silently corrupt delta-tilde) or a wrong sampler cannot both pass V1.
"""

from __future__ import annotations

from multiprocessing import get_context
from typing import Any

import numpy as np

from ..baselines.vihrs import _isotropic_l_minus_r
from .nulls import L_RMIN, R_GRID, load_tables
from .plan import complete
from .prior import build_priors, fixed
from .samplers import simulate
from .seeding import PATTERN, case_rng
from .spec import load_spec

# One mid-prior theta per family at nbar = 250 (cell ids are part of the stream index).
CELLS: dict[str, tuple[int, float, dict[str, float]]] = {
    "poisson": (0, 250.0, {}),
    "thomas":  (1, 250.0, {"mu": 4.0, "s": 0.4}),
    "nested":  (2, 250.0, {"mu1": 3.0, "mu2": 2.0, "s2": 0.2, "rho": 6.0}),
    "matern2": (3, 250.0, {"tau": 0.3}),
    "lgcp":    (4, 250.0, {"sigma2": 1.0, "sp": 1.0}),
}


def _band_width(model: dict[str, float]) -> float:
    if "R" in model:
        return model["R"]
    if "sigma1" in model:
        return 2.0 * float(np.hypot(model["sigma1"], model["sigma"]))
    return 2.0 * model.get("sigma", 0.0)


def _replicates(args: tuple[str, str, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """K-hat with true lambda^2 on R_GRID, and (band, interior) counts, for reps [start, stop)."""
    spec_path, family, start, stop = args
    spec = load_spec(spec_path)
    prior = build_priors(spec)[family]
    cell, nbar, shape = CELLS[family]
    d = complete(prior, fixed(prior, nbar, shape), load_tables(spec.null_tables_path))
    w = max(_band_width(d.model), 0.02)
    K, counts = [], []
    for rep in range(start, stop):
        pts = simulate(family, d, case_rng(spec.root, "validation", family, 10_000 * cell + rep, PATTERN)).points
        n = len(pts)
        lmr = _isotropic_l_minus_r(pts, [0, 0], [1, 1], R_GRID)
        K.append(np.pi * (lmr + R_GRID) ** 2 * n * (n - 1) / nbar**2 if n >= 2 else np.zeros(len(R_GRID)))
        edge = np.minimum(pts, 1.0 - pts).min(axis=1) < w if n else np.zeros(0, bool)
        counts.append((edge.sum(), n - edge.sum()))
    return np.array(K), np.array(counts, float)


def v1(K_reps: np.ndarray, K_true: np.ndarray, rng: np.random.Generator, n_boot: int = 4000) -> dict[str, Any]:
    sel = R_GRID >= L_RMIN
    X, K = K_reps[:, sel], K_true[sel]
    se = X.std(axis=0, ddof=1) / np.sqrt(len(X))
    # Radii where every replicate is identical (K-hat = 0 inside a hard core)
    # cannot be studentised; there the closed form must agree exactly.
    fixed_ok = bool(np.all(np.abs(X.mean(axis=0) - K)[se == 0] <= 1e-12))
    X, K, se = X[:, se > 0], K[se > 0], se[se > 0]
    t = float(np.max(np.abs(X.mean(axis=0) - K) / se))
    corr = np.corrcoef(X, rowvar=False)
    vals, vecs = np.linalg.eigh(corr)
    Z = rng.standard_normal((n_boot, len(vals))) * np.sqrt(np.clip(vals, 0, None)) @ vecs.T
    q99 = float(np.quantile(np.abs(Z).max(axis=1), 0.99))
    return {"sup_z": t, "q99": q99, "n_zero_var_radii": int(len(sel) - len(K) - (~sel).sum()),
            "pass": t <= q99 and fixed_ok}


def v4(counts: np.ndarray, w: float, rng: np.random.Generator, n_boot: int = 2000) -> dict[str, Any]:
    area_band = 1.0 - (1.0 - 2.0 * w) ** 2

    def ratio(c: np.ndarray) -> float:
        return float((c[:, 0].sum() / area_band) / (c[:, 1].sum() / (1.0 - area_band)))

    r = ratio(counts)
    boot = [ratio(counts[rng.integers(0, len(counts), len(counts))]) for _ in range(n_boot)]
    se = float(np.std(boot, ddof=1))
    return {"band_width": w, "ratio": r, "se": se, "pass": abs(r - 1.0) <= max(3.0 * se, 0.01)}


def run(spec_path: str, reps: int = 400, jobs: int = 1, log=print) -> dict[str, Any]:
    spec = load_spec(spec_path)
    priors = build_priors(spec)
    tabs = load_tables(spec.null_tables_path)
    chunk = max(1, reps // max(jobs, 1))
    report: dict[str, Any] = {"reps": reps}
    rng = case_rng(spec.root, "validation", "poisson", 99_999_999, 0)
    for family in spec.families:
        if family not in CELLS:
            continue
        tasks = [(spec_path, family, s, min(s + chunk, reps)) for s in range(0, reps, chunk)]
        if jobs > 1:
            with get_context("spawn").Pool(jobs) as pool:
                parts = pool.map(_replicates, tasks)
        else:
            parts = [_replicates(t) for t in tasks]
        K = np.concatenate([p[0] for p in parts])
        counts = np.concatenate([p[1] for p in parts])
        _, nbar, shape = CELLS[family]
        d = complete(priors[family], fixed(priors[family], nbar, shape), tabs)
        out = {"nbar": nbar, **shape, "delta_tilde": d.regime["delta_tilde"],
               "V1": v1(K, priors[family].K(R_GRID, nbar, shape), rng)}
        if family in ("thomas", "nested", "matern2"):
            out["V4"] = v4(counts, max(_band_width(d.model), 0.02), rng)
        report[family] = out
        log(f"  {family:8s} delta~={out['delta_tilde']:.2f}  V1 sup z {out['V1']['sup_z']:.2f} "
            f"(99% point {out['V1']['q99']:.2f}) {'PASS' if out['V1']['pass'] else 'FAIL'}"
            + (f"  V4 ratio {out['V4']['ratio']:.4f} +- {out['V4']['se']:.4f} "
               f"{'PASS' if out['V4']['pass'] else 'FAIL'}" if "V4" in out else ""))
    report["pass"] = all(v[k]["pass"] for v in report.values() if isinstance(v, dict) for k in ("V1", "V4") if k in v)
    return report


# --- V1 on the generated B cells (gate D3) -------------------------------------------

def _khat_task(args: tuple[list[np.ndarray], list[float]]) -> np.ndarray:
    clouds, nbars = args
    out = np.zeros((len(clouds), len(R_GRID)))
    for i, (pts, nbar) in enumerate(zip(clouds, nbars)):
        n = len(pts)
        if n >= 2:
            out[i] = np.pi * (_isotropic_l_minus_r(pts, [0, 0], [1, 1], R_GRID) + R_GRID) ** 2 * n * (n - 1) / nbar**2
    return out


def v1_cells(spec_path: str, root: str, set_: str = "B", jobs: int = 1, log=print) -> dict[str, Any]:
    """Per cell: mean K-hat (true lambda^2) vs the closed form. Pass: the number of
    cells beyond their 99% point is within the 99% quantile of Bin(#cells, 0.01),
    and none is beyond a Bonferroni bound on the 99.99% point."""
    from scipy.stats import binom, norm

    from .design import cell_shape, read_cells
    from .store import DV3Paths, load_points, read_csv

    spec = load_spec(spec_path)
    priors = build_priors(spec)
    paths = DV3Paths(root)
    rng = case_rng(spec.root, "validation", "poisson", 99_999_998, 0)
    n_radii = int((R_GRID >= L_RMIN).sum())
    extreme = float(norm.isf(1e-4 / (2 * n_radii)))
    cells = [c for c in read_cells(spec.cells_path) if c["set"] == set_ and c["feasible"]]
    report: dict[str, Any] = {"set": set_, "bonferroni_99.99": extreme}
    for family in spec.families:
        mine = [c for c in cells if c["family"] == family]
        if not mine:
            continue
        clouds, _ = load_points(paths.points(set_, family))
        manifest = read_csv(paths.manifest(set_, family))
        step = max(1, len(clouds) // (4 * max(jobs, 1)))
        tasks = [(clouds[i:i + step], [r["nbar"] for r in manifest[i:i + step]]) for i in range(0, len(clouds), step)]
        if jobs > 1:
            with get_context("spawn").Pool(jobs) as pool:
                K = np.concatenate(pool.map(_khat_task, tasks))
        else:
            K = np.concatenate([_khat_task(t) for t in tasks])
        cell_of = np.array([r["cell_id"] for r in manifest])
        per_cell = []
        for c in mine:
            res = v1(K[cell_of == c["cell_id"]], priors[family].K(R_GRID, c["nbar"], cell_shape(c, priors[family])), rng)
            per_cell.append({"cell_id": c["cell_id"], "sup_z": res["sup_z"], "q99": res["q99"]})
        n_fail = sum(p["sup_z"] > p["q99"] for p in per_cell)
        n_extreme = sum(p["sup_z"] > extreme for p in per_cell)
        allowed = int(binom.ppf(0.99, len(per_cell), 0.01))
        report[family] = {"n_cells": len(per_cell), "n_beyond_q99": int(n_fail), "allowed": allowed,
                          "n_extreme": int(n_extreme), "max_sup_z": max(p["sup_z"] for p in per_cell),
                          "pass": n_fail <= allowed and n_extreme == 0, "cells": per_cell}
        log(f"  {set_}/{family:8s} {len(per_cell):3d} cells: {n_fail} beyond their 99% point (allowed {allowed}), "
            f"{n_extreme} extreme, max sup z {report[family]['max_sup_z']:.2f}  "
            f"{'PASS' if report[family]['pass'] else 'FAIL'}")
    report["pass"] = all(v["pass"] for v in report.values() if isinstance(v, dict))
    return report
