#!/usr/bin/env python3
"""Minimum-contrast baseline for the Thomas process -> results/params/thomas/K/mincontrast/seed_0/.

    python scripts/mincontrast.py

The classical arm of the parameter task, with no learning in it: for each TEST pattern, Ripley's K
(cloudforger.classical.lfunction, the same estimator the L curves use) is matched to the Thomas
model by the usual power-transformed contrast

    min_{kappa,sigma}  sum_r (K_hat(r)^c - K_thomas(r; kappa, sigma))^2,   c = 1/4, r <= 0.25

with mu_hat = n / kappa_hat on the unit window. The search is bounded by what a unit window can
resolve (sigma <= 0.25 = r_max, kappa in [0.1, 1e4]): unbounded, the near-CSR patterns run off along
the flat direction where only kappa sigma^2 is identified and land at kappa ~ 1e-15. `converged`
records whether every restart converged away from that box, and is saved per pattern -- failure near
the Poisson limit is a property of the estimator worth reporting, not something to hide.

Deterministic, so one seed is the whole arm. Writes the same predictions.npz / run.json contract as
scripts/train.py, so scripts/regimes.py picks it up like any other run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.classical.lfunction import RADII, k_function   # noqa: E402
from cloudforger.featurization.sweep import BANK                # noqa: E402
from cloudforger.provenance import provenance_stamp             # noqa: E402
from cloudforger.simulation.split import split_of               # noqa: E402

TARGETS = ["kappa", "mu", "sigma"]
C = 0.25                                            # contrast exponent
BOX = ((np.log(0.1), np.log(1e4)), (np.log(1e-3), np.log(RADII[-1])))
STARTS = ((3.0, -3.5), (4.5, -2.5), (6.0, -4.5))    # (log kappa, log sigma)


def fit(k_hat: np.ndarray) -> tuple[float, float, bool]:
    """(kappa, sigma, converged) for one pattern's empirical K on the RADII grid."""
    target = np.maximum(k_hat, 0.0) ** C

    def contrast(x: np.ndarray) -> float:
        kappa, sigma = np.exp(x)
        k = np.pi * RADII**2 + (1.0 - np.exp(-(RADII**2) / (4.0 * sigma**2))) / kappa
        return float(np.sum((target - k**C) ** 2))

    best, ok = None, True
    for x0 in STARTS:
        res = minimize(contrast, x0, method="Nelder-Mead", bounds=BOX,
                       options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 2000})
        ok &= bool(res.success)
        if best is None or res.fun < best.fun:
            best = res
    interior = all(abs(v - lo) > 1e-3 and abs(v - hi) > 1e-3 for v, (lo, hi) in zip(best.x, BOX))
    kappa, sigma = np.exp(best.x)
    return float(kappa), float(sigma), ok and interior


def main() -> None:
    manifest = pd.read_csv(BANK / "thomas" / "manifest.csv")
    test = np.flatnonzero(split_of(manifest["theta"].to_numpy()) == "test")
    z = np.load(BANK / "thomas" / "points.npz")
    points, offsets = z["points"], z["offsets"]
    if len(offsets) - 1 != len(manifest):
        raise SystemExit(f"thomas: {len(offsets) - 1} patterns for {len(manifest)} manifest rows")

    pred, converged = np.empty((len(test), 3)), np.empty(len(test), bool)
    for row, i in enumerate(test):
        kappa, sigma, ok = fit(k_function(points[offsets[i]:offsets[i + 1]]))
        n = float(manifest["n"].iat[i])
        pred[row], converged[row] = (kappa, n / kappa, sigma), ok
        if (row + 1) % 500 == 0:
            print(f"{row + 1}/{len(test)} patterns", flush=True)

    out = ROOT / "results" / "params" / "thomas" / "K" / "mincontrast" / "seed_0"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "predictions.npz",
             case_id=manifest["case_id"].to_numpy(str)[test],
             y_true=manifest[TARGETS].to_numpy(float)[test], y_pred=pred, converged=converged)
    (out / "run.json").write_text(json.dumps({
        "args": {"task": "params", "family": "thomas", "seed": 0, "c": C, "r_max": float(RADII[-1])},
        "families": ["thomas"], "n_test": len(test),
        "targets": {"columns": TARGETS, "log": [True] * 3},
        "converged": float(converged.mean()), "provenance": provenance_stamp(),
    }, indent=2, default=float))
    print(f"wrote {out}  ({converged.mean():.1%} converged)")


if __name__ == "__main__":
    main()
