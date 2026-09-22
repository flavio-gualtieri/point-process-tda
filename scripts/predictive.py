#!/usr/bin/env python3
"""Predictive checks for the parameter runs: results/params/**/predictions.npz -> predictive.npz.

    python scripts/predictive.py --tier 1                          # every params run, seconds
    python scripts/predictive.py --tier 2 --glob 'thomas/L/fixed/seed_*' --jobs 8

Both tiers score ONE TEST PATTERN AT A TIME and are written beside the predictions they came from,
so scripts/regimes.py joins them to the manifest and bins, bootstraps and pairs them exactly like
rmse_log -- nothing downstream knows these are new.

tier 1  `dtilde`: the closed-form distance between the fitted and the true process, in the units of
        the sweep's own regime coordinate,

            D = max_r |L_hat(r) - L_theta(r)| / s0(r; nbar) / c95(nbar),

        i.e. departure.tables.delta_tilde with the TRUE model's L(r) - r in place of the CSR line
        that delta-tilde measures from. D = 1 means the fitted and the true process differ by as
        much as it takes to reject CSR at 5%; D = 0.2 means five times less than that. Both curves
        are the closed form in simulation.families, so there is no simulation and this runs over
        every run in seconds.

        It is invariant to how theta was parameterised, which pooled log-RMSE is not: Thomas carries
        kappa = nbar/mu with nbar readable from log n, so log kappa's error is minus log mu's and
        the pooled score spends two thirds of its weight on one unknown. The amount of that padding
        differs per family (LGCP has nbar as a target outright, Matern's lam_p is fixed by nbar and
        R), so the current columns are not comparable across families and D is.

        Poisson has no second-order structure, so D is identically 0 there -- correctly: its L-curve
        does not depend on its parameter at all. Poisson's error is the count, already reported as
        rmse_log:nbar.

tier 2  `envelope_p`, `envelope_reject`: the check a practitioner would run on the fitted output.
        Simulate B patterns from theta_hat, compute L, F, G, J for each, and rank the observed
        pattern among them under a combined studentised global envelope (Myllymaki et al. 2017,
        Mrkvicka et al. 2017 for the combination):

            M_i = max over (function, radius) of |c_i - mean_j c_j| / sd_j c_j,
            p   = (1 + #{i >= 1 : M_i >= M_0}) / (B + 1).

        The moments are taken over the observed pattern AND the B simulations together, so the B+1
        statistics are exchangeable under theta_hat = theta and p is exactly uniform on the grid
        {1/(B+1), ..., 1}. A flat p-histogram therefore means the fitted parameters reproduce the
        data; a spike at 0 means they do not. `envelope_reject` is 1[p <= alpha], so its cell mean
        IS the rejection rate and reads against the nominal alpha.

        All four functions, not L alone, because the classical arm is trained on L and grading it on
        L alone is not a neutral referee. F, G and J cost 1.5x L together, so this is nearly free.
        Columns where the B+1 curves carry no spread (small r before any pair, the saturated tail of
        F and G) are dropped by a symmetric rule that cannot break the exchangeability.

        Read it as model ADEQUACY, not accuracy: where theta is weakly identified -- every low-delta
        cell, nested's mu1 -- a badly wrong theta_hat still passes, because the data cannot tell.
        That gap against tier 1 is the point of running both.

        Cost is B simulations + 4 curves per pattern (0.25-2.0 s at B = 99, LGCP the slow one), so
        --thetas subsamples. The subsample is the first N test thetas, which is a random sample (the
        thetas are i.i.d. prior draws) and is identical across runs, so regimes.py can still pair
        two runs on the same thetas. Untouched patterns are written as NaN and skipped there.

        The B simulations are NOT conditioned on n in the sweep's [20, 2000] window as the observed
        patterns were. That filter never fired: `pattern_tries` is 1 for all 100 000 patterns in the
        five manifests, so the conditioning is vacuous over this prior and costs the rank test
        nothing, while a theta_hat implying implausible counts is still penalised.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.classical.curves import Config as CurvesConfig, DATA as CLASSICAL  # noqa: E402
from cloudforger.classical.functions import NAMES, curves                           # noqa: E402
from cloudforger.classical.lfunction import RADII, l_minus_r_from_excess            # noqa: E402
from cloudforger.departure.tables import Tables                                     # noqa: E402
from cloudforger.simulation.families import FAMILIES, Rules                         # noqa: E402
from cloudforger.simulation.lgcp_grid import grid_size                              # noqa: E402
from cloudforger.simulation.processes import SAMPLERS, lgcp_eigenvalues             # noqa: E402

RESULTS = ROOT / "results" / "params"
GRID = {"name": "fixed"}            # the observed curves in data/classical/<family>/fixed
ALPHA, B_DEFAULT, THETAS_DEFAULT = 0.05, 99, 400


# ---------------------------------------------------------------- theta -> the two model views

def excess_args(family: str, p: dict) -> tuple[float, dict, float | None]:
    """Model parameters -> (nbar, shape, amp), the arguments Family.excess takes.

    The inverse of Family.model. Matern's nbar is the intensity left by the type II thinning,
    (1 - exp(-lam_p pi R^2)) / (pi R^2), which is always positive and always inside the packing
    bound, so a wild theta_hat cannot push it outside the closed form's domain."""
    if family == "poisson":
        return p["nbar"], {}, None
    if family == "thomas":
        return p["kappa"] * p["mu"], {"sigma": p["sigma"]}, p["mu"]
    if family == "nested":
        return (p["kappa"] * p["mu1"] * p["mu2"],
                {"sigma2": p["sigma2"], "rho": p["sigma1"] / p["sigma2"], "mu1": p["mu1"]},
                p["mu2"])
    if family == "lgcp":
        return p["nbar"], {"s": p["s"]}, p["sigma2"]
    a = math.pi * p["R"] ** 2
    return -math.expm1(-p["lam_p"] * a) / a, {}, p["R"]


def sampler_args(family: str, p: dict, tables: Tables) -> dict | None:
    """Model parameters -> the keyword arguments of processes.SAMPLERS[family].

    The target columns already ARE those keywords for every family but LGCP, which is trained on
    nbar (mu_log is the one target that can go negative) and needs a grid size. `None` when no grid
    below lgcp_grid's cap discretises this theta_hat finely enough."""
    if family != "lgcp":
        return dict(p)
    M = grid_size(p["sigma2"], p["s"], p["nbar"], tables)
    return None if M is None else {"mu_log": math.log(p["nbar"]) - p["sigma2"] / 2,
                                   "sigma2": p["sigma2"], "s": p["s"], "M": M}


# ---------------------------------------------------------------- tier 1

def model_curves(fam, family: str, rows: list[dict]) -> np.ndarray:
    """(N, 512) closed-form L(r) - r, one row per parameter vector."""
    out = np.empty((len(rows), len(RADII)))
    for i, p in enumerate(rows):
        out[i] = l_minus_r_from_excess(fam.excess(RADII, *excess_args(family, p)))
    return out


def tier1(family: str, true_rows: list[dict], pred_rows: list[dict], tables: Tables) -> np.ndarray:
    """D(theta_hat, theta), studentised at the TRUE nbar -- the noise scale of the data being
    scored, and the one guaranteed to sit inside the null tables' range."""
    fam = FAMILIES[family](Rules.load())
    nbar = np.array([excess_args(family, p)[0] for p in true_rows])
    gap = model_curves(fam, family, pred_rows) - model_curves(fam, family, true_rows)
    return tables.delta_tilde(gap, nbar)


# ---------------------------------------------------------------- tier 2

def envelope_p(block: np.ndarray) -> float:
    """Rank p of row 0 among the rest under a combined studentised global envelope.

    `block` is (B+1, K, 512): the observed curves first, then one simulation per row. Moments use
    all B+1 rows and the kept-column rule is a function of the whole set, so every row is treated
    identically and the rank is exactly uniform when the simulations come from the true theta."""
    m, s = block.mean(axis=0), block.std(axis=0, ddof=1)
    keep = s > 1e-6 * s.max(axis=1, keepdims=True)        # per function: drop the flat columns
    if not keep.any():
        return math.nan
    z = np.where(keep, np.abs(block - m) / np.where(keep, s, 1.0), 0.0)
    M = z.reshape(len(block), -1).max(axis=1)
    return float(1 + np.count_nonzero(M[1:] >= M[0])) / len(block)


@lru_cache(maxsize=1)
def _worker_state():
    return Tables(), CurvesConfig.load().f_grid_size


def _task(args) -> float:
    """One test pattern: simulate B from theta_hat, rank the observed curves among them."""
    family, pred, observed, seed, n_sims = args
    tables, f_grid = _worker_state()
    model = sampler_args(family, pred, tables)
    if model is None:
        return math.nan
    if family == "lgcp":
        model["root_lam"] = lgcp_eigenvalues(model["sigma2"], model["s"], model["M"])

    rng = np.random.default_rng(seed)
    block = np.empty((n_sims + 1, len(NAMES), len(RADII)))
    block[0] = observed
    for i in range(n_sims):
        points = SAMPLERS[family](rng, **model)
        if len(points) < 2:                # L undefined: theta_hat is hopeless, not untested.
            return 1.0 / (n_sims + 1)      # the strongest rejection the rank can express
        c = curves(points, GRID, f_grid)
        block[i + 1] = [c[name] for name in NAMES]
    return envelope_p(block)


def tier2(family: str, pred_rows: list[dict], observed: np.ndarray, todo: np.ndarray,
          n_sims: int, seed: int, jobs: int) -> np.ndarray:
    """envelope_p per pattern, NaN where `todo` is False."""
    tasks = [(family, pred_rows[i], observed[i], seed * 1_000_003 + int(i), n_sims)
             for i in np.flatnonzero(todo)]
    if jobs == 1:
        got = [_task(t) for t in tasks]
    else:
        with get_context("spawn").Pool(jobs) as pool:
            got = pool.map(_task, tasks, chunksize=max(1, len(tasks) // (8 * jobs)))
    out = np.full(len(pred_rows), math.nan)
    out[todo] = got
    return out


# ---------------------------------------------------------------- driver

def observed_curves(family: str, case_id: np.ndarray) -> np.ndarray:
    """(N, 4, 512) L, F, G, J of the test patterns themselves, from data/classical."""
    z = np.load(CLASSICAL / family / "fixed" / "curves.npz")
    where = pd.Index(z["case_id"]).get_indexer(case_id)
    if (where < 0).any():
        raise SystemExit(f"{family}: {np.count_nonzero(where < 0)} test patterns missing from data/classical")
    return np.stack([z[name][where] for name in NAMES], axis=1).astype(float)


def subsample(case_id: np.ndarray, n_thetas: int) -> np.ndarray:
    """Both replicates of the first `n_thetas` distinct test thetas. Deterministic, so two runs
    scored this way cover the same thetas and regimes.py can still pair them."""
    theta = np.array([int(c.rsplit("-", 2)[1]) for c in case_id])
    return np.isin(theta, np.unique(theta)[:n_thetas])


def run(seed_dir: Path, tiers: set[int], n_sims: int, n_thetas: int, jobs: int, tables: Tables) -> dict:
    z = np.load(seed_dir / "predictions.npz")
    meta = json.loads((seed_dir / "run.json").read_text())
    family, columns = meta["args"]["family"], meta["targets"]["columns"]
    rows = lambda y: [dict(zip(columns, v)) for v in y]

    out = {}
    if 1 in tiers:
        out["dtilde"] = tier1(family, rows(z["y_true"]), rows(z["y_pred"]), tables)
    if 2 in tiers:
        todo = subsample(z["case_id"], n_thetas)
        p = tier2(family, rows(z["y_pred"]), observed_curves(family, z["case_id"]), todo,
                  n_sims, int(meta["args"]["seed"]), jobs)
        out["envelope_p"] = p
        out["envelope_reject"] = np.where(np.isnan(p), np.nan, (p <= ALPHA).astype(float))

    path = seed_dir / "predictive.npz"
    if path.exists():                       # keep the tier this call did not recompute
        old = np.load(path)
        if np.array_equal(old["case_id"], z["case_id"]):
            out = {**{k: v for k, v in old.items() if k != "case_id"}, **out}
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, case_id=z["case_id"], **out)
    tmp.replace(path)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--glob", default="*/*/*/seed_*", help="<family>/<features>/<variant>/seed_<n>")
    p.add_argument("--tier", default="1", help="'1', '2' or '1,2'")
    p.add_argument("--n-sims", type=int, default=B_DEFAULT, help="tier 2: simulations per pattern")
    p.add_argument("--thetas", type=int, default=THETAS_DEFAULT, help="tier 2: test thetas to score")
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args(argv)

    tiers = {int(t) for t in args.tier.split(",")}
    seed_dirs = sorted(d for d in args.results.glob(args.glob) if (d / "predictions.npz").exists())
    if not seed_dirs:
        raise SystemExit(f"no runs under {args.results / args.glob}")
    tables = Tables()

    for seed_dir in seed_dirs:
        t0 = time.time()
        out = run(seed_dir, tiers, args.n_sims, args.thetas, args.jobs, tables)
        summary = "  ".join(f"{k} {np.nanmean(v):.3f}" for k, v in sorted(out.items()))
        print(f"[predictive] {'/'.join(seed_dir.parts[-4:])}  {time.time() - t0:6.1f}s  {summary}", flush=True)


if __name__ == "__main__":
    main()
