# model/goodness.py


"""
goodness_of_fit.py
==================

Simulation-envelope goodness-of-fit test for Thomas process parameter
estimates, following Myllymäki et al. (2017) and Diggle (2013, ch. 8).

Workflow
--------
For each (cloud, estimator) pair:

1. Load the observed point cloud and crop/rescale it to [0,1]^2 using the
   same procedure as processing.py (via mincontrast.crop_and_rescale).
2. Compute K_obs(r) — the edge-corrected empirical K-function of the observed
   cloud on [0,1]^2.
3. Using the fitted parameters (kappa, sigma, mu), simulate n_sim independent
   Thomas realisations on [0,1]^2 via cloudforger.ThomasProcess.
4. Compute K_sim_i(r) for each simulation.
5. From the n_sim + 1 curves (observed + simulated), compute:
       • Pointwise simulation envelope: 2.5th–97.5th percentile at each r.
       • Global rank-envelope p-value (Myllymäki et al. 2017):
             u_i = min_r rank_r(K_i) among all n_sim+1 curves
             p = #{u_i <= u_obs} / (n_sim + 1)
         where rank_r is the rank from lowest to highest at radius r,
         and the extreme rank is min(u_obs, (n_sim+1) - u_obs + 1) to make it
         two-sided.
       • MAD score: mean_r |K_obs(r) - mean_r K_sim(r)|^0.25
         (same contrast transform used in fitting, to put it on a stable scale).
6. Save one PNG per cloud showing the observed K, the envelope, and a summary
   table of scalar scores for all estimators.

Usage
-----
    # score a single cloud with its mincontrast estimates
    python goodness_of_fit.py --cloud data/barro/bci.tree1_cloud.pkl \
        --params '{"parent_intensity": 60, "cluster_scale": 0.04, "mean_offspring": 15}'

    # score all 8 BCI censuses for all estimators found in compare.py
    python goodness_of_fit.py --all --data-dir data/barro --out-dir results/gof

    # control the test
    python goodness_of_fit.py --all --n-sim 199 --seed 1 --n-jobs 4

Design notes
------------
• The unit of analysis is the cropped, rescaled [0,1]^2 sub-window.  All K
  values and parameter units are in that space, matching the training set.
• n_sim = 199 gives a two-sided 5 % rank-envelope test with exact level
  (Myllymäki et al. 2017, Theorem 1); use 999 for publication.
• The empirical K is O(N^2) in the number of points.  With N ~ TRAIN_TARGET_N
  = 900 and n_sim = 199 this is 200 × 900^2 / 2 ~ 81 M pair comparisons —
  fast in NumPy but multiprocessing (--n-jobs) helps for many censuses.
• The cloudforger ThomasProcess is used directly for simulation so the
  generated clouds are guaranteed to match the training distribution exactly.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
import mincontrast as mc   # crop_and_rescale, empirical_K, K_thomas, TRAIN_TARGET_N

try:
    from cloudforger.core.region import Box
    from cloudforger.processes.thomas import ThomasProcess
    _CLOUDFORGER = True
except ImportError:
    _CLOUDFORGER = False
    warnings.warn(
        "cloudforger not importable — simulation will use the built-in fallback "
        "Thomas simulator (no edge-buffer correction). Install cloudforger for "
        "production use."
    )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL = True
except ImportError:
    _MPL = False
    warnings.warn("matplotlib not found — plots will be skipped.")

try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except ImportError:
    _JOBLIB = False


# ══════════════════════════════════════════════════════════════════════════════
# Thomas process simulator (unit square)
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_thomas_cloudforger(
        kappa: float,
        sigma: float,
        mu: float,
        rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one Thomas realisation on [0,1]^2 using cloudforger."""
    process = ThomasProcess(
        parent_intensity=kappa,
        mean_offspring=mu,
        cluster_scale=sigma,
    )
    unit_box = Box(low=np.zeros(2), high=np.ones(2))
    seed = int(rng.integers(0, 2**31))
    cloud = process.sample(region=unit_box, seed=seed)
    return cloud.points


def _simulate_thomas_fallback(
        kappa: float,
        sigma: float,
        mu: float,
        rng: np.random.Generator,
) -> np.ndarray:
    """Pure-NumPy Thomas simulator — fallback when cloudforger is unavailable.

    Uses a 6-sigma buffer around [0,1]^2 for parents so that clusters whose
    centres lie just outside the window can still deposit offspring inside it.
    """
    buf = 6.0 * sigma
    lo, hi = -buf, 1.0 + buf
    area_expanded = (hi - lo) ** 2

    n_parents = int(rng.poisson(kappa * area_expanded))
    if n_parents == 0:
        return np.empty((0, 2))

    px = rng.uniform(lo, hi, n_parents)
    py = rng.uniform(lo, hi, n_parents)
    n_off = rng.poisson(mu, n_parents)
    total = int(n_off.sum())
    if total == 0:
        return np.empty((0, 2))

    idx = np.repeat(np.arange(n_parents), n_off)
    ox = px[idx] + rng.normal(0.0, sigma, total)
    oy = py[idx] + rng.normal(0.0, sigma, total)
    mask = (ox >= 0) & (ox <= 1) & (oy >= 0) & (oy <= 1)
    return np.column_stack([ox[mask], oy[mask]])


def simulate_thomas(
        kappa: float,
        sigma: float,
        mu: float,
        rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one Thomas realisation on [0,1]^2."""
    if _CLOUDFORGER:
        return _simulate_thomas_cloudforger(kappa, sigma, mu, rng)
    return _simulate_thomas_fallback(kappa, sigma, mu, rng)


# ══════════════════════════════════════════════════════════════════════════════
# K-function helpers
# ══════════════════════════════════════════════════════════════════════════════

def safe_K(points: np.ndarray, t_values: np.ndarray) -> np.ndarray:
    """Return empirical K or NaN array if too few points."""
    if len(points) < 2:
        return np.full(len(t_values), np.nan)
    K, _ = mc.empirical_K(points, t_values)
    return K


# ══════════════════════════════════════════════════════════════════════════════
# Envelope statistics
# ══════════════════════════════════════════════════════════════════════════════

def rank_envelope_pvalue(K_obs: np.ndarray, K_sims: np.ndarray) -> float:
    """Global rank-envelope p-value (Myllymäki et al. 2017).

    Parameters
    ----------
    K_obs  : (n_t,)         observed K-function
    K_sims : (n_sim, n_t)   simulated K-functions

    Returns
    -------
    p-value in (0, 1].  Small values indicate poor fit.

    The statistic u_i is the minimum pointwise rank of curve i among all
    n_sim+1 curves.  The two-sided p-value is the fraction of simulations
    whose extreme rank is <= that of the observed curve.
    """
    all_curves = np.vstack([K_sims, K_obs])          # (n_sim+1, n_t)
    n_total = all_curves.shape[0]

    # Rank each curve at each r (1 = smallest).  Use 'ordinal' to break ties.
    order = np.argsort(np.argsort(all_curves, axis=0, kind="stable"), axis=0) + 1

    obs_rank = order[-1]                              # rank of obs at each r
    sim_ranks = order[:-1]                            # (n_sim, n_t)

    # Extreme rank: how deep inside the envelope is each curve?
    # Two-sided: min of lower tail and upper tail.
    def extreme(row):
        return min(row.min(), n_total - row.max() + 1)

    u_obs  = extreme(obs_rank)
    u_sims = np.array([extreme(sim_ranks[i]) for i in range(len(K_sims))])

    # p = fraction of simulations at least as extreme as observed
    p = float(np.sum(u_sims <= u_obs) / (len(K_sims) + 1))
    return max(p, 1.0 / (len(K_sims) + 1))   # floor at 1/(n_sim+1)


def mad_score(K_obs: np.ndarray, K_sims: np.ndarray, c: float = 0.25) -> float:
    """Mean absolute deviation of K_obs^c from the simulation mean K^c.

    Uses the same contrast power c = 0.25 as minimum-contrast estimation to
    put the score on a variance-stabilised scale.  Lower = better fit.
    """
    K_mean = np.nanmean(K_sims, axis=0)
    return float(np.mean(np.abs(K_obs**c - K_mean**c)))


# ══════════════════════════════════════════════════════════════════════════════
# Core GOF function for one (observed cloud, parameter dict) pair
# ══════════════════════════════════════════════════════════════════════════════

def gof_envelope(
        points_unit: np.ndarray,
        params: dict[str, float],
        n_sim: int = 199,
        r_low: float = 1e-3,
        r_high: float = 0.25,
        n_t: int = 200,
        seed: int = 0,
        n_jobs: int = 1,
) -> dict[str, Any]:
    """Run the simulation-envelope test for one cloud and one parameter set.

    Parameters
    ----------
    points_unit : (N, 2) array in [0,1]^2  (already cropped and rescaled)
    params      : dict with keys parent_intensity, cluster_scale, mean_offspring
    n_sim       : number of simulations (199 → exact 5% two-sided test)
    r_low/high  : K-function evaluation range
    n_t         : number of r values
    seed        : master RNG seed
    n_jobs      : parallel workers for simulation (requires joblib)

    Returns
    -------
    dict with:
        t_values      : (n_t,) r grid
        K_obs         : (n_t,) observed K
        K_sims        : (n_sim, n_t) simulated K curves
        K_lo, K_hi    : (n_t,) pointwise 2.5/97.5 percentile envelope
        K_mean        : (n_t,) simulation mean
        K_theo        : (n_t,) theoretical K at fitted params
        p_value       : global rank-envelope p-value
        mad           : MAD score (lower = better)
        n_obs         : number of observed points
        n_sim_mean    : mean simulated point count
        params        : the parameter dict passed in
    """
    kappa = float(params["parent_intensity"])
    sigma = float(params["cluster_scale"])
    mu    = float(params["mean_offspring"])

    t_values = np.linspace(r_low, r_high, n_t)

    # ── observed K ────────────────────────────────────────────────────────────
    K_obs, _ = mc.empirical_K(points_unit, t_values)

    # ── simulate ──────────────────────────────────────────────────────────────
    master_rng = np.random.default_rng(seed)
    sim_seeds = master_rng.integers(0, 2**31, size=n_sim)

    def _one_sim(s: int) -> tuple[np.ndarray, int]:
        rng_i = np.random.default_rng(int(s))
        pts = simulate_thomas(kappa, sigma, mu, rng_i)
        return safe_K(pts, t_values), len(pts)

    if _JOBLIB and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_one_sim)(s) for s in sim_seeds
        )
    else:
        results = [_one_sim(s) for s in sim_seeds]

    K_sims     = np.array([r[0] for r in results])   # (n_sim, n_t)
    n_sim_pts  = np.array([r[1] for r in results])

    # ── envelope and statistics ───────────────────────────────────────────────
    K_lo   = np.nanpercentile(K_sims, 2.5,  axis=0)
    K_hi   = np.nanpercentile(K_sims, 97.5, axis=0)
    K_mean = np.nanmean(K_sims, axis=0)
    K_theo = mc.K_thomas(t_values, kappa, sigma)

    p_val  = rank_envelope_pvalue(K_obs, K_sims)
    mad    = mad_score(K_obs, K_sims)

    return {
        "t_values":    t_values,
        "K_obs":       K_obs,
        "K_sims":      K_sims,
        "K_lo":        K_lo,
        "K_hi":        K_hi,
        "K_mean":      K_mean,
        "K_theo":      K_theo,
        "p_value":     p_val,
        "mad":         mad,
        "n_obs":       len(points_unit),
        "n_sim_mean":  float(n_sim_pts.mean()),
        "params":      params,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public API: score one cloud against multiple estimators
# ══════════════════════════════════════════════════════════════════════════════

def score_cloud(
        cloud_path: Path,
        estimator_params: dict[str, dict[str, float]],
        crop_seed: int = 0,
        target_n: int = mc.TRAIN_TARGET_N,
        n_sim: int = 199,
        r_low: float = 1e-3,
        r_high: float = 0.25,
        n_t: int = 200,
        gof_seed: int = 42,
        n_jobs: int = 1,
) -> dict[str, dict[str, Any]]:
    """Score one cloud against each set of fitted parameters.

    Parameters
    ----------
    cloud_path       : path to a *_cloud.pkl file
    estimator_params : {estimator_name: {parent_intensity, cluster_scale,
                        mean_offspring}}
    crop_seed        : RNG seed for the sub-window crop (match processing.py)
    target_n         : target point count after crop
    n_sim            : simulations per estimator
    gof_seed         : master seed for the GOF simulations

    Returns
    -------
    {estimator_name: gof_envelope(...) result dict}
    """
    cloud = mc.load_cloud(cloud_path)
    points = np.asarray(cloud["points"], dtype=float)
    reg    = cloud.get("region", {})
    low    = np.asarray(reg.get("low",  [0.0, 0.0]), dtype=float)
    high   = np.asarray(reg.get("high", [1.0, 1.0]), dtype=float)

    rng = np.random.default_rng(crop_seed)
    points_unit = mc.crop_and_rescale(points, low, high, target_n=target_n, rng=rng)

    results: dict[str, dict[str, Any]] = {}
    for name, params in estimator_params.items():
        print(f"    [{name}] simulating {n_sim} realisations ...", end=" ", flush=True)
        results[name] = gof_envelope(
            points_unit, params,
            n_sim=n_sim, r_low=r_low, r_high=r_high, n_t=n_t,
            seed=gof_seed, n_jobs=n_jobs,
        )
        r = results[name]
        print(f"p={r['p_value']:.3f}  MAD={r['mad']:.4f}  "
              f"n_obs={r['n_obs']}  n_sim_mean={r['n_sim_mean']:.0f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

_COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]


def plot_envelopes(
        results: dict[str, dict[str, Any]],
        title: str,
        out_path: Path,
        c: float = 0.25,
) -> None:
    """One figure with one panel per estimator showing K^c envelope + obs."""
    if not _MPL:
        return

    n_est = len(results)
    fig, axes = plt.subplots(
        1, n_est,
        figsize=(5.5 * n_est, 4.5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, (name, res), col in zip(axes, results.items(), _COLOURS):
        t   = res["t_values"]
        c_  = c  # contrast power for display

        K_obs_c  = res["K_obs"]  ** c_
        K_lo_c   = res["K_lo"]   ** c_
        K_hi_c   = res["K_hi"]   ** c_
        K_mean_c = res["K_mean"] ** c_
        K_theo_c = res["K_theo"] ** c_

        # envelope band
        ax.fill_between(t, K_lo_c, K_hi_c,
                        alpha=0.25, color=col, label="95% envelope")
        # simulation mean
        ax.plot(t, K_mean_c, color=col, lw=1.0, ls="--", label="sim mean")
        # theoretical
        ax.plot(t, K_theo_c, color="grey", lw=1.0, ls=":", label="theoretical")
        # observed
        ax.plot(t, K_obs_c, color="black", lw=1.8, label="observed")

        ax.set_title(
            f"{name}\np={res['p_value']:.3f}   MAD={res['mad']:.4f}",
            fontsize=10,
        )
        ax.set_xlabel("r")
        ax.set_ylabel(f"K(r)^{c_}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {out_path}")


def plot_summary_table(
        all_scores: dict[str, dict[str, dict[str, Any]]],
        out_path: Path,
) -> None:
    """Time-series of p-value and MAD across censuses for each estimator."""
    if not _MPL:
        return

    census_years = {
        1: "81–83", 2: "1985", 3: "90–92", 4: "95–96",
        5: "00–01", 6: "05–06", 7: "10–11", 8: "13–16",
    }

    # all_scores[census_key][estimator_name] -> result dict
    census_keys = sorted(all_scores)
    estimator_names = list(next(iter(all_scores.values())))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    x = np.arange(len(census_keys))
    x_labels = [census_years.get(k, str(k)) for k in census_keys]

    for i, name in enumerate(estimator_names):
        col = _COLOURS[i % len(_COLOURS)]
        p_vals = [all_scores[c].get(name, {}).get("p_value", np.nan)
                  for c in census_keys]
        mads   = [all_scores[c].get(name, {}).get("mad", np.nan)
                  for c in census_keys]
        ax1.plot(x, p_vals, marker="o", label=name, color=col, lw=1.8)
        ax2.plot(x, mads,   marker="o", label=name, color=col, lw=1.8)

    ax1.axhline(0.05, color="red", ls="--", lw=1, label="α = 0.05")
    ax1.set_ylabel("Rank-envelope p-value")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("MAD score  (lower = better)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax2.set_xlabel("Census")

    fig.suptitle("GOF summary — BCI 50-ha plot", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary plot → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--cloud", type=Path, metavar="CLOUD_PKL",
        help="score a single cloud; requires --params",
    )
    mode.add_argument(
        "--all", action="store_true",
        help="score all 8 BCI censuses using estimators from compare.py",
    )

    p.add_argument(
        "--params", type=str, default=None,
        metavar="JSON",
        help="JSON dict of estimator→params for --cloud mode, e.g. "
             "'{\"mc\": {\"parent_intensity\": 60, \"cluster_scale\": 0.03, "
             "\"mean_offspring\": 15}}'",
    )
    p.add_argument("--data-dir",  type=Path, default=Path("data/barro"))
    p.add_argument("--out-dir",   type=Path, default=None)
    p.add_argument("--n-sim",     type=int,  default=199,
                   help="simulations per test (default: 199)")
    p.add_argument("--n-jobs",    type=int,  default=1,
                   help="parallel workers (requires joblib; default: 1)")
    p.add_argument("--crop-seed", type=int,  default=0,
                   help="RNG seed for sub-window crop (default: 0)")
    p.add_argument("--gof-seed",  type=int,  default=42,
                   help="RNG seed for GOF simulations (default: 42)")
    p.add_argument("--r-high",    type=float, default=0.25,
                   help="upper K integration limit (default: 0.25)")
    p.add_argument("--n-t",       type=int,  default=200,
                   help="number of r values (default: 200)")
    p.add_argument("--estimators", nargs="+", default=None,
                   help="subset of estimators to run (default: all in compare.py)")
    p.add_argument("--censuses",  type=int,  default=8,
                   help="number of BCI censuses to process (default: 8)")
    return p


def _load_compare_estimates(
        data_dir: Path,
        estimator_names: list[str] | None,
        n_censuses: int,
) -> dict[int, dict[str, dict[str, float]]]:
    """Run compare.run_all() and return {census: {estimator: params}}."""
    try:
        import compare as cmp
    except ImportError:
        raise ImportError("compare.py not found. Run from the project root.")

    available = list(cmp.ESTIMATORS)
    selected  = estimator_names or available
    unknown   = [n for n in selected if n not in cmp.ESTIMATORS]
    if unknown:
        raise ValueError(f"Unknown estimators: {unknown}. Available: {available}")

    raw = cmp.run_all(data_dir, selected, n_censuses=n_censuses)
    # raw[estimator_name][census] = {param: value}
    # invert to census-first
    censuses: dict[int, dict[str, dict[str, float]]] = {}
    for name, by_census in raw.items():
        for c, params in by_census.items():
            censuses.setdefault(c, {})[name] = params
    return censuses


def main() -> None:
    args   = _build_parser().parse_args()
    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gof_kwargs = dict(
        n_sim=args.n_sim,
        r_high=args.r_high,
        n_t=args.n_t,
        gof_seed=args.gof_seed,
        crop_seed=args.crop_seed,
        n_jobs=args.n_jobs,
    )

    # ── single-cloud mode ─────────────────────────────────────────────────────
    if args.cloud is not None:
        if args.params is None:
            raise SystemExit("--params JSON is required with --cloud.")
        estimator_params: dict[str, dict[str, float]] = json.loads(args.params)
        print(f"\nScoring {args.cloud.name} ...")
        results = score_cloud(args.cloud, estimator_params, **gof_kwargs)

        if _MPL:
            plot_envelopes(
                results,
                title=args.cloud.stem,
                out_path=out_dir / f"{args.cloud.stem}_gof.png",
            )

        # print scalar summary
        print(f"\n{'Estimator':<18}  {'p-value':>8}  {'MAD':>10}  {'n_obs':>7}  {'n_sim_mean':>10}")
        print("-" * 60)
        for name, r in results.items():
            print(f"{name:<18}  {r['p_value']:>8.3f}  {r['mad']:>10.5f}  "
                  f"{r['n_obs']:>7d}  {r['n_sim_mean']:>10.0f}")
        return

    # ── all-censuses mode ─────────────────────────────────────────────────────
    print("Estimating parameters for all censuses via compare.py ...")
    census_params = _load_compare_estimates(
        args.data_dir, args.estimators, args.censuses
    )

    all_scores: dict[int, dict[str, dict[str, Any]]] = {}

    print(f"\nRunning GOF tests (n_sim={args.n_sim}) ...")
    for census, est_params in sorted(census_params.items()):
        cloud_path = args.data_dir / f"bci.tree{census}_cloud.pkl"
        if not cloud_path.exists():
            print(f"  census {census}: {cloud_path.name} not found — skipping")
            continue
        print(f"\nCensus {census}  ({cloud_path.name})")
        results = score_cloud(cloud_path, est_params, **gof_kwargs)
        all_scores[census] = results

        if _MPL:
            plot_envelopes(
                results,
                title=f"BCI census {census}",
                out_path=out_dir / f"census{census}_gof.png",
            )

    # ── summary table ─────────────────────────────────────────────────────────
    estimator_names = list(next(iter(all_scores.values())))
    print(f"\n{'Census':<8}", end="")
    for name in estimator_names:
        print(f"  {name+' p':>12}  {name+' MAD':>12}", end="")
    print()
    print("-" * (8 + 28 * len(estimator_names)))
    for census in sorted(all_scores):
        print(f"{census:<8}", end="")
        for name in estimator_names:
            r = all_scores[census].get(name, {})
            pv  = r.get("p_value", float("nan"))
            mad = r.get("mad",     float("nan"))
            print(f"  {pv:>12.3f}  {mad:>12.5f}", end="")
        print()

    plot_summary_table(
        all_scores,
        out_path=out_dir / "gof_summary.png",
    )


if __name__ == "__main__":
    main()