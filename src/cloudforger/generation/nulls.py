# src/cloudforger/generation/nulls.py
"""CSR null tables for the departure measures, and delta-tilde
(generation.tex, "delta: distance from CSR as the classical test sees it").

For each n on a grid, binomial patterns (n uniform points in W) give the
null mean m0(r; n) and s.d. s0(r; n) of Phi_hat in {L - r, G, F}, from the
first half of the replicates, and the 95% quantile c95(n) of
max_{r in R} |Phi_hat - m0| / s0 from the held-out second half (so c95 is not
optimistically biased). Between grid values of n the departure statistic is
interpolated linearly in log n, each bracketing grid point evaluated with its
own m0, s0 and c95. This is exactly docs/theory/scripts/departure.py, which
produced the design numbers in generation.tex.

delta_tilde evaluates the same statistic on the closed-form L(r) - r at
n = nbar, *without* subtracting m0 (departure.py's center=False): the closed
form of CSR is exactly 0, so CSR has delta_tilde = 0, and the estimator's own
bias is left to the replicate-based delta-hat of B and C.

Estimators are the repo's: Ripley isotropic L with lambda^2 = n(n-1) on 513
radii over [0, 0.25] (baselines/vihrs.py), border-corrected G and F
(baselines/summstats.py).
"""

from __future__ import annotations

import io
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from ..baselines import summstats
from ..baselines.vihrs import _isotropic_l_minus_r, default_r_grid
from ..data_generation.point_processes import PoissonProcess
from .seeding import PARAMS, PATTERN, case_rng

R_GRID = default_r_grid()          # linspace(0, 0.25, 513)
FG_GRID = R_GRID
L_RMIN = 0.01                      # R = [0.01, 0.25] for L
SD_FRAC = 0.10                     # G, F: radii where s0 >= 10% of its max
FUNCS = ("L", "G", "F")
_LOW, _HIGH = np.zeros(2), np.ones(2)
_REP_MAX = 100_000


def null_index(n: int, rep: int) -> int:
    """Stream index of null replicate `rep` at size n (set 'pilot', family 'poisson')."""
    if not 0 <= rep < _REP_MAX:
        raise ValueError(f"rep must be in [0, {_REP_MAX})")
    return _REP_MAX * int(n) + rep


def null_curves(root: int, n: int, rep: int) -> np.ndarray:
    """(3, 513) array of L - r, G, F for one binomial pattern."""
    from .samplers import WINDOW  # local: samplers imports prior, which imports nothing from here
    rng = case_rng(root, "pilot", "poisson", null_index(n, rep), PATTERN)
    pts = PoissonProcess().sample(int(n), WINDOW, rng=rng).points
    lmr = _isotropic_l_minus_r(pts, _LOW, _HIGH, R_GRID)
    f, g, _ = summstats.compute_fgj(pts, _LOW, _HIGH, FG_GRID)
    return np.stack([lmr, g, f]).astype(np.float64)


def _curves_task(args: tuple[int, int, int]) -> np.ndarray:
    return null_curves(*args)


# --- statistics -----------------------------------------------------------------

def _mask(func: str, s0: np.ndarray) -> np.ndarray:
    ok = s0 > 0
    if func == "L":
        return ok & (R_GRID >= L_RMIN)
    return ok & (s0 >= SD_FRAC * s0.max())


def _tmax(curves: np.ndarray, m0: np.ndarray | float, s0: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, (curves - m0) / np.where(mask, s0, 1.0), 0.0)
    return np.abs(z).max(axis=-1)


def summarize(arr: np.ndarray, func: str, boot_rng: np.random.Generator, n_boot: int = 500) -> dict[str, Any]:
    """arr: (reps, 513) curves at one n. First half -> m0, s0; held-out half -> c95."""
    h = arr.shape[0] // 2
    m0 = arr[:h].mean(axis=0)
    s0 = arr[:h].std(axis=0, ddof=1)
    mask = _mask(func, s0)
    t = _tmax(arr[h:], m0, s0, mask)
    boot = np.quantile(t[boot_rng.integers(0, len(t), size=(n_boot, len(t)))], 0.95, axis=1)
    return {"m0": m0, "s0": s0, "mask": mask, "c95": float(np.quantile(t, 0.95)),
            "c95_se": float(boot.std(ddof=1)), "median": float(np.median(t))}


def build_tables(root: int, n_grid: list[int], reps: int, jobs: int = 1, log=print) -> dict[str, Any]:
    n_grid = [int(n) for n in n_grid]
    if reps % 2 or reps < 4:
        raise ValueError("reps must be even (half fit, half held out)")
    G, nr = len(n_grid), len(R_GRID)
    tabs: dict[str, Any] = {"n_grid": np.array(n_grid, float), "r_grid": R_GRID, "reps": reps}
    for func in FUNCS:
        tabs[func] = {"m0": np.empty((G, nr)), "s0": np.empty((G, nr)), "mask": np.empty((G, nr), bool),
                      "c95": np.empty(G), "c95_se": np.empty(G), "median": np.empty(G)}
    boot_rng = case_rng(root, "pilot", "poisson", 0, PARAMS)
    ctx = get_context("spawn")
    with ctx.Pool(jobs) if jobs > 1 else _Serial() as pool:
        for i, n in enumerate(n_grid):
            arr = np.stack(pool.map(_curves_task, [(root, n, rep) for rep in range(reps)], chunksize=64))
            for k, func in enumerate(FUNCS):
                for key, val in summarize(arr[:, k], func, boot_rng).items():
                    tabs[func][key][i] = val
            log(f"  n={n:5d}  c95(L)={tabs['L']['c95'][i]:.3f} +- {tabs['L']['c95_se'][i]:.3f}")
    return tabs


class _Serial:
    def map(self, f, xs, chunksize=None):
        return [f(x) for x in xs]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def tables_bytes(tabs: dict[str, Any]) -> bytes:
    flat = {"n_grid": tabs["n_grid"], "r_grid": tabs["r_grid"], "reps": np.array(tabs["reps"])}
    for func in FUNCS:
        for key, val in tabs[func].items():
            flat[f"{func}_{key}"] = np.asarray(val)
    buf = io.BytesIO()
    np.savez(buf, **flat)
    return buf.getvalue()


def load_tables(path: Path | str) -> dict[str, Any]:
    with np.load(path) as z:
        tabs: dict[str, Any] = {"n_grid": z["n_grid"], "r_grid": z["r_grid"], "reps": int(z["reps"])}
        for func in FUNCS:
            tabs[func] = {k: z[f"{func}_{k}"] for k in ("m0", "s0", "mask", "c95", "c95_se", "median")}
    if not np.array_equal(tabs["r_grid"], R_GRID):
        raise ValueError(f"{path}: r_grid differs from the estimator grid")
    return tabs


# --- departure and delta-tilde ----------------------------------------------------

def _bracket(n: np.ndarray, n_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ln_grid = np.log(n_grid)
    ln = np.log(np.clip(np.asarray(n, float), n_grid[0], n_grid[-1]))
    hi = np.clip(np.searchsorted(ln_grid, ln, side="right"), 1, len(ln_grid) - 1)
    lo = hi - 1
    return lo, (ln - ln_grid[lo]) / (ln_grid[hi] - ln_grid[lo])


def departure(curves: np.ndarray, n: np.ndarray, func: str, tabs: dict[str, Any],
              center: bool = True) -> np.ndarray:
    """S_max for each row of `curves` (m, 513) at sizes n (m,). S <= 1 <=> not
    rejected at 5% by this Monte Carlo test. center=False: |Phi| / s0 (population curves)."""
    curves = np.atleast_2d(np.asarray(curves, float))
    t = tabs[func]
    lo, w = _bracket(np.atleast_1d(n), tabs["n_grid"])
    out = np.empty(len(curves))
    for i in np.unique(lo):
        rows = np.flatnonzero(lo == i)
        parts = [_tmax(curves[rows], t["m0"][j] if center else 0.0, t["s0"][j], t["mask"][j]) / t["c95"][j]
                 for j in (i, i + 1)]
        out[rows] = (1 - w[rows]) * parts[0] + w[rows] * parts[1]
    return out


def l_minus_r(excess: np.ndarray, r: np.ndarray = R_GRID) -> np.ndarray:
    """L(r) - r from the excess e = K - pi r^2, as (e/pi) / (sqrt(r^2 + e/pi) + r):
    no cancellation near CSR, and exactly 0 when e = 0."""
    e = np.asarray(excess, float) / np.pi
    root = np.sqrt(np.clip(r**2 + e, 0.0, None))
    return np.where(r**2 + e <= 0.0, -r, np.divide(e, root + r, out=np.zeros_like(e), where=(root + r) > 0))


def delta_tilde(excess: np.ndarray, nbar: float, tabs: dict[str, Any]) -> float:
    """Design-time delta from a closed-form excess K - pi r^2 on R_GRID."""
    return float(departure(l_minus_r(excess)[None, :], np.array([nbar]), "L", tabs, center=False)[0])


def s0_at(n: float, func: str, tabs: dict[str, Any]) -> np.ndarray:
    """Null s.d. of Phi_hat at size n, interpolated linearly in log n."""
    lo, w = _bracket(np.array([n]), tabs["n_grid"])
    s0 = tabs[func]["s0"]
    return (1 - w[0]) * s0[lo[0]] + w[0] * s0[lo[0] + 1]
