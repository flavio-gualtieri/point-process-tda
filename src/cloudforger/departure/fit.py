"""Stored null curves -> spline tables in log n.

Everything the tables hold shrinks roughly like 1/n -- the pointwise m0, s0 and the scalar
extremum moments alike -- so every spline is fitted to n*quantity (weighted by its Monte Carlo
s.e.) and divided back out at use time.

Three passes over the stored curves, in the order the reductions depend on each other:

  fit batch          pointwise m0(r), s0(r), and the scalar moments m_lo, s_lo, m_hi, s_hi
  calibration batch  c_alpha for `sup`, `lo` and `hi`, each under the FITTED moments
  calibration batch  c_alpha for `ext`, under the FITTED c_alpha of its two arms -- so `ext`'s
                     own critical value IS the joint scale k that holds the two-arm rule at alpha

Adding a reduction refits from the stored curves and never re-simulates.
"""

from __future__ import annotations

import numpy as np

from ..classical.lfunction import RADII
from .config import TABLES, Config
from .simulate import curves_path
from .tables import REDUCTIONS, SCALAR, basis, knots, r_min


def raw_moments(x: np.ndarray):
    n = len(x)
    m = x.mean(axis=0)
    s = x.std(axis=0, ddof=1)
    mu4 = ((x - m) ** 4).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se_s = np.sqrt(np.maximum(mu4 - s**4, 0.0) / n) / (2 * s)
    return m, s, s / np.sqrt(n), se_s


def quantile_se(t: np.ndarray, q: float, h: float = 0.01) -> float:
    """Order-statistic s.e. of a sample quantile, with `h` widened on ties.

    The raw extremum of a curve sampled on RADII is quantised onto that grid, and at a few n the
    whole [q - h, q + h] band lands on a single grid step. A zero s.e. there would hand that n
    unbounded weight in the spline fit -- the same failure `min_pairs` exists to prevent for the
    sup -- so the band widens until it resolves a spacing, and reports `inf` if it never does,
    which drops the row from the fit rather than dominating it."""
    for _ in range(6):
        lo, hi = np.quantile(t, [max(q - h, 0.0), min(q + h, 1.0)])
        if hi > lo:
            return float((hi - lo) / (2 * h) * np.sqrt(q * (1 - q) / len(t)))
        h *= 2
    return float("inf")


def wls(b: np.ndarray, y: np.ndarray, se: np.ndarray) -> np.ndarray:
    coef = np.zeros((b.shape[1], y.shape[1]))
    for j in range(y.shape[1]):
        ok = np.isfinite(se[:, j]) & (se[:, j] > 0)
        w = 1 / se[ok, j]
        coef[:, j] = np.linalg.lstsq(b[ok] * w[:, None], y[ok, j] * w, rcond=None)[0]
    return coef


def _z(y, fit, se, mask):
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (y - fit) / se
    return z[mask & np.isfinite(z)]


def _scalar(name: str, curves: np.ndarray) -> np.ndarray:
    """One reduction's raw extremum, before any studentising. Needs no tables and no n."""
    return REDUCTIONS[name].raw(None, curves, None, True)


def fit(cfg: Config) -> dict:
    grid = cfg.grid()
    nn = grid[:, None].astype(float)
    load = lambda n, a, b: np.load(curves_path(n), mmap_mode="r")[a:b].astype(np.float64)
    q = 1 - cfg.alpha

    # ---------------------------------------------------------------- pass 1: the fit batch
    pw, sc = [], {name: [] for name in SCALAR}
    for n in grid:
        x = load(n, 0, cfg.fit)
        pw.append(raw_moments(x))
        for name in SCALAR:
            sc[name].append(raw_moments(_scalar(name, x)[:, None]))
    m, s, se_m, se_s = map(np.stack, zip(*pw))

    t_m = knots(grid[0], grid[-1], cfg.knots_moments)
    b = basis(grid, t_m)
    coef_m, coef_s = wls(b, nn * m, nn * se_m), wls(b, nn * s, nn * se_s)
    m_fit, s_fit = b @ coef_m / nn, b @ coef_s / nn

    t_s = knots(grid[0], grid[-1], cfg.knots_moments)
    b_s = basis(grid, t_s)
    sm = {name: np.stack([v[0] for v in sc[name]]) for name in SCALAR}      # (grid, 1)
    ss = {name: np.stack([v[1] for v in sc[name]]) for name in SCALAR}
    se_sm = {name: np.stack([v[2] for v in sc[name]]) for name in SCALAR}
    se_ss = {name: np.stack([v[3] for v in sc[name]]) for name in SCALAR}
    coef_sm = {name: wls(b_s, nn * sm[name], nn * se_sm[name])[:, 0] for name in SCALAR}
    coef_ss = {name: wls(b_s, nn * ss[name], nn * se_ss[name])[:, 0] for name in SCALAR}
    sm_fit = {name: (b_s @ coef_sm[name])[:, None] / nn for name in SCALAR}
    ss_fit = {name: (b_s @ coef_ss[name])[:, None] / nn for name in SCALAR}

    # ---------------------------------------------------------------- pass 2: the calibration batch
    windowed = RADII[None, :] >= r_min(grid, cfg.min_pairs)[:, None]
    T = {name: [None] * len(grid) for name in REDUCTIONS}
    c = {name: np.empty(len(grid)) for name in REDUCTIONS}
    se_c = {name: np.empty(len(grid)) for name in REDUCTIONS}
    for g, n in enumerate(grid):
        x = load(n, cfg.fit, cfg.reps)
        z = np.abs(x - m_fit[g]) / s_fit[g]
        T["sup"][g] = np.where(windowed[g][None, :], z, 0.0).max(axis=1)
        for name in SCALAR:
            T[name][g] = (_scalar(name, x) - sm_fit[name][g]) / ss_fit[name][g]
        for name in ("sup", *SCALAR):
            c[name][g], se_c[name][g] = np.quantile(T[name][g], q), quantile_se(T[name][g], q)

    t_c = knots(grid[0], grid[-1], cfg.knots_c95)
    b_c = basis(grid, t_c)
    coef_c = {name: wls(b_c, c[name][:, None], se_c[name][:, None])[:, 0] for name in ("sup", *SCALAR)}

    # ------------------------------------------- pass 3: the composite arms, under their fitted c_alpha
    for name, red in REDUCTIONS.items():
        if not red.arms:
            continue
        c_arm = {a: b_c @ coef_c[a] for a in red.arms}
        for g in range(len(grid)):
            M = np.max([T[a][g] / c_arm[a][g] for a in red.arms], axis=0)
            c[name][g], se_c[name][g] = np.quantile(M, q), quantile_se(M, q)
        coef_c[name] = wls(b_c, c[name][:, None], se_c[name][:, None])[:, 0]

    TABLES.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TABLES, n_low=grid[0], n_high=grid[-1], min_pairs=cfg.min_pairs,
             t_moments=t_m, coef_m=coef_m, coef_s=coef_s, t_scalar=t_s, t_c=t_c,
             **{f"coef_sm__{name}": coef_sm[name] for name in SCALAR},
             **{f"coef_ss__{name}": coef_ss[name] for name in SCALAR},
             **{f"coef_c__{name}": coef_c[name] for name in REDUCTIONS},
             grid=grid,
             **{f"c_raw__{name}": c[name] for name in REDUCTIONS},
             **{f"se_c__{name}": se_c[name] for name in REDUCTIONS})

    z_m, z_s = _z(m, m_fit, se_m, windowed), _z(s, s_fit, se_s, windowed)
    ok = np.ones_like(sm[SCALAR[0]], dtype=bool)
    summary = lambda z: {"mean_z2": float(np.mean(z**2)), "p99_abs_z": float(np.quantile(np.abs(z), 0.99))}
    return {
        "grid": {"size": len(grid), "low": int(grid[0]), "high": int(grid[-1])},
        "min_pairs": cfg.min_pairs,
        "m0": summary(z_m), "s0": summary(z_s),
        "scalar_moments": {
            name: {"m": {**summary(_z(sm[name], sm_fit[name], se_sm[name], ok)),
                         "n_times_m_range": [float((nn * sm[name]).min()), float((nn * sm[name]).max())]},
                   "s": {**summary(_z(ss[name], ss_fit[name], se_ss[name], ok)),
                         "n_times_s_range": [float((nn * ss[name]).min()), float((nn * ss[name]).max())]}}
            for name in SCALAR},
        "c_alpha": {name: {**summary((c[name] - b_c @ coef_c[name]) / se_c[name]),
                           "windowed": REDUCTIONS[name].windowed,
                           "arms": list(REDUCTIONS[name].arms),
                           "range": [float(c[name].min()), float(c[name].max())],
                           "max_se": float(se_c[name].max())} for name in REDUCTIONS},
        "expected_mean_z2": {"moments": 1 - b.shape[1] / len(grid), "c_alpha": 1 - b_c.shape[1] / len(grid)},
    }
