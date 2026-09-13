"""Label-free departure-from-CSR statistics (notes, Sec. 4.1).

For a summary function Phi in {L-r, G, F} estimated on the repo grids, with
m0(r; n), s0(r; n) the Monte Carlo mean / sd of Phi_hat(r) under a binomial
process (CSR conditional on n points) in the same window:

    T(x)   = max_{r in R}  |Phi_hat(r) - m0(r; n)| / s0(r; n)      (studentized sup)
    Tint(x)= mean_{r in R} ((Phi_hat(r) - m0(r; n)) / s0(r; n))^2  (integrated)
    S(x)   = T(x) / c95(n)          S <= 1  <=>  not rejected at 5% by this MC test

c95 comes from the held-out half of the null simulations (the other half gives
m0, s0), so it is not optimistically biased. Tables live on a grid of n and the
statistic is interpolated linearly in log n between the two bracketing grid
points (each evaluated with its own self-consistent m0, s0, mask, c95).

R: L uses r >= 0.01; G and F use radii where s0 >= 10% of its max (outside
that range the null has essentially no variability, so the studentized
deviation is numerically meaningless).
"""

from __future__ import annotations

import numpy as np

FUNCS = {"L": "lmr", "G": "g", "F": "f"}
L_RMIN = 0.01
SD_FRAC = 0.10


def _mask(func: str, s0: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    ok = s0 > 0
    if func == "L":
        return ok & (r_grid[None, :] >= L_RMIN)
    return ok & (s0 >= SD_FRAC * s0.max(axis=1, keepdims=True))


def _z(curves: np.ndarray, m0: np.ndarray, s0: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """curves (..., R); m0/s0/mask (R,) -> studentized deviations, 0 off-mask."""
    safe = np.where(mask, s0, 1.0)
    return np.where(mask, (curves - m0) / safe, 0.0)


def _stats(z: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.abs(z)
    k = a.argmax(axis=-1)
    tmax = np.take_along_axis(a, k[..., None], axis=-1)[..., 0]
    signed = np.take_along_axis(z, k[..., None], axis=-1)[..., 0]
    tint = (z ** 2).sum(axis=-1) / max(int(mask.sum()), 1)
    return tmax, tint, signed


def null_tables(null_npz) -> dict:
    """Mean/sd tables (first half of reps) and c95 (second half) per function and grid n."""
    z = np.load(null_npz)
    n_grid = z["n_grid"].astype(float)
    tabs: dict = {"n_grid": n_grid, "r_grid": z["r_grid"]}
    for func, key in FUNCS.items():
        arr = z[key].astype(np.float64)                  # (G, M, R)
        h = arr.shape[1] // 2
        m0 = arr[:, :h].mean(axis=1)
        s0 = arr[:, :h].std(axis=1, ddof=1)
        mask = _mask(func, s0, z["r_grid"])
        c95_max, c95_int, med_max = [], [], []
        for i in range(len(n_grid)):
            zz = _z(arr[i, h:], m0[i], s0[i], mask[i])
            tmax, tint, _ = _stats(zz, mask[i])
            c95_max.append(np.quantile(tmax, 0.95))
            c95_int.append(np.quantile(tint, 0.95))
            med_max.append(np.median(tmax))
        tabs[func] = dict(m0=m0, s0=s0, mask=mask, c95_max=np.array(c95_max),
                          c95_int=np.array(c95_int), med_max=np.array(med_max))
    return tabs


def departure(curves: np.ndarray, n: np.ndarray, func: str, tabs: dict,
              center: bool = True) -> dict[str, np.ndarray]:
    """S_max, S_int and the signed z at the argmax for each row of `curves`.

    center=False measures |Phi(r)| / s0 instead of |Phi_hat - m0| / s0, for a
    population curve (closed-form L(r) - r) whose CSR value is exactly 0.
    """
    t = tabs[func]
    ln_grid = np.log(tabs["n_grid"])
    ln = np.log(np.clip(np.asarray(n, float), tabs["n_grid"][0], tabs["n_grid"][-1]))
    hi = np.clip(np.searchsorted(ln_grid, ln, side="right"), 1, len(ln_grid) - 1)
    lo = hi - 1
    w = (ln - ln_grid[lo]) / (ln_grid[hi] - ln_grid[lo])
    out = {k: np.empty(len(curves)) for k in ("S_max", "S_int", "signed")}
    for i in np.unique(lo):
        rows = np.flatnonzero(lo == i)
        parts = []
        for j in (i, i + 1):
            m0 = t["m0"][j] if center else 0.0
            zz = _z(curves[rows].astype(np.float64), m0, t["s0"][j], t["mask"][j])
            tmax, tint, signed = _stats(zz, t["mask"][j])
            parts.append((tmax / t["c95_max"][j], tint / t["c95_int"][j], signed))
        ww = w[rows]
        out["S_max"][rows] = (1 - ww) * parts[0][0] + ww * parts[1][0]
        out["S_int"][rows] = (1 - ww) * parts[0][1] + ww * parts[1][1]
        out["signed"][rows] = np.where(ww < 0.5, parts[0][2], parts[1][2])
    return out


# ---------------------------------------------------------------- closed forms

def K_thomas(r, kappa, sigma):
    r = np.asarray(r, float)
    return np.pi * r**2 + (1.0 - np.exp(-r**2 / (4.0 * sigma**2))) / kappa


def K_nested(r, kappa, mu1, sigma1, sigma2):
    r = np.asarray(r, float)
    return (np.pi * r**2
            + (1.0 - np.exp(-r**2 / (4.0 * sigma2**2))) / (kappa * mu1)
            + (1.0 - np.exp(-r**2 / (4.0 * (sigma1**2 + sigma2**2)))) / kappa)


def disc_pair_cdf(d):
    """CDF of the distance between two independent uniform points in the unit disc."""
    d = np.clip(np.asarray(d, float), 0.0, 2.0)
    return (1.0 + (2.0 / np.pi) * (d**2 - 1.0) * np.arccos(d / 2.0)
            - (d / np.pi) * (1.0 + d**2 / 2.0) * np.sqrt(1.0 - d**2 / 4.0))


def K_matern_cluster(r, kappa, R):
    r = np.asarray(r, float)
    return np.pi * r**2 + disc_pair_cdf(r / R) / kappa


def K_lgcp_exp(r, sigma2, s, n_terms=80):
    """K for an LGCP with covariance sigma2 * exp(-r/s): series from
    g = exp(sigma2 e^{-r/s}) = sum_k sigma2^k e^{-k r/s} / k!."""
    r = np.asarray(r, float)
    out = np.pi * r**2
    term = 1.0
    for k in range(1, n_terms + 1):
        term *= sigma2 / k                       # sigma2^k / k!
        a = k / s
        out = out + 2.0 * np.pi * term * (1.0 - (1.0 + a * r) * np.exp(-a * r)) / a**2
    return out


def matern2_intensity(lam_p, R):
    return (1.0 - np.exp(-lam_p * np.pi * R**2)) / (np.pi * R**2)


def matern2_pcf(r, lam_p, R):
    """Pair correlation of Matern Type II (derived in the notes, Sec. 1)."""
    r = np.asarray(r, float)
    a = np.pi * R**2
    d = np.clip(r / (2.0 * R), 0.0, 1.0)
    lens = 2.0 * R**2 * (np.arccos(d) - d * np.sqrt(1.0 - d**2))   # |b(0,R) cap b(r,R)|
    U = 2.0 * a - lens
    with np.errstate(divide="ignore", invalid="ignore"):
        rho2 = 2.0 * (U * (1.0 - np.exp(-lam_p * a)) - a * (1.0 - np.exp(-lam_p * U))) / (a * U * (U - a))
    lam = matern2_intensity(lam_p, R)
    return np.where(r < R, 0.0, rho2 / lam**2)


def K_from_pcf(r, pcf, n_fine=20001):
    r = np.asarray(r, float)
    t = np.linspace(0.0, r.max(), n_fine)
    g = pcf(t)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] * t[1:] + g[:-1] * t[:-1]) * np.diff(t))])
    return 2.0 * np.pi * np.interp(r, t, cum)


def l_minus_r(K, r):
    return np.sqrt(np.clip(K, 0.0, None) / np.pi) - np.asarray(r, float)
