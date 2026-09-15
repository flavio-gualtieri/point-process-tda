# src/cloudforger/evaluation/classical.py
"""The classical CSR test on the standard summary functions, run on DV3's
evaluation sets so its power across regimes sits in the same tables as the
learned classifiers'.

WHAT IS TESTED. For each pattern and each summary function Phi in {L - r,
G, F} (Ripley's isotropic L, border-corrected G and F -- the exact
estimators the vihrs/summstats baselines and the DV3 null tables use), the
studentised global envelope statistic (Myllymaki et al. 2017)

    S_Phi(x) = max_{r in R_Phi} |Phi_hat_x(r) - m0(r; n)| / s0(r; n)  /  c95(n)

with m0, s0 and c95 the binomial-process null moments and 95% point at the
pattern's realised n, read from configs/generation/null_tables.npz (the
tables delta-tilde itself is defined by), with the null moments
interpolated in log n between grid sizes (studentised_departure explains
why not nulls.departure's statistic interpolation). S > 1 is the
pre-registered 5% rejection. S_L on a pattern is therefore the realised
counterpart of the design-time delta-tilde, and its power curve along a C
ladder is the yardstick delta-tilde was built against: power ~ 5% at
delta ~ 0, crossing ~50% near delta ~ 1.

Combined test. S_LGF = max(S_L, S_G, S_F) rejects if any function does. Its
nominal level is above 5% (three looks), which is why every detector in the
regime tables is compared at an EMPIRICAL 5% threshold calibrated on the C
CSR anchors (regimes.calibrate_threshold) rather than at its nominal one --
the nominal-threshold power is kept alongside for the single-function tests,
where it is a genuine 5% test.

J is not tested separately: J = (1 - G)/(1 - F) is a deterministic function
of the two curves already in the max, and the tables carry no null for it.

Small n. The null grid starts at n = 50; nulls.departure clamps below it,
so a pattern with n < 50 is tested against the n = 50 null. DV3 has
nbar >= 100 and n >= 15, so this only touches strongly clustered patterns
at small nbar (recorded per set as `frac_n_below_grid`).
"""

from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..baselines import summstats
from ..baselines.vihrs import _isotropic_l_minus_r
from ..core.io import load_pickle
from ..generation import nulls
from ..paths import PROJECT_ROOT
from .dv3 import DEFAULT_DV3_ROOT, data_paths

NULL_TABLES = PROJECT_ROOT / "configs" / "generation" / "null_tables.npz"
STATS_FILENAME = "envelope_stats.npz"
FUNCTIONS: tuple[str, ...] = ("L", "G", "F")
TESTS: tuple[str, ...] = ("L", "G", "F", "LGF")
_LOW, _HIGH = np.zeros(2), np.ones(2)


def pattern_curves(points: np.ndarray) -> np.ndarray:
    """(3, 513) L - r, G, F of one pattern on the null tables' own radius grid
    (order = nulls.FUNCS)."""
    pts = np.asarray(points, dtype=np.float64)
    lmr = _isotropic_l_minus_r(pts, _LOW, _HIGH, nulls.R_GRID)
    f, g, _ = summstats.compute_fgj(pts, _LOW, _HIGH, nulls.FG_GRID)
    return np.stack([lmr, g, f])


def _curves_task(points: np.ndarray) -> np.ndarray:
    return pattern_curves(points).astype(np.float32)


def studentised_departure(curves: np.ndarray, n: np.ndarray, func: str, tabs: dict[str, Any]) -> np.ndarray:
    """S for each row of `curves` at realised sizes `n`, with the null
    MOMENTS interpolated in log n (m0, s0, c95; mask = both brackets' masks),
    then one statistic evaluated against them.

    Not nulls.departure(center=True), which interpolates the STATISTIC
    between the two bracketing grid sizes, each computed against its own
    m0(r; n_grid). That is harmless for L - r (nearly n-invariant after the
    n(n-1) normalisation) but not for F: F(r) ~ 1 - exp(-n pi r^2) moves with
    n by many fixed-n null s.d.s between adjacent grid sizes, so a pattern at
    n = 137 sits far from both m0(124) and m0(149) and is rejected almost
    surely -- measured size 1.00 for F at n = 137 (0.043 with moments
    interpolated), and 0.60 on DV3's own CSR anchors, whose n is Poisson and
    therefore almost never on the grid. nulls.departure itself is left as
    is: delta-tilde (center=False, L only) is defined by it and the DV3 plan
    is checksummed against those values."""
    curves = np.atleast_2d(np.asarray(curves, float))
    t = tabs[func]
    lo, w = nulls._bracket(np.atleast_1d(n), tabs["n_grid"])
    out = np.empty(len(curves))
    for i in np.unique(lo):
        rows = np.flatnonzero(lo == i)
        ww = w[rows][:, None]
        m0 = (1 - ww) * t["m0"][i] + ww * t["m0"][i + 1]
        s0 = (1 - ww) * t["s0"][i] + ww * t["s0"][i + 1]
        c95 = (1 - w[rows]) * t["c95"][i] + w[rows] * t["c95"][i + 1]
        mask = t["mask"][i] & t["mask"][i + 1]
        z = np.where(mask, (curves[rows] - m0) / np.where(mask, s0, 1.0), 0.0)
        out[rows] = np.abs(z).max(axis=1) / c95
    return out


def envelope_statistics(
    clouds: Sequence[dict[str, Any]], tabs: dict[str, Any], jobs: int = 1,
) -> dict[str, np.ndarray]:
    """{"S_L", "S_G", "S_F", "S_LGF", "n"} for a list of cloud records."""
    pts = [np.asarray(c["points"], dtype=np.float64) for c in clouds]
    n = np.array([len(p) for p in pts], dtype=np.int64)
    if jobs > 1:
        with get_context("spawn").Pool(jobs) as pool:
            curves = np.stack(pool.map(_curves_task, pts, chunksize=64))
    else:
        curves = np.stack([_curves_task(p) for p in pts])
    out: dict[str, np.ndarray] = {"n": n}
    for k, func in enumerate(nulls.FUNCS):
        out[f"S_{func}"] = studentised_departure(curves[:, k].astype(np.float64), n, func, tabs)
    out["S_LGF"] = np.max(np.stack([out[f"S_{f}"] for f in FUNCTIONS]), axis=0)
    return out


def stats_path(set_: str, family: str, root: Path | str = DEFAULT_DV3_ROOT) -> Path:
    return data_paths(set_, family, root).process_dir / STATS_FILENAME


def load_or_compute(
    set_: str, family: str, root: Path | str = DEFAULT_DV3_ROOT,
    tables_path: Path | str = NULL_TABLES, jobs: int = 1, force: bool = False,
) -> dict[str, np.ndarray]:
    """Cached per (set, family) next to its clouds.pkl; the cache is keyed by
    case_id, so a regenerated set with different patterns is detected and
    recomputed rather than silently reused."""
    clouds_path = data_paths(set_, family, root).clouds()
    cache = stats_path(set_, family, root)
    clouds = load_pickle(clouds_path)
    ids = np.array([str(c["case_id"]) for c in clouds], dtype=np.str_)
    if cache.exists() and not force:
        with np.load(cache, allow_pickle=False) as z:
            if np.array_equal(z["case_id"], ids):
                return {k: z[k] for k in z.files}
        print(f"  [envelope] {cache} is stale (case_ids differ) -- recomputing")
    tabs = nulls.load_tables(tables_path)
    print(f"  [envelope] {set_}/{family}: {len(clouds)} patterns ...")
    stats = envelope_statistics(clouds, tabs, jobs=jobs)
    stats["case_id"] = ids
    stats["family"] = np.array([str(c["process"]) for c in clouds], dtype=np.str_)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **stats)
    return stats
