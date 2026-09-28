"""Patterns for scoring: observed ones from the bank, and simulations from a model.

Models are simulated with the bank's own samplers and the bank's n-conditioning (a pattern counts
only if n is in the bank's range), but never with the bank's seeds: every (cloud, model) pair has
its own stream, rng_for(seed, case_id, model), so any score is reproducible alone.
"""

from __future__ import annotations

import inspect
import json
import zlib
from functools import lru_cache

import numpy as np
import pandas as pd

from ..paths import BANK
from ..simulation.bank import Config as BankConfig
from ..simulation.processes import SAMPLERS, lgcp_eigenvalues

M_FALLBACK = 1024          # LGCP grid when an estimate has no admissible one (bank rule, lgcp_grid)


@lru_cache(maxsize=1)
def n_range() -> tuple[int, int]:
    return BankConfig.load().n_range


@lru_cache(maxsize=1)
def _tables():
    from ..departure.tables import Tables
    return Tables()


def observed(chosen: pd.DataFrame) -> dict[str, np.ndarray]:
    """case_id -> (n, 2) points, for the bank rows `chosen` (indexed by case_id, with `family`)."""
    out = {}
    for family, rows in chosen.groupby("family"):
        index = pd.read_csv(BANK / family / "manifest.csv", usecols=["case_id"]).case_id
        pos = pd.Series(np.arange(len(index)), index=index).loc[rows.index]
        z = np.load(BANK / family / "points.npz")
        points, offsets = z["points"], z["offsets"]
        out |= {c: points[offsets[i]:offsets[i + 1]].copy() for c, i in pos.items()}   # copies: let
        del points                                                                    # the arrays go
    return out


def true_model(row: pd.Series) -> tuple[str, dict]:
    """(family, sampler kwargs) of a bank row: the model that generated it."""
    family = row["family"]
    args = [a for a in inspect.signature(SAMPLERS[family]).parameters if a not in ("rng", "root_lam")]
    kw = {a: row[a] for a in args}
    if "M" in kw:
        kw["M"] = int(kw["M"])
    return family, kw


def sampler_kwargs(family: str, theta: dict) -> dict:
    """theta_hat (keyed by pipeline.core.TARGETS) -> the bank sampler's arguments. LGCP is estimated
    as (nbar, sigma2, s); the sampler wants mu_log = log nbar - sigma2 / 2 and a grid M, picked as the
    bank picks it."""
    if family == "cell":                                         # k is a count; estimates are real
        return {"nbar": theta["nbar"], "k": int(np.clip(round(theta["k"]), 2, 30))}
    if family != "lgcp":
        return dict(theta)
    from ..simulation.lgcp_grid import grid_size
    nbar, sigma2, s = theta["nbar"], theta["sigma2"], theta["s"]
    M = grid_size(sigma2, s, nbar, _tables()) or M_FALLBACK
    return {"mu_log": float(np.log(nbar) - sigma2 / 2), "sigma2": sigma2, "s": s, "M": int(M)}


def fit_key(family: str, theta: dict) -> str:
    """Identical fits share one simulation set: this is their identity."""
    return f"{family}:" + json.dumps({k: round(v, 10) for k, v in sorted(theta.items())})


def rng_for(seed: int, case_id: str, model: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(case_id.encode()), zlib.crc32(model.encode())])


def simulate(family: str, kwargs: dict, k: int, rng, max_tries: int = 1000) -> list[np.ndarray]:
    """k patterns of the model, each with n in the bank's range."""
    kwargs = dict(kwargs)
    if family == "lgcp":
        kwargs["root_lam"] = lgcp_eigenvalues(kwargs["sigma2"], kwargs["s"], kwargs["M"])
    lo, hi = n_range()
    out = []
    for _ in range(k):
        for _ in range(max_tries):
            pts = SAMPLERS[family](rng, **kwargs)
            if lo <= len(pts) <= hi:
                out.append(pts)
                break
        else:
            raise RuntimeError(f"{family} {kwargs}: no pattern with n in [{lo}, {hi}]")
    return out
