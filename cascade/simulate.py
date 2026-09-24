"""(family, theta) -> simulated patterns, with the bank's own samplers on W = [0, 1]^2.

Two ways in:
  from_estimate(family, theta_hat)   the pipeline's output, keyed by cloudforger's TARGETS
  from_manifest(row)                 the TRUE model of a bank row (the evaluation's `oracle`)
Both return sampler kwargs, which `patterns` turns into point sets.

Choices (and where to change them):
  - LGCP is estimated as (nbar, sigma2, s), the training targets; the sampler wants
    mu_log = log(nbar) - sigma2 / 2 and a grid size M, which is picked exactly as the bank picks it
    (simulation.lgcp_grid.grid_size). An estimate with no admissible grid falls back to M_FALLBACK.
  - condition_n: patterns are redrawn until n lies in the bank's range [20, 2000], the same
    conditioning every bank pattern had, so a simulated pattern is a draw from the same thing the
    observed one is. Off: plain unconditional draws.
  - Seeding: a pattern's rng is (seed, case, model, index), never the bank's streams, so a
    simulation can never reproduce the observed pattern itself.
"""

from __future__ import annotations

import inspect
import zlib
from functools import lru_cache

import numpy as np

import common  # noqa: F401  (puts src/ on the path)
from cloudforger.simulation.bank import Config as BankConfig
from cloudforger.simulation.lgcp_grid import grid_size
from cloudforger.simulation.processes import SAMPLERS, lgcp_eigenvalues

M_FALLBACK = 1024


@lru_cache(maxsize=1)
def _tables():
    from cloudforger.departure.tables import Tables
    return Tables()


@lru_cache(maxsize=1)
def n_range() -> tuple[int, int]:
    return BankConfig.load().n_range


def sampler_args(family: str) -> list[str]:
    return [a for a in inspect.signature(SAMPLERS[family]).parameters if a not in ("rng", "root_lam")]


def from_estimate(family: str, theta: dict) -> dict:
    if family != "lgcp":
        return {a: theta[a] for a in sampler_args(family)}
    nbar, sigma2, s = theta["nbar"], theta["sigma2"], theta["s"]
    M = grid_size(sigma2, s, nbar, _tables()) or M_FALLBACK
    return {"mu_log": float(np.log(nbar) - sigma2 / 2), "sigma2": sigma2, "s": s, "M": int(M)}


def from_manifest(row) -> dict:
    kw = {a: row[a] for a in sampler_args(row["family"])}
    if "M" in kw:
        kw["M"] = int(kw["M"])
    return kw


def rng_for(seed: int, case_id: str, model: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(case_id.encode()), zlib.crc32(model.encode())])


def patterns(family: str, kwargs: dict, k: int, rng, condition_n: bool = True,
             max_tries: int = 1000) -> list[np.ndarray]:
    kwargs = dict(kwargs)
    if family == "lgcp":
        kwargs["root_lam"] = lgcp_eigenvalues(kwargs["sigma2"], kwargs["s"], kwargs["M"])
    lo, hi = n_range() if condition_n else (0, np.inf)
    out = []
    for _ in range(k):
        for _ in range(max_tries):
            pts = SAMPLERS[family](rng, **kwargs)
            if lo <= len(pts) <= hi:
                out.append(pts)
                break
        else:
            raise RuntimeError(f"{family} {kwargs}: no pattern with n in [{lo}, {hi}] in {max_tries} tries")
    return out
