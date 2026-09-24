"""Bank access and simulation for the scoring harness: observed patterns, true models, CSR.

Self-contained on purpose: this package imports cloudforger and reads data/bank/, and nothing
from the rest of cascade/, which is being changed independently. The simulation logic mirrors
cascade/simulate.py (the bank's own samplers, the bank's n-conditioning, never the bank's seeds).
"""

from __future__ import annotations

import inspect
import sys
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.simulation.bank import DATA as BANK, Config as BankConfig    # noqa: E402
from cloudforger.simulation.processes import SAMPLERS, lgcp_eigenvalues       # noqa: E402
from cloudforger.simulation.split import split_of                             # noqa: E402

FAMILIES = ("poisson", "thomas", "nested", "lgcp", "matern2")


@lru_cache(maxsize=1)
def n_range() -> tuple[int, int]:
    return BankConfig.load().n_range


def pick(per_bin: int, delta_edges: list[float], seed: int) -> pd.DataFrame:
    """Test-split clouds (replicate 0), stratified by delta-tilde: `per_bin` per (family, bin) for
    the structured families, and per_bin * (number of bins) Poisson clouds as the noise floor."""
    frames, n_bins = [], len(delta_edges) - 1
    for family in FAMILIES:
        m = pd.read_csv(BANK / family / "manifest.csv")
        m = m[(split_of(m.theta.to_numpy()) == "test") & (m.rep == 0)]
        if family == "poisson":
            frames.append(m.sample(per_bin * n_bins, random_state=seed))
            continue
        bins = pd.cut(m.delta_tilde, delta_edges, right=False)
        frames += [g.sample(min(per_bin, len(g)), random_state=seed) for _, g in m.groupby(bins, observed=True)]
    return pd.concat(frames).set_index("case_id")


def observed(chosen: pd.DataFrame) -> dict[str, np.ndarray]:
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
    family = row["family"]
    args = [a for a in inspect.signature(SAMPLERS[family]).parameters if a not in ("rng", "root_lam")]
    kw = {a: row[a] for a in args}
    if "M" in kw:
        kw["M"] = int(kw["M"])
    return family, kw


def rng_for(seed: int, case_id: str, model: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(case_id.encode()), zlib.crc32(model.encode())])


def simulate(family: str, kwargs: dict, k: int, rng, max_tries: int = 1000) -> list[np.ndarray]:
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
