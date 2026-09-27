"""Shared paths and readers for extensions/: exploratory runs beside the paper version.

The paper version is oneshot/ and its results in oneshot/results/default/. Nothing here writes
there, and nothing here imports oneshot/ or cascade/*.py (both are still being edited): the paper
run's outputs are read as files, and the two helpers taken from cascade/evaluate.py are copied
below. The only code imported from outside extensions/ is cloudforger (src/) and the scoring
package cascade/scoring/, which is self-contained by design.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extensions"
PAPER = ROOT / "oneshot" / "results" / "default"            # read only
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cascade" / "scoring"))

from cloudforger.simulation.bank import DATA as BANK        # noqa: E402
from cloudforger.simulation.split import split_of           # noqa: E402
from cloudforger.training.data import TARGETS               # noqa: E402

FAMILIES = ["poisson", "thomas", "nested", "lgcp", "matern2", "ring", "matern1", "cell"]


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def results_dir(*parts: str) -> Path:
    out = EXT.joinpath("results", *parts)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


# ------------------------------------------------------------------------------------ bank rows

def rows(families=FAMILIES, thin: int = 1) -> pd.DataFrame:
    """Bank manifest rows of `families`, indexed by case_id, with cloudforger's theta split."""
    out = pd.concat([pd.read_csv(BANK / f / "manifest.csv") for f in families], ignore_index=True)
    out["split"] = split_of(out.theta.to_numpy())
    out = out[out.theta % thin == 0]
    return out.set_index("case_id")


def points(chosen: pd.DataFrame) -> dict[str, np.ndarray]:
    """case_id -> (n, 2) float32 coordinates in the unit square."""
    out = {}
    for family, r in chosen.groupby("family"):
        index = pd.read_csv(BANK / family / "manifest.csv", usecols=["case_id"]).case_id
        pos = pd.Series(np.arange(len(index)), index=index).loc[r.index].to_numpy()
        z = np.load(BANK / family / "points.npz")
        P, off = z["points"], z["offsets"]
        out |= {c: P[off[i]:off[i + 1]].astype(np.float32) for c, i in zip(r.index, pos)}
    return out


# ------------------------------------------------------------------------ paper-run predictions

def paper_classifier(model: str) -> pd.DataFrame:
    """The paper run's class posteriors (val + test rows), one column per family."""
    z = np.load(PAPER / "classify" / model / "predictions.npz")
    return pd.DataFrame(z["posterior"], columns=[str(c) for c in z["classes"]],
                        index=pd.Index(z["case_id"], name="case_id"))


def paper_estimator(family: str, model: str) -> pd.DataFrame:
    """The paper run's theta_hat from `model` for `family`, on every val + test row (any family)."""
    z = np.load(PAPER / "estimate" / family / model / "predictions.npz")
    return pd.DataFrame(z["theta_hat"], columns=[str(c) for c in z["targets"]],
                        index=pd.Index(z["case_id"], name="case_id"))


# ------------------------------------------------- copied from cascade/evaluate.py at 7691ae42

M_FALLBACK = 1024          # LGCP grid when an estimate has no admissible one (bank rule, lgcp_grid)


@lru_cache(maxsize=1)
def _tables():
    from cloudforger.departure.tables import Tables
    return Tables()


def sampler_kwargs(family: str, theta: dict) -> dict:
    """theta_hat (keyed by the TARGETS) -> the bank sampler's arguments."""
    if family == "cell":                                         # k is a count; estimates are real
        return {"nbar": theta["nbar"], "k": int(np.clip(round(theta["k"]), 2, 30))}
    if family != "lgcp":
        return dict(theta)
    from cloudforger.simulation.lgcp_grid import grid_size
    nbar, sigma2, s = theta["nbar"], theta["sigma2"], theta["s"]
    M = grid_size(sigma2, s, nbar, _tables()) or M_FALLBACK
    return {"mu_log": float(np.log(nbar) - sigma2 / 2), "sigma2": sigma2, "s": s, "M": int(M)}
