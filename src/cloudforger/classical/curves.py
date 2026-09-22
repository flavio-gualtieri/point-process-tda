"""Summary-function curves for every simulated pattern, one family and grid at a time.

Input   data/bank/<family>/{points.npz, manifest.csv}
Output  data/classical/<family>/<tag>/curves.npz, rows in manifest.csv order:
            case_id          (P,)
            L, F, G, J       (P, 512) float32
            axis             (512,)  r under `fixed`, u = r sqrt(n) under `sqrtn`
            spec             the config entry, as json

Curves are fixed length, so unlike the diagrams there is nothing ragged here: no sharding, no merge
step and no worker pool. One pass over a family is about a minute (L dominates, O(n^2) in the pairs
within r_max), and all five families take roughly six.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..featurization.sweep import BANK, families
from .functions import GRID_SIZE, NAMES, axis, curves, tag

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = PROJECT_ROOT / "configs" / "classical" / "config.yaml"
DATA = PROJECT_ROOT / "data" / "classical"

__all__ = ["Config", "DATA", "families", "run", "tag"]


@dataclass(frozen=True)
class Config:
    grids: tuple[dict, ...]
    f_grid_size: int

    @classmethod
    def load(cls, path: Path = CONFIG) -> Config:
        c = yaml.safe_load(path.read_text())
        return cls(tuple(c["grids"]), int(c["f_grid_size"]))

    def spec(self, tag_: str) -> dict:
        return next(s for s in self.grids if tag(s) == tag_)


def run(family: str, tag_: str, cfg: Config) -> Path:
    """Every pattern of one family on one grid. Skips the work if the output already exists."""
    path = DATA / family / tag_ / "curves.npz"
    if path.exists():
        return path
    spec = cfg.spec(tag_)
    case_id = pd.read_csv(BANK / family / "manifest.csv").case_id.to_numpy(str)
    z = np.load(BANK / family / "points.npz")
    points, offsets = z["points"], z["offsets"]
    if len(offsets) - 1 != len(case_id):
        raise SystemExit(f"{family}: {len(offsets) - 1} patterns for {len(case_id)} manifest rows")

    out = {name: np.empty((len(case_id), GRID_SIZE), dtype=np.float32) for name in NAMES}
    for i in range(len(case_id)):
        for name, curve in curves(points[offsets[i]:offsets[i + 1]], spec, cfg.f_grid_size).items():
            out[name][i] = curve

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, case_id=case_id, axis=axis(spec), spec=np.array(json.dumps(spec)), **out)
    tmp.replace(path)
    return path


def load(family: str, tag_: str, name: str) -> np.ndarray:
    """One family's curves for one function, (P, 512) float32 in manifest order."""
    return np.load(DATA / family / tag_ / "curves.npz")[name]
