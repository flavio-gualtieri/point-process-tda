"""What the pilot's steps share: config, paths, the row table, targets, coordinates and weights.

The pilot imports cloudforger and cascade/ read-only and writes only under pilot/ and data/pilot/.
Its module names avoid cascade's (common, features, regime, ...), since both directories sit on
sys.path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cascade"))

from cloudforger.simulation.bank import DATA as BANK        # noqa: E402

CONFIG = Path(os.environ.get("PILOT_CONFIG", ROOT / "pilot" / "config.yaml"))   # smoke: pilot/smoke.yaml
NAME = yaml.safe_load(CONFIG.read_text())["name"]
DATA = ROOT / "data" / "pilot" / NAME
RESULTS = ROOT / "pilot" / "results" / NAME

# The family's own parameters (bank families as cascade.common.TARGETS).
TARGETS = {
    "thomas": ["kappa", "mu", "sigma"],
    "nested": ["kappa", "mu1", "mu2", "sigma1", "sigma2"],
    "matern2": ["R", "lam_p"],
    "lgcp": ["nbar", "sigma2", "s"],
    "ring": ["kappa", "mu", "rho", "sigma"],
    "matern1": ["R", "lam_p"],
    "cell": ["nbar", "k"],
}

# Regime coordinates for the new families, in cascade.regime.DERIVED's style (bank families use it).
DERIVED = {
    "ring": {"omega": lambda r: r.rho * np.sqrt(r.kappa),              # ring overlap
             "jitter": lambda r: r.sigma / r.rho},
    "matern1": {"core": lambda r: r.R * np.sqrt(r.nbar),
                "zeta": lambda r: r.R * r.nbar},
}


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def families(cfg: dict) -> list[str]:
    return cfg["bank_families"] + cfg["new_families"]


def source(cfg: dict, family: str) -> Path:
    """Where a family's points.npz + manifest.csv live."""
    return BANK / family if family in cfg["bank_families"] else DATA / "bank" / family


def split_of(cfg: dict, family: str, theta: np.ndarray) -> np.ndarray:
    blocks = cfg["split"]["bank" if family in cfg["bank_families"] else "new"]
    out = np.full(len(theta), "", dtype=object)
    for name, (lo, hi) in blocks.items():
        out[(theta >= lo) & (theta < hi)] = name
    return out


def rows(cfg: dict) -> pd.DataFrame:
    """Every pilot row of every family, indexed by case_id, with split and parameter columns."""
    parts = []
    for f in families(cfg):
        m = pd.read_csv(source(cfg, f) / "manifest.csv")
        m["split"] = split_of(cfg, f, m.theta.to_numpy())
        parts.append(m[m.split != ""])
    return pd.concat(parts, ignore_index=True).set_index("case_id")


def coordinate(rows_: pd.DataFrame, family: str, name: str) -> np.ndarray:
    from regime import values                                   # cascade/regime.py
    fn = DERIVED.get(family, {}).get(name)
    return np.asarray(fn(rows_), float) if fn else values(rows_, family, name)


def group_weights(family: pd.Series, groups: dict) -> np.ndarray:
    """balanced_groups over the families in `groups`: 1/3 per group, split evenly within a group,
    divided by each family's row count. Mean 1 over the rows given."""
    sizes = pd.Series(groups).value_counts()
    counts = family.value_counts()
    w = family.map(lambda f: 1 / (len(sizes) * sizes[groups[f]] * counts[f])).to_numpy(float)
    return w * len(w) / w.sum()


def equal_weights(family: pd.Series) -> np.ndarray:
    """Every family the same total weight. Mean 1."""
    w = 1 / family.map(family.value_counts()).to_numpy(float)
    return w * len(w) / w.sum()
