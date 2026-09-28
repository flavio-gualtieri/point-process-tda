from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..paths import CONFIGS, DEPARTURE as DATA  # noqa: F401  (DATA is re-exported to simulate.py)

CONFIG = CONFIGS / "departure.yaml"
TABLES = CONFIGS / "departure_tables.npz"          # fitted null tables, shipped with the repo
REPORT = CONFIGS / "departure_report.json"


@dataclass(frozen=True)
class Config:
    root: int
    n_low: int
    n_high: int
    n_step: float
    fit: int
    calibrate: int
    validation: int
    min_pairs: float
    alpha: float
    knots_moments: int
    knots_c95: int

    @property
    def reps(self) -> int:
        return self.fit + self.calibrate

    def grid(self) -> np.ndarray:
        k = np.arange(int(np.ceil(np.log(self.n_high / self.n_low) / np.log1p(self.n_step))) + 1)
        n = np.unique(np.round(self.n_low * (1 + self.n_step) ** k).astype(int))
        return np.append(n[n < self.n_high], self.n_high)


def load(path: Path = CONFIG) -> Config:
    c = yaml.safe_load(path.read_text())
    return Config(
        root=int(c["root"]), n_low=int(c["n"]["low"]), n_high=int(c["n"]["high"]), n_step=float(c["n"]["step"]),
        fit=int(c["reps"]["fit"]), calibrate=int(c["reps"]["calibrate"]), validation=int(c["validation"]),
        min_pairs=float(c["min_pairs"]), alpha=float(c["alpha"]),
        knots_moments=int(c["knots"]["moments"]), knots_c95=int(c["knots"]["c95"]),
    )
