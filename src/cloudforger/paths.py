"""Where everything lives.

Configs are part of the repo. Data and results sit under two roots, each overridable from the
environment, so a rerun can write to a fresh root and never touch an earlier one:

    CLOUDFORGER_DATA      default <repo>/data
        bank/<family>/{points.npz, manifest.csv}           scripts/simulate.py (+ relabel.py: delta-tilde)
        departure/                                         scripts/departure.py simulate (CSR null curves)
        featurization/<family>/<tag>/diagrams.npz          scripts/featurize.py
        classical/<family>/<grid>/curves.npz               scripts/classical.py
        tables/<source>/<family>.npz                       scripts/tables.py (table-learner inputs)
        mincontrast/<run>/                                 scripts/mincontrast.py clouds

    CLOUDFORGER_RESULTS   default <repo>/results
        <run>/classify/<model>/, <run>/estimate/<family>/<model>/     scripts/train.py
        <run>/{compare, evaluation/<set>, mincontrast}/              compare / endtoend / mincontrast
        scores/<name>/                                                scripts/power.py
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = Path(os.environ.get("CLOUDFORGER_DATA", ROOT / "data")).absolute()
RESULTS = Path(os.environ.get("CLOUDFORGER_RESULTS", ROOT / "results")).absolute()

BANK = DATA / "bank"
DEPARTURE = DATA / "departure"
DIAGRAMS = DATA / "featurization"
CURVES = DATA / "classical"
TABLES = DATA / "tables"
MINCONTRAST = DATA / "mincontrast"
