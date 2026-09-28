"""Where everything lives.

Configs are part of the repo (configs/). CLOUDFORGER_CONFIGS may name a directory whose files replace
the configs/ files of the same name -- configs/smoke/ is one -- and everything it lacks falls back to
configs/. Data and results sit under two roots, each overridable from the environment, so a rerun
can write to a fresh root and never touch an earlier one:

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

OVERRIDES = Path(os.environ["CLOUDFORGER_CONFIGS"]).absolute() if os.environ.get("CLOUDFORGER_CONFIGS") else None


def config_file(name: str) -> Path:
    """A non-YAML file of configs/ (the null tables): <CLOUDFORGER_CONFIGS>/<name> replaces it when present."""
    if OVERRIDES and (OVERRIDES / name).exists():
        return OVERRIDES / name
    return CONFIGS / name


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def read_config(path: str | Path) -> dict:
    """A YAML config as a dict; a bare name means configs/<name>. For a file in configs/, the file of
    the same name in CLOUDFORGER_CONFIGS (if any) is merged on top: mappings merge key by key, and
    anything else (lists, numbers, strings) is replaced."""
    import yaml
    path = Path(path)
    if path.parent == Path("."):
        path = CONFIGS / path
    cfg = yaml.safe_load(path.read_text())
    over = OVERRIDES / path.name if OVERRIDES else None
    if over is not None and over.exists() and path.resolve().parent == CONFIGS.resolve():
        cfg = _merge(cfg, yaml.safe_load(over.read_text()) or {})
    return cfg


BANK = DATA / "bank"
DEPARTURE = DATA / "departure"
DIAGRAMS = DATA / "featurization"
CURVES = DATA / "classical"
TABLES = DATA / "tables"
MINCONTRAST = DATA / "mincontrast"
