"""Feature tables for table learners: a registry of sources, each a function (spec, family) -> the
family's (case_id, X, names). An input in the config names a source plus its options:

    inputs:
      classical: {source: classical}
      ph_dtm10: {source: ph, filtration: dtm_k10}

To add a source, write a loader and register it in SOURCES; any model can then list it in `inputs`.
Tables are joined to the row table on case_id, so a source may store rows in any order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core import DATA, ROOT


def _npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not path.exists():
        raise SystemExit(f"{path} missing -- see oneshot/README.md (prepare)")
    z = np.load(path)
    return z["case_id"], z["X"], list(z["names"])


def classical(spec: dict, family: str):
    """cascade/features.py's 111 summary-function samples (python cascade/features.py --family F)."""
    return _npz(ROOT / "data" / "cascade" / "features" / f"{family}.npz")


def ph_path(filtration: str, family: str) -> Path:
    return DATA / "features" / f"ph_{filtration}" / f"{family}.npz"


def ph(spec: dict, family: str):
    """Persistence summaries of one filtration (oneshot/prepare.py ph)."""
    return _npz(ph_path(spec["filtration"], family))


SOURCES = {"classical": classical, "ph": ph}


def load(cfg: dict, names: list[str], rows: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """The named inputs, concatenated column-wise, aligned to `rows` (by case_id)."""
    blocks, columns = [], []
    for name in names:
        spec = cfg["inputs"][name]
        parts = []
        for family in cfg["families"]:
            case_id, X, cols = SOURCES[spec["source"]](spec, family)
            parts.append(pd.DataFrame(X, index=case_id))
        table = pd.concat(parts)
        at = table.index.get_indexer(rows.index)
        if (at < 0).any():
            raise SystemExit(f"input {name}: {(at < 0).sum()} rows missing (e.g. {rows.index[at < 0][0]})")
        blocks.append(table.to_numpy(np.float32)[at])
        columns += [f"{name}:{c}" for c in cols]
    return np.hstack(blocks), columns
