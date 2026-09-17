# src/cloudforger/core/records.py
"""Reading pickled persistence-diagram bundles back into PersistenceDiagram objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .diagram import PersistenceDiagram
from .io import load_pickle


# ---------------------------------------------------------------------------
# Persistence diagrams
# ---------------------------------------------------------------------------


def record_to_diagram(record: dict[str, Any]) -> PersistenceDiagram:
    return PersistenceDiagram(
        diagrams={int(dim): np.asarray(pairs, dtype=float) for dim, pairs in record["diagrams"].items()},
        generator_name=record.get("process", ""),
        generator_params=dict(record.get("params", {})),
        seed=record.get("seed"),
        filtration_name=record.get("filtration", ""),
        filtration_params=dict(record.get("filtration_params", {})),
    )


def as_diagram(value: Any) -> PersistenceDiagram:
    if isinstance(value, PersistenceDiagram):
        return value
    if isinstance(value, dict):
        return record_to_diagram(value)
    raise TypeError(f"Expected PersistenceDiagram or dict, got {type(value)}.")


def load_diagram_bundle(path: Path) -> dict[str, Any]:
    data = load_pickle(path)
    if isinstance(data, dict) and "diagrams" in data:
        return data
    return {"diagrams": data}


def load_diagrams(path: Path) -> tuple[list[PersistenceDiagram], dict[str, Any]]:
    bundle = load_diagram_bundle(path)
    return [as_diagram(d) for d in bundle["diagrams"]], bundle
