# scripts/processing/params/pipeline_lib/betti.py
"""Stage: compute_betti — Betti curves from persistence diagrams."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cloudforger.core.betti import BettiCurveFeature
from cloudforger.tda.features.betti_curve import BettiCurve

from pipeline_lib.io import build_labels, dump_pickle, load_items
from pipeline_lib.records import as_diagram, betti_feature_to_record, diagram_params, diagram_seed


def build_betti(config: dict[str, Any]) -> BettiCurve:
    betti_cfg = config["betti"]
    # BettiCurve.__init__ only special-cases a `tuple` for homology_dims (a
    # plain list, e.g. from YAML, gets wrapped as a single unhashable
    # element instead of unpacked) -- coerce explicitly so both a Python
    # tuple literal and a YAML list work the same way.
    homology_dims = tuple(betti_cfg.get("homology_dims", (0, 1)))
    return BettiCurve(
        homology_dims=homology_dims,
        grid_size=betti_cfg.get("grid_size", 128),
        grid_range=betti_cfg.get("grid_range", (0.0, 1.0)),
        drop_infinite=betti_cfg.get("drop_infinite", True),
        normalize=betti_cfg.get("normalize", False),
    )


def compute_betti_from_diagrams(
    diagrams_path: Path,
    out_path: Path,
    betti: BettiCurve,
    process: str,
    curve_format: str,
) -> None:
    diagrams = load_items(diagrams_path, "diagrams")
    labels, label_names, params = build_labels(diagrams, diagram_params, "diagrams")
    seeds = [diagram_seed(d) for d in diagrams]

    computed: list[BettiCurveFeature] = []
    print(f"  Computing {len(diagrams)} Betti curves from {diagrams_path} ...")

    for i, d in enumerate(diagrams):
        computed.append(betti.compute(as_diagram(d)))
        print(f"\r    {i + 1}/{len(diagrams)}", end="", flush=True)
    print()

    curves = [betti_feature_to_record(c) for c in computed] if curve_format == "dict" else computed
    homology_dims = list(getattr(betti, "homology_dims", betti.params.get("homology_dims", (0, 1))))
    betti_matrices = {int(dim): np.stack([feature.curves[dim] for feature in computed]) for dim in homology_dims}

    payload: dict[str, Any] = {
        "curves": curves,
        "betti_matrices": betti_matrices,
        "labels": labels,
        "label_names": label_names,
        "params": params,
        "seeds": seeds,
        "process": process,
        "betti_params": betti.params,
    }

    # Keep the historical convenience keys for downstream code that expects them.
    if 0 in betti_matrices:
        payload["betti0_matrix"] = betti_matrices[0]
    if 1 in betti_matrices:
        payload["betti1_matrix"] = betti_matrices[1]

    dump_pickle(out_path, payload)
    print(f"  Saved {len(curves)} Betti curves → {out_path}")
