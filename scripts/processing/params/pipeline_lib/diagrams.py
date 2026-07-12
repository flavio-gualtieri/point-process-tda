# scripts/processing/params/pipeline_lib/diagrams.py
"""Stage: compute_diagrams — persistence diagrams from point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.tda.filtration.base import Filtration
from cloudforger.tda.filtration.rips import RipsFiltration
from cloudforger.tda.filtration.dtm import DTMFiltration

from pipeline_lib.io import build_labels, dump_pickle, load_items
from pipeline_lib.records import cloud_params, cloud_seed, diagram_to_record, to_pointcloud


def build_filtration(config: dict[str, Any]) -> Filtration:
    filtration_cfg = config["filtration"]
    if filtration_cfg["type"] == "rips":
        return RipsFiltration(
            maxdim=int(filtration_cfg.get("maxdim", 1)),
            thresh=filtration_cfg.get("thresh", None),
        )
    else:
        return DTMFiltration(
            maxdim=int(filtration_cfg.get("maxdim", 1)),
            k=filtration_cfg.get("k", 5),
            q=filtration_cfg.get("q", 2),
            thresh=filtration_cfg.get("thresh", None),
        )


def compute_diagrams_from_clouds(
    clouds_path: Path,
    out_path: Path,
    filtration: Filtration,
    process: str,
    diagram_format: str,
) -> None:
    clouds = load_items(clouds_path, "clouds")
    labels, label_names, params = build_labels(clouds, cloud_params, "clouds")
    seeds = [cloud_seed(c) for c in clouds]

    computed: list[PersistenceDiagram] = []
    print(f"  Computing {len(clouds)} diagrams from {clouds_path} ...")

    for i, cloud in enumerate(clouds):
        computed.append(filtration.compute(to_pointcloud(cloud)))
        print(f"\r    {i + 1}/{len(clouds)}", end="", flush=True)
    print()

    diagrams = [diagram_to_record(d) for d in computed] if diagram_format == "dict" else computed
    dump_pickle(
        out_path,
        {
            "diagrams": diagrams,
            "labels": labels,
            "label_names": label_names,
            "params": params,
            "seeds": seeds,
            "process": process,
            "filtration_params": filtration.params,
        },
    )
    print(f"  Saved {len(diagrams)} diagrams → {out_path}")
