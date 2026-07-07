# scripts/processing/params/pipeline_lib/records.py
"""Conversions between cloudforger domain objects and their pickled dict records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cloudforger.core.betti import BettiCurveFeature
from cloudforger.core.cloud import PointCloud
from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.core.features import CorrelationFeatures
from cloudforger.core.region import Box

from pipeline_lib.io import load_pickle


# ---------------------------------------------------------------------------
# Point clouds
# ---------------------------------------------------------------------------


def cloud_to_record(cloud: PointCloud) -> dict[str, Any]:
    record: dict[str, Any] = {
        "points": cloud.points,
        "params": dict(cloud.generator_params),
        "process": cloud.generator_name,
        "seed": cloud.seed,
        "n_points": cloud.n_points,
        "dimension": cloud.dimension,
    }

    if isinstance(cloud.region, Box):
        record["region"] = {"low": cloud.region.low, "high": cloud.region.high}

    return record


def to_pointcloud(cloud: Any) -> PointCloud:
    if not isinstance(cloud, dict):
        return cloud

    region = None
    reg = cloud.get("region")
    if reg is not None:
        try:
            region = Box(
                low=np.asarray(reg["low"], dtype=float),
                high=np.asarray(reg["high"], dtype=float),
            )
        except Exception:
            region = None

    return PointCloud(
        points=np.asarray(cloud["points"]),
        generator_name=cloud.get("process", ""),
        generator_params=dict(cloud.get("params", {})),
        seed=cloud.get("seed"),
        region=region,
    )


def cloud_params(cloud: Any) -> dict[str, Any]:
    return dict(cloud["params"]) if isinstance(cloud, dict) else dict(cloud.generator_params)


def cloud_seed(cloud: Any) -> Any:
    return cloud.get("seed") if isinstance(cloud, dict) else cloud.seed


# ---------------------------------------------------------------------------
# Persistence diagrams
# ---------------------------------------------------------------------------


def diagram_to_record(diagram: PersistenceDiagram) -> dict[str, Any]:
    return {
        "diagrams": {int(dim): np.asarray(pairs) for dim, pairs in diagram.diagrams.items()},
        "params": dict(diagram.generator_params),
        "seed": diagram.seed,
        "process": diagram.generator_name,
        "filtration": diagram.filtration_name,
        "filtration_params": dict(diagram.filtration_params),
    }


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


def diagram_params(diagram: Any) -> dict[str, Any]:
    return dict(diagram["params"]) if isinstance(diagram, dict) else dict(diagram.generator_params)


def diagram_seed(diagram: Any) -> Any:
    return diagram.get("seed") if isinstance(diagram, dict) else diagram.seed


def load_diagram_bundle(path: Path) -> dict[str, Any]:
    data = load_pickle(path)
    if isinstance(data, dict) and "diagrams" in data:
        return data
    return {"diagrams": data}


def load_diagrams(path: Path) -> tuple[list[PersistenceDiagram], dict[str, Any]]:
    bundle = load_diagram_bundle(path)
    return [as_diagram(d) for d in bundle["diagrams"]], bundle


# ---------------------------------------------------------------------------
# Betti curves
# ---------------------------------------------------------------------------


def betti_feature_to_record(feature: BettiCurveFeature) -> dict[str, Any]:
    return {
        "curves": {int(dim): np.asarray(curve) for dim, curve in feature.curves.items()},
        "vector": feature.vector(),
        "params": dict(feature.generator_params),
        "seed": feature.seed,
        "process": feature.generator_name,
        "filtration": feature.filtration_name,
        "filtration_params": dict(feature.filtration_params),
        "feature": feature.feature_name,
        "feature_params": dict(feature.feature_params),
    }


# ---------------------------------------------------------------------------
# Pairwise/correlation features
# ---------------------------------------------------------------------------


def features_to_record(cf: CorrelationFeatures) -> dict[str, Any]:
    return {
        "features": {k: np.asarray(v) for k, v in cf.features.items()},
        "params": dict(cf.generator_params),
        "seed": cf.seed,
        "process": cf.generator_name,
        "statistic_params": dict(cf.statistic_params),
    }
