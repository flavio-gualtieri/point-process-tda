# model/processing.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pickle
from typing import Any

from cloudforger.core.betti import BettiCurveFeature
from cloudforger.core.cloud import PointCloud
from cloudforger.core.region import Box
from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.tda.features.betti_curve import BettiCurve
from cloudforger.tda.filtration.rips import RipsFiltration


def load_cloud(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    # process_barro.py writes one cloud per file as a 1-element list,
    # matching the training container schema.
    if isinstance(data, list):
        data = data[0]
    return data


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
        generator_name=cloud.get("process") or "",
        generator_params=dict(cloud.get("params") or {}),
        seed=cloud.get("seed"),
        region=region,
    )


# Mean point count across data/params/2d/thomas/clouds.pkl, the training set
# this real-data crop is meant to be density-comparable to.
TRAIN_TARGET_N = 900


def sample_unit_box_window(
        point_cloud: PointCloud,
        target_n: int = TRAIN_TARGET_N,
        rng: np.random.Generator | None = None,
) -> PointCloud:
    """Crop a random square window sized to hold ~target_n points, rescaled to (0,1)^d.

    The raw cloud is far denser than the training clouds it's compared against
    (e.g. the BCI census has ~235k points vs. a training mean of ~900), so a
    naive clip to the existing (0,1) region is a no-op and blows up Rips.
    Instead we pick a window whose area matches the training density, crop it,
    and rescale it onto the unit box.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    region = point_cloud.region
    if region is None:
        low = point_cloud.points.min(axis=0)
        high = point_cloud.points.max(axis=0)
        region = Box(low=low, high=high)

    extent = region.high - region.low
    density = point_cloud.n_points / region.volume
    window = float(np.sqrt(target_n / density))
    window = min(window, float(extent.min()))

    slack = extent - window
    corner = region.low + rng.uniform(0.0, 1.0, size=extent.shape) * slack

    mask = np.all(
        (point_cloud.points >= corner) & (point_cloud.points <= corner + window),
        axis=1,
    )
    rescaled = (point_cloud.points[mask] - corner) / window

    return PointCloud(
        points=rescaled,
        generator_name=point_cloud.generator_name,
        generator_params=point_cloud.generator_params,
        seed=point_cloud.seed,
        region=Box(low=np.zeros(point_cloud.dimension), high=np.ones(point_cloud.dimension)),
    )


def diagram_to_record(diagram: PersistenceDiagram) -> dict[str, Any]:
    return {
        "diagrams": {
            int(dim): np.asarray(pairs)
            for dim, pairs in diagram.diagrams.items()
        },
        "params": dict(diagram.generator_params),
        "seed": diagram.seed,
        "process": diagram.generator_name,
        "filtration": diagram.filtration_name,
        "filtration_params": dict(diagram.filtration_params),
    }


def compute_diagrams(
        cloud_path: Path,
        out_path: Path,
        filtration: RipsFiltration, process: str,
        diagram_format: str = "dict"
) -> None:
    cloud = load_cloud(cloud_path)
    if isinstance(cloud, dict) and not cloud.get("process"):
        cloud = {**cloud, "process": process}

    print(f"  Computing persistence diagram from {cloud_path} ...")
    point_cloud = to_pointcloud(cloud)
    point_cloud = sample_unit_box_window(point_cloud)
    computed = filtration.compute(point_cloud)
    print(f"\r    Done", end="")
    print()

    diagram = diagram_to_record(computed) if diagram_format == "dict" else computed

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "diagrams": diagram,
                "labels": None,
                "label_names": None,
                "params": None,
                "seeds": None,
                "process": process,
                "filtration_params": filtration.params,
            },
            f,
        )

    print(f"  Saved diagram → {out_path}")
    return computed


def betti_features_to_record(curve: BettiCurveFeature) -> dict[str, Any]:
    return {
        "curves": {
            int(dim): np.asarray(values)
            for dim, values in curve.curves.items()
        },
        "vector": curve.vector(),
        "params": dict(curve.generator_params),
        "seed": curve.seed,
        "process": curve.generator_name,
        "filtration": curve.filtration_name,
        "filtration_params": dict(curve.filtration_params),
        "feature": curve.feature_name,
        "feature_params": dict(curve.feature_params),
    }


def compute_curves(
        diagram: PersistenceDiagram,
        out_path: Path,
        process: str,
        curve_format: str = "dict"
) -> None:
    betti = BettiCurve(
        homology_dims=0,
    )
    print(f"  Computing betti_0 curve ...")

    computed = betti.compute(diagram)
    print(f"\r    Done", end="")
    print()

    curve = betti_features_to_record(computed) if curve_format == "dict" else computed

    betti0_curve = computed.curves[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "curve": curve,
                "betti0_curve": betti0_curve,
                "params": dict(computed.generator_params),
                "seed": computed.seed,
                "process": process,
                "betti_params": betti.params,
            },
            f,
        )

    print(f"  Saved betti_0 curve → {out_path}")


def main() -> None:
    for census in range(1, 9):
        cloud_path = Path(
            f"/Users/qp252676/Desktop/point-process-tda/data/barro/bci.tree{census}_cloud.pkl"
        )
        diagram_path = Path(
            f"/Users/qp252676/Desktop/point-process-tda/data/barro/bci.tree{census}_diagram.pkl"
        )
        betti0_path = Path(
            f"/Users/qp252676/Desktop/point-process-tda/data/barro/bci.tree{census}_betti0.pkl"
        )
        filtration = RipsFiltration(maxdim=1, thresh=None)
        diagram = compute_diagrams(
            cloud_path=cloud_path,
            out_path=diagram_path,
            filtration=filtration,
            process="barro",
        )
        compute_curves(
            diagram=diagram,
            out_path=betti0_path,
            process="barro",
        )


if __name__ == "__main__":
    main()
