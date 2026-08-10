# src/cloudforger/experiments/raw_pc.py

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pathlib import Path

from cloudforger.core.cloud import PointCloud
from cloudforger.core.region import Box
from cloudforger.training.data import PointCloudDataset, pad_point_cloud_collate
from cloudforger.experiments.base import Experiment, register, log_zscore_fit_once


def _cloud_list(payload: Any):
    if isinstance(payload, dict) and "clouds" in payload:
        return payload["clouds"]
    return payload


def _to_pointcloud(cloud) -> PointCloud:
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


def _scalarize_params(params: dict) -> dict[str, float]:
    """Expand vector-valued params, e.g. beta -> beta_0, beta_1, ..."""
    out: dict[str, float] = {}
    for key, value in params.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            out[key] = float(arr)
        else:
            for j, item in enumerate(arr.ravel()):
                out[f"{key}_{j}"] = float(item)
    return out


def _labels_from_clouds(cloud_list) -> tuple[np.ndarray, list[str]]:
    def params_of(c):
        raw = dict(c["params"]) if isinstance(c, dict) else dict(c.generator_params)
        return _scalarize_params(raw)

    params = [params_of(c) for c in cloud_list]
    names = list(params[0].keys())

    for p in params:
        if list(p.keys()) != names:
            raise ValueError("Clouds have inconsistent parameter keys; cannot stack labels.")

    labels = np.array([[p[k] for k in names] for p in params], dtype=float)
    return labels, names


@register("raw_pc")
class RawPointCloudExperiment(Experiment):
    file_key = "raw_pc"
    subdir = "raw_pc"
    collate_fn = staticmethod(pad_point_cloud_collate)

    def __init__(self, cfg: dict, hom_dim: int | None = None):
        super().__init__(cfg, hom_dim)
        self.ambient_dim: int | None = None

    @property
    def tag(self) -> str:
        return f"raw_pc(dim={self.ambient_dim})"

    @property
    def extra_meta(self) -> dict:
        meta: dict[str, Any] = {"ambient_dim": self.ambient_dim}
        norm = getattr(self, "_n_points_norm", None)
        if norm is not None:
            meta["n_points_log_mean"] = norm["mean"]
            meta["n_points_log_std"] = norm["std"]
        return meta

    def extract_labels(self, payload):
        return _labels_from_clouds(_cloud_list(payload))

    def build_dataset(self, payload, labels):
        clouds = [_to_pointcloud(c) for c in _cloud_list(payload)]
        self.ambient_dim = clouds[0].dimension
        return PointCloudDataset(clouds, labels, n_points=None, dtype=torch.float32)

    def extract_head_extra(self, payload, dataset_path: Path) -> np.ndarray:
        # Unlike betti/pi/pairwise, n(x) is already in this experiment's own
        # payload (each cloud record carries "n_points") -- no sibling-file
        # join needed. Worth having explicitly regardless: PointNetEncoder
        # max-pools over per-point features, which is cardinality-invariant
        # by construction, so this encoder architecturally cannot recover
        # n(x) from the points themselves the way a curve/image encoder can
        # (weakly) infer it from density.
        clouds = _cloud_list(payload)
        n_points = np.array([c["n_points"] for c in clouds], dtype=np.float64)
        return log_zscore_fit_once(self, n_points, attr="_n_points_norm")

    def build_encoder(self, dataset):
        from cloudforger.encoders.point_cloud import PointNetEncoder

        return PointNetEncoder(
            input_dim=self.ambient_dim,
            embedding_dim=self.cfg["embedding_dim"],
            hidden_dims=self.cfg["hidden_dims"],
        )