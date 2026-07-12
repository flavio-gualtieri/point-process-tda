# src/cloudforger/nn/experiments/pairwise.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from cloudforger.nn.data import CorrelationFeatureDataset
from cloudforger.nn.experiments.base import Experiment, register, n_points_head_extra


@register("pairwise")
class PairwiseExperiment(Experiment):
    file_key = "pairwise"
    subdir = "pairwise"

    @property
    def extra_meta(self) -> dict:
        meta: dict[str, Any] = {}
        norm = getattr(self, "_n_points_norm", None)
        if norm is not None:
            meta["n_points_log_mean"] = norm["mean"]
            meta["n_points_log_std"] = norm["std"]
        return meta

    def build_dataset(self, payload, labels):
        return CorrelationFeatureDataset(payload["features"], labels, dtype=torch.float32)

    def extract_head_extra(self, payload, dataset_path: Path) -> np.ndarray:
        return n_points_head_extra(self, payload, dataset_path)

    def build_encoder(self, dataset):
        from cloudforger.nn.encoders.stats import StatsEncoder

        return StatsEncoder(
            input_dim=dataset.input_dim,
            embedding_dim=self.cfg["embedding_dim"],
            hidden_dims=self.cfg["hidden_dims"],
        )