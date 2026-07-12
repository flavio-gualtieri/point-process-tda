# src/cloudforger/nn/experiments/betti.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cloudforger.nn.data import BettiCurveDataset
from cloudforger.nn.experiments.base import Experiment, register, n_points_head_extra


@register("betti")
class BettiCurveExperiment(Experiment):
    file_key = "betti"

    @property
    def subdir(self) -> str:
        return f"betti_{self.hom_dim}"

    @property
    def extra_meta(self) -> dict:
        meta: dict[str, Any] = {"hom_dim": self.hom_dim}
        norm = getattr(self, "_n_points_norm", None)
        if norm is not None:
            meta["n_points_log_mean"] = norm["mean"]
            meta["n_points_log_std"] = norm["std"]
        return meta

    def build_dataset(self, payload, labels):
        # BettiCurveDataset accepts "betti_curves"/"curves"/"betti<d>_matrix"
        # payloads directly and flattens the requested dim to a vector.
        return BettiCurveDataset(payload, labels, homology_dims=[self.hom_dim])

    def extract_head_extra(self, payload, dataset_path: Path) -> np.ndarray:
        return n_points_head_extra(self, payload, dataset_path)

    def build_encoder(self, dataset):
        from cloudforger.nn.encoders.stats import StatsEncoder

        return StatsEncoder(
            input_dim=dataset.input_dim,
            embedding_dim=self.cfg["embedding_dim"],
            hidden_dims=self.cfg["hidden_dims"],
        )