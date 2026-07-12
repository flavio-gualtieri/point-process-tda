# src/cloudforger/nn/experiments/persistence_image.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cloudforger.nn.data import PersistenceImageDataset
from cloudforger.nn.experiments.base import Experiment, register, n_points_head_extra


@register("pi")
class PersistenceImageExperiment(Experiment):
    file_key = "pi"

    @property
    def subdir(self) -> str:
        return f"pi_{self.hom_dim}"

    @property
    def extra_meta(self) -> dict:
        meta: dict[str, Any] = {"hom_dim": self.hom_dim}
        norm = getattr(self, "_n_points_norm", None)
        if norm is not None:
            meta["n_points_log_mean"] = norm["mean"]
            meta["n_points_log_std"] = norm["std"]
        return meta

    def build_dataset(self, payload, labels):
        # Persistence image pixels are raw Gaussian-kernel sums with a huge
        # dynamic range (spans several orders of magnitude, dominated by a
        # few outlier clouds) -- z-score globally before this reaches the
        # CNN, same fit-once/freeze-for-adversarial pattern used for the
        # Betti curve fix in betti_cnn.py.
        dim = self.hom_dim
        raw = np.asarray(payload["image_tensors"][dim], dtype=np.float64)

        if not hasattr(self, "_image_norm"):
            mean = float(raw.mean())
            std = float(raw.std())
            self._image_norm = {"mean": mean, "std": std if std > 0 else 1.0}

        norm = self._image_norm
        normalized = ((raw - norm["mean"]) / norm["std"]).astype(np.float32)

        # PersistenceImageDataset checks "image_tensors" before "images", so
        # a minimal dict with only this key guarantees it reads the
        # normalized array rather than the untouched per-cloud "images" list.
        return PersistenceImageDataset({"image_tensors": {dim: normalized}}, labels, homology_dims=[dim])

    def extract_head_extra(self, payload, dataset_path: Path) -> np.ndarray:
        return n_points_head_extra(self, payload, dataset_path)

    def build_encoder(self, dataset):
        from cloudforger.nn.encoders.persistence_image import PIEncoder

        return PIEncoder(in_channels=1, embedding_dim=self.cfg["embedding_dim"])