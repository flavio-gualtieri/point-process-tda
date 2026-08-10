# src/cloudforger/experiments/persistence_image.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cloudforger.training.data import PersistenceImageDataset
from cloudforger.experiments.base import (
    Experiment,
    register,
    n_points_head_extra,
    persistence_entropy_head_extra,
)


@register("pi")
class PersistenceImageExperiment(Experiment):
    file_key = "pi"

    @property
    def _dims(self) -> tuple[int, ...]:
        return self.hom_dim if isinstance(self.hom_dim, tuple) else (self.hom_dim,)

    @property
    def subdir(self) -> str:
        return f"pi_{''.join(str(d) for d in self._dims)}"

    @property
    def extra_meta(self) -> dict:
        meta: dict[str, Any] = {"hom_dim": list(self._dims)}
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

        if not hasattr(self, "_image_norm"):
            self._image_norm = {}

        image_tensors: dict[int, np.ndarray] = {}
        for dim in self._dims:
            raw = np.asarray(payload["image_tensors"][dim], dtype=np.float64)
            if dim not in self._image_norm:
                mean = float(raw.mean())
                std = float(raw.std())
                self._image_norm[dim] = {"mean": mean, "std": std if std > 0 else 1.0}
            norm = self._image_norm[dim]
            image_tensors[dim] = ((raw - norm["mean"]) / norm["std"]).astype(np.float32)

        return PersistenceImageDataset(
            {"image_tensors": image_tensors}, labels, homology_dims=list(self._dims), return_dict=False
        )

    def extract_head_extra(self, payload, dataset_path: Path) -> np.ndarray:
        n_x = n_points_head_extra(self, payload, dataset_path)
        return persistence_entropy_head_extra(self, payload, self._dims, n_x)

    def build_encoder(self, dataset):
        from cloudforger.encoders.persistence_image import PIEncoder

        return PIEncoder(in_channels=len(self._dims), embedding_dim=self.cfg["embedding_dim"])