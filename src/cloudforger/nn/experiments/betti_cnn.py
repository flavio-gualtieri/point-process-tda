# src/cloudforger/nn/experiments/betti_cnn.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cloudforger.nn.data import BettiCurveDataset
from cloudforger.nn.experiments.base import (
    Experiment,
    register,
    n_points_head_extra,
    persistence_entropy_head_extra,
)


@register("betti_cnn", "betti_cnn_weighted")
class BettiCurveCNNExperiment(Experiment):
    """Same Betti-curve data as BettiCurveExperiment (betti_0/betti_1), but
    encoded with SequenceCNNEncoder (Conv1D, matching the architecture
    Vihrs (2022) uses for the L(r)-r curve) instead of StatsEncoder (a plain
    dense MLP), and additionally feeding the point count n(x) into the head
    alongside the encoder's embedding -- also matching the paper, which uses
    n(x) as a second input because the curve alone can't recover
    intensity-related parameters. Holding the data/head/training procedure
    otherwise fixed isolates whether the encoder/n(x) or the feature itself
    explains the gap to scripts/runners/params/run_new_feature.py."""

    file_key = "betti"

    @property
    def _dims(self) -> tuple[int, ...]:
        return self.hom_dim if isinstance(self.hom_dim, tuple) else (self.hom_dim,)

    @property
    def subdir(self) -> str:
        # cfg["method"] is the full method string (e.g. "betti_cnn_0" or
        # "betti_cnn_weighted_0") -- using it directly, rather than
        # rebuilding "betti_cnn_<dims>" from self._dims, keeps
        # betti_cnn_weighted_* in its own results dir instead of colliding
        # with the unweighted betti_cnn_* run for the same dims.
        return self.cfg["method"]

    @property
    def extra_meta(self) -> dict:
        meta: dict[str, Any] = {"hom_dim": list(self._dims)}
        norm = getattr(self, "_n_points_norm", None)
        if norm is not None:
            meta["n_points_log_mean"] = norm["mean"]
            meta["n_points_log_std"] = norm["std"]
        return meta
    
    def build_dataset(self, payload, labels):
        # Global-scalar z-score (one mean/std for the whole curve, not
        # per-bin) -- matches Vihrs (2022) section 2.2 item 2d exactly:
        # "for {L_i} the mean and standard deviation were calculated both
        # over all n_train simulations and over all m values for r, meaning
        # that all values of {L_i} were scaled by the same amount." Per-bin
        # normalization would give every position its own affine transform,
        # which breaks the translation-equivariance the Conv1D relies on.

        if not hasattr(self, "_curve_norm"):
            self._curve_norm = {}

        normalized: dict[str, np.ndarray] = {}
        for dim in self._dims:
            matrix_key = f"betti{dim}_matrix"
            raw = np.asarray(payload[matrix_key], dtype=np.float64)

            if dim not in self._curve_norm:
                mean = float(raw.mean())
                std = float(raw.std())
                self._curve_norm[dim] = {"mean": mean, "std": std if std > 0 else 1.0}

            norm = self._curve_norm[dim]
            normalized[matrix_key] = ((raw - norm["mean"]) / norm["std"]).astype(np.float32)

        return BettiCurveDataset(normalized, labels, homology_dims=list(self._dims))

    def build_encoder(self, dataset):
        from cloudforger.nn.encoders.sequence_cnn import SequenceCNNEncoder

        return SequenceCNNEncoder(
            input_dim=dataset.input_dim,
            embedding_dim=self.cfg["embedding_dim"],
            pool_size=5,
        )

    def extract_head_extra(self, payload, dataset_path: Path) -> np.ndarray:
        n_x = n_points_head_extra(self, payload, dataset_path)
        return persistence_entropy_head_extra(self, payload, self._dims, n_x)