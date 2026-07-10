# src/cloudforger/nn/experiments/betti_cnn.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from cloudforger.nn.data import BettiCurveDataset
from cloudforger.nn.experiments.base import Experiment, register


@register("betti_cnn")
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
    def subdir(self) -> str:
        return f"betti_cnn_{self.hom_dim}"

    @property
    def extra_meta(self) -> dict:
        meta: dict[str, Any] = {"hom_dim": self.hom_dim}
        norm = getattr(self, "_n_points_norm", None)
        if norm is not None:
            meta["n_points_log_mean"] = norm["mean"]
            meta["n_points_log_std"] = norm["std"]
        return meta

    def build_dataset(self, payload, labels):
        return BettiCurveDataset(payload, labels, homology_dims=[self.hom_dim])

    def build_encoder(self, dataset):
        from cloudforger.nn.encoders.sequence_cnn import SequenceCNNEncoder

        return SequenceCNNEncoder(
            input_dim=dataset.input_dim,
            embedding_dim=self.cfg["embedding_dim"],
        )

    def extract_head_extra(self, payload, dataset_path: Path) -> np.ndarray:
        # betti.pkl carries no n_points itself; join it from the sibling
        # clouds.pkl (or adversarial_clouds.pkl) by the shared 'seed' field,
        # the same join key models/evaluate.py's attach_betti_curves uses.
        clouds_path = dataset_path.with_name(dataset_path.name.replace("betti", "clouds"))
        with open(clouds_path, "rb") as f:
            clouds = pickle.load(f)
        n_points_by_seed = {c["seed"]: c["n_points"] for c in clouds}

        seeds = payload["seeds"]
        n_points = np.array([n_points_by_seed[int(s)] for s in seeds], dtype=np.float64)
        log_n = np.log(n_points)

        # Fit log+zscore on the first call (the train_test payload) and
        # reuse the frozen stats for the later adversarial-payload call --
        # the same fit-once/apply-frozen convention _normalize_labels_by_name
        # uses for the targets.
        if not hasattr(self, "_n_points_norm"):
            std = float(log_n.std())
            self._n_points_norm = {"mean": float(log_n.mean()), "std": std if std else 1.0}

        norm = self._n_points_norm
        return ((log_n - norm["mean"]) / norm["std"]).astype(np.float32)
