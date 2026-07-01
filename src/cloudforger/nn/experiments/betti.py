# src/cloudforger/nn/experiments/betti.py

from __future__ import annotations

from cloudforger.nn.data import BettiCurveDataset
from cloudforger.nn.experiments.base import Experiment, register


@register("betti")
class BettiCurveExperiment(Experiment):
    file_key = "betti"

    @property
    def subdir(self) -> str:
        return f"betti_{self.hom_dim}"

    @property
    def extra_meta(self) -> dict:
        return {"hom_dim": self.hom_dim}

    def build_dataset(self, payload, labels):
        # BettiCurveDataset accepts "betti_curves"/"curves"/"betti<d>_matrix"
        # payloads directly and flattens the requested dim to a vector.
        return BettiCurveDataset(payload, labels, homology_dims=[self.hom_dim])

    def build_encoder(self, dataset):
        from cloudforger.nn.encoders.stats import StatsEncoder

        return StatsEncoder(
            input_dim=dataset.input_dim,
            embedding_dim=self.cfg["embedding_dim"],
            hidden_dims=self.cfg["hidden_dims"],
        )