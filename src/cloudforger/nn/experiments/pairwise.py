# src/cloudforger/nn/experiments/pairwise.py

from __future__ import annotations

import torch

from cloudforger.nn.data import CorrelationFeatureDataset
from cloudforger.nn.experiments.base import Experiment, register


@register("pairwise")
class PairwiseExperiment(Experiment):
    file_key = "pairwise"
    subdir = "pairwise"

    def build_dataset(self, payload, labels):
        return CorrelationFeatureDataset(payload["features"], labels, dtype=torch.float32)

    def build_encoder(self, dataset):
        from cloudforger.nn.encoders.stats import StatsEncoder

        return StatsEncoder(
            input_dim=dataset.input_dim,
            embedding_dim=self.cfg["embedding_dim"],
            hidden_dims=self.cfg["hidden_dims"],
        )