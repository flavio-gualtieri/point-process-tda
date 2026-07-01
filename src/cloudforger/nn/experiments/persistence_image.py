# src/cloudforger/nn/experiments/persistence_image.py

from __future__ import annotations

from cloudforger.nn.data import PersistenceImageDataset
from cloudforger.nn.experiments.base import Experiment, register


@register("pi")
class PersistenceImageExperiment(Experiment):
    file_key = "pi"

    @property
    def subdir(self) -> str:
        return f"pi_{self.hom_dim}"

    @property
    def extra_meta(self) -> dict:
        return {"hom_dim": self.hom_dim}

    def build_dataset(self, payload, labels):
        # PersistenceImageDataset unpacks "image_tensors"/"images", selects the
        # requested dim, and adds the channel axis expected by PIEncoder.
        return PersistenceImageDataset(payload, labels, homology_dims=[self.hom_dim])

    def build_encoder(self, dataset):
        from cloudforger.nn.encoders.persistence_image import PIEncoder

        return PIEncoder(in_channels=1, embedding_dim=self.cfg["embedding_dim"])