# src/cloudforger/models/single_modal.py
import torch
import torch.nn as nn
from ..encoders.base import Encoder

class SingleModalModel(nn.Module):
    """One encoder + one head."""

    def __init__(self, encoder: Encoder, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    @property
    def modality(self) -> str:
        return self.encoder.input_modality

    def forward(self, x: torch.Tensor, covariates: torch.Tensor | None = None) -> torch.Tensor:
        embedding = self.encoder(x)

        if covariates is not None:
            embedding = torch.cat([embedding, covariates], dim=1)

        return self.head(embedding)