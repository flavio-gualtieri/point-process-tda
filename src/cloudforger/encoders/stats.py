# src/cloudforger/nn/encoders/stats.py

import torch
import torch.nn as nn

from .base import Encoder

class StatsEncoder(Encoder):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: tuple[int, ...] = (128, 128),
    ):
        super().__init__(embedding_dim=embedding_dim)
        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers += nn.Linear(prev, h), nn.ReLU()
            prev = h

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], embedding_dim)

    @property
    def input_modality(self) -> str:
        return "correlation_features"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x)
        embedding = self.head(h)
        return embedding