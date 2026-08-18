# src/cloudforger/encoders/flatten_mlp.py
"""The shared flatten-MLP encoder: the primary, encoder-confound-free path
for every vectorization (persistence images, landscapes, silhouettes,
persistence statistics alike -- see
experiments/pi_multik/vectorized_multik.py). Removes the choice of encoder
architecture as a variable between vectorizations: whatever headline
comparison is being made, this is the encoder every arm runs through, so
differences in test loss are attributable to the vectorization, not to one
vectorization getting a bespoke CNN and another a plain MLP."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder


class FlattenMLPEncoder(Encoder):
    """flatten -> Linear(512) -> ReLU -> Dropout(0.1) -> Linear(128) -> ReLU.

    Output width is fixed at 128 by construction (not read from any
    embedding_dim config) so fusion/head sizing stays identical across
    every vectorization x encoder_path combination this backs -- see
    module docstring."""

    def __init__(self, input_dim: int, embedding_dim: int = 128, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__(embedding_dim=embedding_dim)
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim), nn.ReLU(),
        )

    @property
    def input_modality(self) -> str:
        return "flatten_mlp"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
