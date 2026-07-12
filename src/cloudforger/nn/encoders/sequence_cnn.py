# src/cloudforger/nn/encoders/sequence_cnn.py

import torch
import torch.nn as nn

from .base import Encoder


class SequenceCNNEncoder(Encoder):

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        n_filters: int = 64,
        kernel_size: int = 7,
        pool_size: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__(embedding_dim=embedding_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=kernel_size), nn.ReLU(), nn.MaxPool1d(pool_size), nn.Dropout1d(dropout),
            nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size), nn.ReLU(), nn.MaxPool1d(pool_size), nn.Dropout1d(dropout),
            nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size), nn.ReLU(), nn.Dropout1d(dropout),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, 1, input_dim)).flatten(1).shape[1]
        self.head = nn.Linear(flat_dim, embedding_dim)
        self.flat_dropout = nn.Dropout(dropout)

    @property
    def input_modality(self) -> str:
        return "sequence"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))
        h = h.flatten(1)
        h = self.flat_dropout(h)
        return self.head(h)
