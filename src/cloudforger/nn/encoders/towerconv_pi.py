# src/cloudforger/nn/encoders/towerconv_pi.py

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder


class _ChannelLayerNorm(nn.Module):

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, K)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TowerConv(Encoder):
    def __init__(
        self,
        n_k: int,
        embedding_dim: int = 64,
        hidden: int = 128,
        out_dim: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__(embedding_dim=(out_dim * n_k))
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden, kernel_size=kernel_size, padding=padding),
            _ChannelLayerNorm(hidden), nn.ReLU(), nn.Dropout1d(dropout),
            nn.Conv1d(hidden, out_dim, kernel_size=kernel_size, padding=padding),
            _ChannelLayerNorm(out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim * n_k

    @property
    def input_modality(self) -> str:
        return "multiscale_pi_embedding_sequence_flat"

    def forward(self, seq: torch.Tensor) -> torch.Tensor:  # (B, K, C)
        x = seq.transpose(1, 2)          # (B, C, K)
        x = self.net(x)                  # (B, out_dim, K)
        return (x).flatten(1)  # (B, out_dim * K)
