# src/cloudforger/nn/encoders/scaleconv_pi.py

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dim of a (B, C, K) tensor -- normalizes
    each (sample, k-position) independently using only its own channel
    statistics. Swapped in for BatchNorm1d, which pools statistics across
    the batch AND the k-axis jointly into one running mean/var per channel;
    with a handful of scales spanning a wide range (e.g. topo_superset's
    m=0.01..0.90, fine-dense to coarse-sparse through the SAME shared-weight
    encoder), that one shared statistic mixes very heterogeneous per-scale
    distributions, which showed up as an erratic val-loss curve at n_k=7
    (tame at n_k=3, spiky at n_k=7 -- see the 7ch topo_superset run)."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, K)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ScaleConvFusion(Encoder):
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden: int = 128,
        out_dim: int = 128,
        kernel_size: int = 3,
    ):
        super().__init__(embedding_dim=out_dim)
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden, kernel_size=kernel_size, padding=padding),
            _ChannelLayerNorm(hidden), nn.ReLU(),
            nn.Conv1d(hidden, out_dim, kernel_size=kernel_size, padding=padding),
            _ChannelLayerNorm(out_dim), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = out_dim

    @property
    def input_modality(self) -> str:
        return "multiscale_pi_embedding_sequence"

    def forward(self, seq: torch.Tensor) -> torch.Tensor:  # (B, K, C)
        x = seq.transpose(1, 2)          # (B, C, K)
        x = self.net(x)                  # (B, out_dim, K)
        return self.pool(x).squeeze(-1)  # (B, out_dim)
