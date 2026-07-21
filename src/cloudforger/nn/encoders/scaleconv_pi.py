# src/cloudforger/nn/encoders/scaleconv_pi.py
"""Fuses the K per-k CoordConvPIEncoder embeddings along the ordered k/m
axis with a small Conv1d block, instead of pi_multik's original order-blind
flat concat (still available as PIMultiK(use_fusion=False)). See
nn/experiments/pi_multik_scaleconv.py for the registered experiment that
wires this in."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Encoder


class ScaleConvFusion(Encoder):
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden: int = 128,
        out_dim: int = 128,
    ):
        super().__init__(embedding_dim=out_dim)
        self.net = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim), nn.ReLU(),
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
