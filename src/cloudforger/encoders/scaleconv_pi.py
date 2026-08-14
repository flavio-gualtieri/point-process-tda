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


class ConvFusion(Encoder):
    """Fuses a (B, K, embedding_dim) per-k embedding sequence into a single
    vector via two Conv1d layers over the ordered k axis (so adjacent scales
    can mix/reinforce each other -- see _ChannelLayerNorm's docstring for
    why LayerNorm over BatchNorm1d), then either collapses the k axis or
    keeps it:

    pool="avg": AdaptiveAvgPool1d(1) over k -> (B, out_dim). Scale-count
    invariant: output width doesn't depend on n_k, and the fused vector is
    a straight average over scale positions.

    pool="flatten": keeps every k position, concatenated -> (B, out_dim *
    K). Preserves per-scale identity (position i's slice always corresponds
    to k_values[i]) at the cost of an output width that grows with n_k, and
    of losing invariance to how many/which scales are fused.

    Absorbs what were previously two near-duplicate encoders (scaleconv_pi's
    old avg-pool-only ScaleConvFusion and towerconv_pi's flatten-only
    TowerConv) into one class with a pool switch, so the two fusion
    strategies stay a config choice rather than a fork."""

    def __init__(
        self,
        n_k: int,
        embedding_dim: int = 64,
        hidden: int = 128,
        out_dim: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.0,
        pool: str = "avg",
    ):
        if pool not in ("avg", "flatten"):
            raise ValueError(f"ConvFusion: pool must be 'avg' or 'flatten', got {pool!r}.")
        super().__init__(embedding_dim=(out_dim if pool == "avg" else out_dim * n_k))
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden, kernel_size=kernel_size, padding=padding),
            _ChannelLayerNorm(hidden), nn.ReLU(), nn.Dropout1d(dropout),
            nn.Conv1d(hidden, out_dim, kernel_size=kernel_size, padding=padding),
            _ChannelLayerNorm(out_dim), nn.ReLU(),
        )
        self.pool_type = pool
        self.pool = nn.AdaptiveAvgPool1d(1) if pool == "avg" else None
        self.out_dim = self.embedding_dim

    @property
    def input_modality(self) -> str:
        return "multiscale_pi_embedding_sequence" if self.pool_type == "avg" else "multiscale_pi_embedding_sequence_flat"

    def forward(self, seq: torch.Tensor) -> torch.Tensor:  # (B, K, C)
        x = seq.transpose(1, 2)          # (B, C, K)
        x = self.net(x)                  # (B, out_dim, K)
        if self.pool_type == "avg":
            return self.pool(x).squeeze(-1)  # (B, out_dim)
        return x.flatten(1)                  # (B, out_dim * K)
