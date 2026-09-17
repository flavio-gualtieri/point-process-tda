"""One small CNN per homology dimension, concatenated with log n, then an MLP head.

Dimensions are never stacked as channels of one convolution: H0 and H1 images are calibrated on
their own boxes, so the same pixel means different things in each, and under rips/alpha they do not
even have the same rank. Each dimension gets its own encoder, and their embeddings are concatenated.

Several filtrations (the multi-k arm) are the encoder's channel axis instead: one shared-weight
encoder is applied to each, and the embeddings are concatenated, so an extra filtration adds an
embedding rather than a copy of the network.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """Conv stack -> global average pool -> embedding. 1-D or 2-D, following the image rank."""

    def __init__(
        self,
        rank: int,
        embedding_dim: int = 64,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        conv, pool, gap = ((nn.Conv1d, nn.AvgPool1d, nn.AdaptiveAvgPool1d) if rank == 1
                           else (nn.Conv2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))
        layers: list[nn.Module] = []
        prev = 1
        for i, ch in enumerate(conv_channels):
            layers += [conv(prev, ch, kernel_size=3, padding=1), nn.ReLU()]
            if i < len(conv_channels) - 1:
                layers += [pool(2), nn.Dropout(dropout)]
            prev = ch
        layers.append(gap(1))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(prev, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, ...) -> (B, embedding_dim)
        return self.fc(self.conv(x).flatten(1))


def mlp(in_dim: int, out_dim: int, hidden_dims: tuple[int, ...] = (64, 32), dropout: float = 0.1):
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class PHNet(nn.Module):
    def __init__(
        self,
        ranks: list[int],          # image rank per homology dimension, in the dataset's dim order
        n_tags: int,               # filtrations sharing one encoder (the multi-k arm)
        n_covariates: int,
        n_outputs: int,
        embedding_dim: int = 64,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.2,
        head_hidden_dims: tuple[int, ...] = (64, 32),
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tags = n_tags
        self.encoders = nn.ModuleList(
            [ImageEncoder(rank, embedding_dim, conv_channels, dropout) for rank in ranks]
        )
        head_in = len(ranks) * n_tags * embedding_dim + n_covariates
        self.head = mlp(head_in, n_outputs, head_hidden_dims, head_dropout)

    def forward(self, images: list[torch.Tensor], covariates: torch.Tensor) -> torch.Tensor:
        parts = []
        for encoder, x in zip(self.encoders, images):   # x: (B, n_tags, ...)
            b = x.shape[0]
            flat = x.reshape(b * self.n_tags, 1, *x.shape[2:])   # tags share the encoder
            parts.append(encoder(flat).reshape(b, -1))
        parts.append(covariates)
        return self.head(torch.cat(parts, dim=1))
