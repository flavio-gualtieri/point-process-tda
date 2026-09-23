"""One small CNN per homology dimension, concatenated with log n, then an MLP head.

Dimensions are never stacked as channels of one convolution: H0 and H1 images are calibrated on
their own boxes, so the same pixel means different things in each, and under rips/alpha they do not
even have the same rank. Each dimension gets its own encoder, and their embeddings are concatenated.

Two ways an input can carry several signals, and which applies is a property of the axis:

  separately (n_tags): several filtrations of the multi-k arm, each its own calibrated box, so one
  shared-weight encoder is applied to each in turn and the embeddings are concatenated.

  stacked (channels): the classical arm's L, F, G, J, which live on ONE shared r axis, so they go in
  as channels of the first convolution and the encoder can compare them at the same r -- something
  separate encoders can never do, since they only ever meet after pooling.

Only the encoder is about images: PHNet's `encoder` argument swaps in another one of the same
(B, in_channels, ...) -> (B, embedding_dim) shape, which is how the PersLay arm reuses the fusion
and the head unchanged (cloudforger.vectorization.perslay).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


POOL_OUT = 4


class ImageEncoder(nn.Module):
    """Conv stack -> coarse spatial pool -> embedding. 1-D or 2-D, following the image rank.

    The final pool keeps a POOL_OUT-wide map rather than collapsing to one number per channel.
    Convolution is translation-equivariant and a global average is translation-INVARIANT, for every
    possible set of weights, so a global pool would make the encoder unable to tell a feature at
    persistence 0.5 from the same feature at persistence 2.0 -- in a persistence image the position
    is the measurement, not a nuisance. Measured on two identical blobs at different positions, the
    relative distance between their embeddings is 0.0002 under a global pool and 0.14 at POOL_OUT=4.

    Dropout is channel-wise (Dropout1d/2d), not element-wise: neighbouring pixels of a feature map
    are strongly correlated, so dropping single activations regularizes little.
    """

    def __init__(
        self,
        rank: int,
        embedding_dim: int = 64,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.2,
        pool_out: int = POOL_OUT,
        in_channels: int = 1,
    ):
        super().__init__()
        conv, pool, adaptive, drop = ((nn.Conv1d, nn.AvgPool1d, nn.AdaptiveAvgPool1d, nn.Dropout1d) if rank == 1
                                      else (nn.Conv2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Dropout2d))
        layers: list[nn.Module] = []
        prev = in_channels
        for i, ch in enumerate(conv_channels):
            layers += [conv(prev, ch, kernel_size=3, padding=1), nn.ReLU()]
            if i < len(conv_channels) - 1:
                layers += [pool(2), drop(dropout)]
            prev = ch
        layers.append(adaptive(pool_out))
        self.conv = nn.Sequential(*layers)
        # ReLU after fc: without it this linear map would compose with the head's first Linear into
        # a single linear map, making the embedding a rank bottleneck and nothing more.
        self.fc = nn.Sequential(nn.Linear(prev * pool_out ** rank, embedding_dim), nn.ReLU())
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_channels, ...) -> (B, embedding_dim)
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
        ranks: list[int],          # rank of each input, in the dataset's key order
        n_tags: int,               # inputs of one key passed SEPARATELY through its encoder (multi-k)
        n_covariates: int,
        n_outputs: int,
        channels: list[int] | None = None,   # input channels per key, stacked INTO its first conv
        embedding_dim: int = 64,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.2,
        head_hidden_dims: tuple[int, ...] = (64, 32),
        head_dropout: float = 0.1,
        pool_out: int = POOL_OUT,
        encoder: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()
        self.n_tags = n_tags
        channels = channels or [1] * len(ranks)
        # `encoder` builds one per key from (rank, in_channels), for an arm whose inputs are not
        # images: the PersLay arm passes one, and everything below the encoder -- the pass per
        # filtration, the concatenation with log n, the head -- is the same either way.
        if encoder is None:
            def encoder(rank, c):
                return ImageEncoder(rank, embedding_dim, conv_channels, dropout, pool_out, c)
        self.encoders = nn.ModuleList([encoder(rank, c) for rank, c in zip(ranks, channels)])
        # asked of the encoders rather than assumed: a supplied `encoder` sets its own width.
        head_in = n_tags * sum(e.embedding_dim for e in self.encoders) + n_covariates
        self.head = mlp(head_in, n_outputs, head_hidden_dims, head_dropout)

    def forward(self, images: list[torch.Tensor], covariates: torch.Tensor) -> torch.Tensor:
        parts = []
        for encoder, x in zip(self.encoders, images):   # x: (B, n_tags * in_channels, ...)
            b = x.shape[0]
            flat = x.reshape(b * self.n_tags, x.shape[1] // self.n_tags, *x.shape[2:])
            parts.append(encoder(flat).reshape(b, -1))
        parts.append(covariates)
        return self.head(torch.cat(parts, dim=1))
