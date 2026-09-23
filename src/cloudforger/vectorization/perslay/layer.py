"""PersLay (Carriere et al., AISTATS 2020): a learned, permutation-invariant vectorization.

A diagram is a SET of points, so any vectorization of it must be permutation invariant. PersLay
writes every such map in one form,

    V(D) = op( { w(p) * phi(p) : p in D } )

with a point transformation phi, a scalar weight w, and a permutation-invariant pooling op, and
makes phi and w learnable instead of fixed. The persistence image is the frozen member of this
family: phi is a Gaussian on a fixed grid of centres, w is the linear weight w(p) = persistence,
and op is the sum. This arm keeps that shape and learns the three of them, so what changes against
the image arm is only whether the vectorization is calibrated or trained, not what it is applied to.

phi here is the Gaussian one, q = 1..Q learnable centres with their own bandwidths, since that is
the image's phi and keeps the two arms comparable; w is a small MLP of the point, the paper's
generic weight, and it multiplies the point's `mass` column so the n-scaling of the image arm
carries over unchanged (see pad.py). op defaults to the sum, the image's own pooling.

The input is what DiagramPadder produces, (B, 1, capacity, 3) once PHNet has split the filtrations
out into the batch: (birth, persistence) standardized, and a mass that is 0 exactly on padded rows.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

OPS = ("sum", "max", "mean")


class PersLay(nn.Module):
    """Padded diagrams -> embedding. Same (B, in_channels, ...) -> (B, embedding_dim) contract as
    ImageEncoder, so PHNet's head and fusion across filtrations are untouched.

    Centres are initialized spread over [-2, 2]^2 of the standardized plane, with a bandwidth of
    4 / sqrt(Q) -- about the spacing of Q centres over that square, the analogue of the image's
    sigma being measured in pixels rather than in absolute units.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_transforms: int = 64,
        op: str = "sum",
        weight_hidden: int = 16,
        dropout: float = 0.2,
        in_channels: int = 1,
    ):
        super().__init__()
        if op not in OPS:
            raise ValueError(f"op must be one of {OPS}, got {op!r}")
        if in_channels != 1:
            raise ValueError("PersLay takes one diagram at a time; stack filtrations as n_tags")
        self.op = op
        self.centres = nn.Parameter(torch.empty(n_transforms, 2).uniform_(-2.0, 2.0))
        self.log_sigma = nn.Parameter(torch.full((n_transforms,), math.log(4.0 / math.sqrt(n_transforms))))
        # softplus, not a bare linear: a weight is a mass, and a negative one would let a point
        # subtract another point's contribution, which no member of the PersLay family does.
        self.weight = nn.Sequential(nn.Linear(2, weight_hidden), nn.ReLU(),
                                    nn.Linear(weight_hidden, 1), nn.Softplus())
        # ReLU after fc for the reason ImageEncoder has one: otherwise this linear map and the
        # head's first Linear compose into a single linear map, and the embedding is a rank
        # bottleneck and nothing more.
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_transforms, embedding_dim), nn.ReLU())
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, capacity, 3) -> (B, embedding_dim)
        points, mass = x[:, 0, :, :2], x[:, 0, :, 2]
        # cdist rather than a broadcast difference: the (B, capacity, Q, 2) intermediate is the
        # largest tensor in the arm and is never needed, only its squared sum over the last axis.
        phi = torch.exp(-torch.cdist(points, self.centres) ** 2
                        / (2.0 * torch.exp(self.log_sigma) ** 2))       # (B, capacity, Q)
        weighted = phi * (self.weight(points).squeeze(-1) * mass).unsqueeze(-1)
        if self.op == "sum":
            pooled = weighted.sum(dim=1)
        elif self.op == "max":
            # phi >= 0 and w > 0, so a padded row contributes exactly 0 and cannot win a max
            # unless every real point is 0 there, in which case 0 is the right answer anyway.
            pooled = weighted.max(dim=1).values
        else:
            counts = (mass != 0).sum(dim=1, keepdim=True).clamp(min=1)
            pooled = weighted.sum(dim=1) / counts
        return self.fc(pooled)
