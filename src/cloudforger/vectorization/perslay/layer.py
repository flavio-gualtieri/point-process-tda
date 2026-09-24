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

`norm="zscore"` standardizes the pooled vector before `fc`, the counterpart of the image arm's
pixel z-score: without it the pooled scale is whatever the softplus weight, the bandwidths and the
data's position relative to the centres make it. The statistics are FIXED -- estimated once on
training diagrams at initialization (`calibrate`) and stored as buffers, never updated by the
optimizer or by later batches -- so the map is the same at train and test time and the run is
reproducible from model.pt alone. They describe the initial centres, and go stale as the centres
move; a BatchNorm would track them, at the cost of batch-dependent training behaviour.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

OPS = ("sum", "max", "mean")
NORMS = ("none", "zscore")


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
        norm: str = "none",
    ):
        super().__init__()
        if op not in OPS:
            raise ValueError(f"op must be one of {OPS}, got {op!r}")
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
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
        self.norm = norm
        # buffers, not parameters: saved in model.pt, moved by .to(), invisible to the optimizer.
        # Identity until calibrate() runs, so an uncalibrated zscore layer is the plain one.
        self.register_buffer("pool_mean", torch.zeros(n_transforms))
        self.register_buffer("pool_std", torch.ones(n_transforms))
        self._moments: list[torch.Tensor] | None = None

    def pool(self, x: torch.Tensor) -> torch.Tensor:      # (B, 1, capacity, 3) -> (B, n_transforms)
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
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, 1, capacity, 3) -> (B, embedding_dim)
        pooled = self.pool(x)
        if self._moments is not None:                     # inside calibrate(): count, sum, sum of squares
            p = pooled.detach().double()
            self._moments[0] += p.shape[0]
            self._moments[1] += p.sum(dim=0)
            self._moments[2] += (p ** 2).sum(dim=0)
        if self.norm == "zscore":
            pooled = (pooled - self.pool_mean) / self.pool_std
        return self.fc(pooled)


@torch.no_grad()
def calibrate(model: nn.Module, run: Callable[[], object], eps: float = 1e-6) -> int:
    """Fix the pooled-vector statistics of every zscore PersLay in `model` from one pass of `run`,
    a callable that pushes TRAINING batches through the whole model (its outputs are discarded).

    Going through the model rather than the layer means each PersLay sees exactly what PHNet hands
    it -- one diagram per row, filtrations split out into the batch -- with no reshaping duplicated
    here. The model is put in eval mode for the pass and restored after, and the RNG state is
    restored too: iterating a DataLoader draws a base seed from the global generator even without
    shuffling, which would otherwise shift every later shuffle and dropout mask of the seed and
    break the pairing with an unnormalized run of the same seed. Returns how many layers were
    calibrated."""
    layers = [m for m in model.modules() if isinstance(m, PersLay) and m.norm == "zscore"]
    if not layers:
        return 0
    was_training = model.training
    model.eval()
    for layer in layers:
        layer._moments = [torch.zeros((), dtype=torch.float64, device=layer.pool_mean.device),
                          torch.zeros_like(layer.pool_mean, dtype=torch.float64),
                          torch.zeros_like(layer.pool_mean, dtype=torch.float64)]
    cuda = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    try:
        with torch.random.fork_rng(devices=cuda):
            run()
        for layer in layers:
            n, s, sq = layer._moments
            if n == 0:
                raise ValueError("calibrate: `run` pushed no rows through the model")
            mean = s / n
            std = torch.sqrt(torch.clamp(sq / n - mean ** 2, min=0.0))
            layer.pool_mean.copy_(mean.float())
            # eps floors a transform that no calibration diagram reaches (pooled ~0 everywhere):
            # it stays ~0 after the shift instead of being blown up by 1/0.
            layer.pool_std.copy_(torch.clamp(std, min=eps).float())
    finally:
        for layer in layers:
            layer._moments = None
        model.train(was_training)
    return len(layers)
