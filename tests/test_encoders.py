# tests/test_encoders.py
"""CoordConvPIEncoder regression test: it must handle a non-square (H, W)
raster, not just the resolution x resolution persistence images every
caller used to feed it. _coord_channels used to build one (resolution,
resolution) coordinate grid from x.shape[-1] alone -- silently correct for
persistence images (always square) but wrong for persistence landscapes
((K, G), K = layer count from the coverage rule, G = grid resolution, K !=
G in practice) -- crashed torch.cat with "Sizes of tensors must match
except in dimension 1. Expected size 16 but got size 128" the first time a
real vec_multik landscape SLURM run exercised it. See
encoders/coordconv_pi.py's docstring for the fix."""

from __future__ import annotations

import torch

from cloudforger.encoders.coordconv_pi import CoordConvPIEncoder


def test_coordconv_handles_non_square_raster():
    # Exactly the shape a landscape arm produces: in_channels = number of
    # homology dims, (K, G) = (landscape layers, grid resolution).
    encoder = CoordConvPIEncoder(in_channels=2, embedding_dim=64, conv_channels=(32, 64, 128), pool_type="avg")
    x = torch.randn(4, 2, 16, 128)  # (B, C, K, G) -- K != G
    out = encoder(x)
    assert out.shape == (4, 64)


def test_coordconv_still_handles_square_raster():
    # The original persistence-image case must keep working identically.
    encoder = CoordConvPIEncoder(in_channels=2, embedding_dim=64, conv_channels=(32, 64, 128), pool_type="max")
    x = torch.randn(4, 2, 64, 64)
    out = encoder(x)
    assert out.shape == (4, 64)


def test_coord_channels_square_case_matches_single_ramp():
    # The fix must be numerically identical to the old single-linspace
    # version when height == width (parity with pi_multik.py).
    coords = CoordConvPIEncoder._coord_channels(1, 32, 32, torch.device("cpu"), torch.float32)
    ramp = torch.linspace(-1, 1, 32)
    y, x = torch.meshgrid(ramp, ramp, indexing="ij")
    expected = torch.stack([x, y], dim=0).unsqueeze(0)
    assert torch.allclose(coords, expected)


def test_coord_channels_non_square_shape():
    coords = CoordConvPIEncoder._coord_channels(4, 16, 128, torch.device("cpu"), torch.float32)
    assert coords.shape == (4, 2, 16, 128)
