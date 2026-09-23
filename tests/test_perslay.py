"""PersLay / DiagramPadder: the properties that make this a fair second PH arm.

The image arm is tested in test_persistence_image.py; what has to hold here instead is that the
padding is invisible to the layer (a diagram is a set, and its length is an artefact of batching)
and that the same n-scaling the imager applies survives into the pooled vector.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cloudforger.vectorization.persistence_images import Scaling
from cloudforger.vectorization.perslay import DiagramPadder, PersLay, fit_padder

RNG = np.random.default_rng(0)


def _pairs(n, rng=RNG, lo=0.0, hi=0.5):
    b = rng.uniform(lo, hi, n)
    return np.column_stack([b, b + rng.uniform(lo, hi, n)])


def _padder(capacity=16, scaling=Scaling("none", False)):
    return DiagramPadder(capacity, mean=np.zeros(2), std=np.ones(2), scaling=scaling)


def test_columns_are_birth_persistence_mass():
    pairs = _pairs(5)
    padded = _padder().transform(pairs, 100)
    assert padded.shape == (16, 3)
    assert np.allclose(padded[:5, 0], pairs[:, 0])
    assert np.allclose(padded[:5, 1], pairs[:, 1] - pairs[:, 0])
    assert np.allclose(padded[:5, 2], 1.0)          # mass, with density scaling off
    assert not padded[5:].any()                     # padded rows are exactly zero, mask included


def test_empty_diagram_is_zeros():
    assert not _padder().transform(np.empty((0, 2)), 100).any()


def test_truncation_keeps_the_most_persistent_points():
    pairs = _pairs(40)
    kept = _padder(capacity=10).transform(pairs, 100)[:, 1]
    persistence = np.sort(pairs[:, 1] - pairs[:, 0])[::-1][:10]
    assert np.allclose(np.sort(kept)[::-1], persistence)


def test_scaling_removes_the_n_dependence():
    """The imager's contract, one stage later: two diagrams identical up to the 1/sqrt(n) spacing
    must reach the layer as the same point set, carrying the same total mass."""
    pairs = _pairs(30)
    padder = DiagramPadder(64, mean=np.zeros(2), std=np.ones(2))     # sqrt_n coords, density mass
    a = padder.transform(pairs / np.sqrt(100), 100)
    b = padder.transform(pairs / np.sqrt(800), 800)
    assert np.allclose(a[:, :2], b[:, :2])
    assert np.allclose(a[:, 2] * 100, b[:, 2] * 800)


def test_fit_padder_sizes_the_capacity_and_reports_truncation():
    pairs = [_pairs(m) for m in [5] * 99 + [400]]
    n = np.full(len(pairs), 100)
    padder, truncated = fit_padder(pairs, n, coverage=0.9, max_points=1024, scaling=Scaling("none", False))
    assert padder.capacity == 5                       # 90% of the diagrams fit whole
    assert truncated == pytest.approx(0.01)
    capped, _ = fit_padder(pairs, n, coverage=1.0, max_points=32, scaling=Scaling("none", False))
    assert capped.capacity == 32                      # the cap wins over the coverage quantile
    assert np.allclose(padder.mean, np.concatenate([p for p in pairs])[:, 0].mean(), atol=0.2)


def _batch(diagrams, capacity=16):
    padder = _padder(capacity)
    return torch.from_numpy(np.stack([padder.transform(d, 100) for d in diagrams]))[:, None]


def test_padding_and_order_are_invisible_to_the_layer():
    """A diagram is a SET: the layer must not see how its points were ordered, nor how many zero
    rows batching added after them."""
    torch.manual_seed(0)
    layer = PersLay(embedding_dim=8, n_transforms=16).eval()
    pairs = _pairs(9)
    with torch.no_grad():
        plain = layer(_batch([pairs], capacity=16))
        shuffled = layer(_batch([RNG.permutation(pairs)], capacity=16))
        longer = layer(_batch([pairs], capacity=64))
    assert torch.allclose(plain, shuffled, atol=1e-5)
    assert torch.allclose(plain, longer, atol=1e-5)


@pytest.mark.parametrize("op", ["sum", "max", "mean"])
def test_every_op_is_permutation_invariant_and_shaped(op):
    torch.manual_seed(0)
    layer = PersLay(embedding_dim=8, n_transforms=16, op=op).eval()
    batch = _batch([_pairs(7), _pairs(3)])
    with torch.no_grad():
        out = layer(batch)
        flipped = layer(batch.flip(dims=(2,)))
    assert out.shape == (2, 8)
    assert torch.allclose(out, flipped, atol=1e-5)


def test_an_empty_diagram_contributes_nothing():
    """Sum pooling over no points is 0, so the embedding is whatever the head makes of a zero
    vector -- never a Gaussian centred on the padding at the origin."""
    torch.manual_seed(0)
    layer = PersLay(embedding_dim=8, n_transforms=16).eval()
    with torch.no_grad():
        empty = layer(_batch([np.empty((0, 2))]))
        zero = layer.fc(torch.zeros(1, 16))
    assert torch.allclose(empty, zero, atol=1e-6)


def test_the_weight_is_never_negative():
    """PersLay's w is a mass: a negative one would let a point cancel another's contribution,
    which no member of the family (the image's linear weight included) does."""
    torch.manual_seed(0)
    layer = PersLay(embedding_dim=8, n_transforms=4)
    points = torch.from_numpy(RNG.normal(size=(50, 2)).astype(np.float32))
    assert (layer.weight(points) > 0).all()


def test_gradients_reach_the_learned_vectorization():
    """The point of the arm: phi's centres and bandwidths and w's weights are trained, not fitted
    on the training diagrams and then frozen."""
    torch.manual_seed(0)
    layer = PersLay(embedding_dim=8, n_transforms=16)
    layer(_batch([_pairs(7), _pairs(4)])).sum().backward()
    assert layer.centres.grad.abs().sum() > 0
    assert layer.log_sigma.grad.abs().sum() > 0
    assert all(p.grad.abs().sum() > 0 for p in layer.weight.parameters())


def test_one_diagram_at_a_time():
    with pytest.raises(ValueError, match="one diagram at a time"):
        PersLay(in_channels=2)
    with pytest.raises(ValueError, match="op must be"):
        PersLay(op="median")
