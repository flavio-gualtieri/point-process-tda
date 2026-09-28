"""The network inputs: fake diagrams and curves on disk -> training.data's datasets, and the encoder.

Small and synthetic, so it runs anywhere; what it checks is the wiring, not the science: that rips H0
becomes a 1-D image and DTM H0 a 2-D one, that imagers are fitted on train rows only, that the split
reaching the model is split_of's, and that the PersLay and curve arms see the same rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cloudforger.training import data as D

FAMILIES = ["poisson", "thomas"]
THETAS = {"train": [0, 1, 2, 3, 4, 5], "val": [7000, 7001], "test": [8000, 8001]}


def _write_family(root: Path, featurization: Path, family: str, tags: list[str], rng) -> int:
    rows = []
    for split_thetas in THETAS.values():
        for theta in split_thetas:
            for rep in (0, 1):
                n = int(rng.integers(100, 800))
                rows.append({"case_id": f"{family}-{theta:05d}-{rep}", "family": family, "theta": theta,
                             "rep": rep, "nbar": float(n), "delta": 0.5, "n": n,
                             "kappa": 20.0 + n / 100, "mu": 4.0 + rep, "sigma": 0.01 + n / 50000})
    manifest = pd.DataFrame(rows)
    (root / family).mkdir(parents=True, exist_ok=True)
    manifest.to_csv(root / family / "manifest.csv", index=False)

    for tag in tags:
        pairs = {0: [], 1: []}
        for dim in (0, 1):
            for _ in range(len(manifest)):
                m = int(rng.integers(3, 12))
                birth = np.zeros(m) if (dim == 0 and not tag.startswith("dtm")) else rng.uniform(0, 0.05, m)
                pairs[dim].append(np.column_stack([birth, birth + rng.uniform(0.01, 0.08, m)]))
        out = {"case_id": manifest.case_id.to_numpy(str)}
        for dim in (0, 1):
            out[f"h{dim}"] = np.concatenate(pairs[dim])
            out[f"h{dim}_offsets"] = np.concatenate([[0], np.cumsum([len(p) for p in pairs[dim]])])
        (featurization / family / tag).mkdir(parents=True, exist_ok=True)
        np.savez(featurization / family / tag / "diagrams.npz", **out)
    return len(manifest)


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    simulation, featurization = tmp_path / "simulation", tmp_path / "featurization"
    n_rows = sum(_write_family(simulation, featurization, f, ["rips", "dtm_k10"], rng) for f in FAMILIES)
    monkeypatch.setattr(D, "BANK", simulation)
    monkeypatch.setattr(D, "FEATURIZATION", featurization)
    return n_rows


def test_image_rank_follows_the_filtration(fake_data):
    rips = D.build(FAMILIES, ["rips"], [0, 1], resolution=8, verbose=False)
    assert rips.images[0].shape[1:] == (1, 8)        # H0 births all 0 -> 1-D
    assert rips.images[1].shape[1:] == (1, 8, 8)
    dtm = D.build(FAMILIES, ["dtm_k10"], [0], resolution=8, verbose=False)
    assert dtm.images[0].shape[1:] == (1, 8, 8)      # DTM H0 births are a density -> 2-D


def test_split_and_channel_stacking(fake_data):
    dataset = D.build(FAMILIES, ["rips", "dtm_k10"], [1], resolution=8, verbose=False)
    assert dataset.images[1].shape[1] == 2           # one channel per filtration
    counts = {s: len(dataset.index(s)) for s in ("train", "val", "test")}
    assert counts == {"train": 24, "val": 8, "test": 8}   # 2 families x 2 reps x thetas
    train_thetas = set(dataset.manifest.theta[dataset.index("train")])
    assert train_thetas == set(THETAS["train"])


def test_perslay_arm_pads_the_same_diagrams(fake_data):
    """Same diagrams, same split, same covariate -- only the vectorization differs, which is what
    makes the two PH arms comparable."""
    images = D.build(FAMILIES, ["dtm_k10"], [0, 1], resolution=8, verbose=False)
    padded = D.build_diagrams(FAMILIES, ["dtm_k10"], [0, 1], max_points=16, verbose=False)
    assert padded.images[0].shape[:2] == images.images[0].shape[:2]     # (patterns, filtrations)
    assert padded.images[0].shape[2:] == (padded.imagers[("dtm_k10", 0)].capacity, 3)
    assert np.array_equal(padded.covariates, images.covariates)
    assert [len(padded.index(s)) for s in ("train", "val", "test")] == [24, 8, 8]
    # A pattern's real points are its diagram's, and every remaining row is padding.
    pairs = D.load_pairs(FAMILIES[0], "dtm_k10", 0)[0]
    assert (padded.images[0][0, 0, :, 2] != 0).sum() == min(len(pairs), padded.images[0].shape[2])


def _blob(cx, cy, r=64, s=4.0):
    y, x = np.mgrid[0:r, 0:r]
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s)).astype(np.float32)


def test_encoder_distinguishes_where_mass_sits():
    """The same feature at two positions must not give the same embedding.

    A conv stack under a GLOBAL average pool is translation-invariant for every set of weights, so
    this would fail by construction if the final pool went back to 1 -- and in a persistence image
    the position is the measurement. Untrained weights are the point: the property is structural.
    """
    from cloudforger.training.model import ImageEncoder

    a = torch.from_numpy(_blob(20, 20))[None, None]
    b = torch.from_numpy(_blob(44, 44))[None, None]
    distances = []
    for seed in range(5):
        torch.manual_seed(seed)
        encoder = ImageEncoder(rank=2).eval()
        with torch.no_grad():
            ea, eb = encoder(a), encoder(b)
        distances.append((torch.norm(ea - eb) / torch.norm(ea)).item())
    assert np.mean(distances) > 0.02      # ~0.14 at pool_out=4; ~0.0002 under a global pool


def test_sqrt_transform_tames_the_dynamic_range(fake_data):
    """H1 images are mostly empty with a few very bright pixels; sqrt is what keeps the standardized
    input from being a floor plus a long tail."""
    linear = D.build(FAMILIES, ["dtm_k10"], [1], resolution=16, transform="none", verbose=False)
    root = D.build(FAMILIES, ["dtm_k10"], [1], resolution=16, transform="sqrt", verbose=False)
    # After standardizing, what matters is the skew: how far the bright tail reaches compared with
    # the empty floor. sqrt pulls the tail in and lets the floor use more of the range.
    def skew(images):
        return images.max() / abs(images.min())

    assert skew(root.images[1]) < skew(linear.images[1])


def _write_curves(tmp_path, rng, grids=("sqrtn_u2", "fixed")):
    """Minimal <data>/classical/<family>/<grid>/curves.npz for the classical arm."""
    root = tmp_path / "classical"
    for family in FAMILIES:
        manifest = pd.read_csv(D.BANK / family / "manifest.csv")
        for grid in grids:
            out = {"case_id": manifest.case_id.to_numpy(str), "axis": np.linspace(0, 2, 64)}
            for name in ("L", "F", "G", "J"):
                out[name] = rng.random((len(manifest), 64)).astype(np.float32)
            (root / family / grid).mkdir(parents=True, exist_ok=True)
            np.savez(root / family / grid / "curves.npz", **out)
    return root


def test_curves_on_one_grid_share_an_encoder(fake_data, tmp_path, monkeypatch):
    """Sharing a grid means index i is the same radius in every channel, so one encoder can compare
    them there; --curves-separate is the ablation that gives each its own."""
    monkeypatch.setattr(D, "CLASSICAL", _write_curves(tmp_path, np.random.default_rng(1)))
    curves = D.parse_curves("L,F,G,J", "sqrtn_u2")

    stacked = D.build_curves(FAMILIES, curves, verbose=False)
    assert len(stacked.images) == 1
    assert next(iter(stacked.images.values())).shape[1] == 4      # one key, four channels

    separate = D.build_curves(FAMILIES, curves, stack=False, verbose=False)
    assert sorted(separate.images) == ["F@sqrtn_u2", "G@sqrtn_u2", "J@sqrtn_u2", "L@sqrtn_u2"]
    assert all(v.shape[1] == 1 for v in separate.images.values())


def test_curves_on_different_grids_get_their_own_encoders(fake_data, tmp_path, monkeypatch):
    """F and G saturate long before r_max on the fixed axis, so they earn their own grid -- and then
    there is no shared index to convolve across, so stacking is refused rather than silently wrong."""
    monkeypatch.setattr(D, "CLASSICAL", _write_curves(tmp_path, np.random.default_rng(1)))
    curves = D.parse_curves("L@fixed,F,G,J", "sqrtn_u2")
    assert curves == [("L", "fixed"), ("F", "sqrtn_u2"), ("G", "sqrtn_u2"), ("J", "sqrtn_u2")]

    mixed = D.build_curves(FAMILIES, curves, verbose=False)      # separate, without being asked
    assert sorted(mixed.images) == ["F@sqrtn_u2", "G@sqrtn_u2", "J@sqrtn_u2", "L@fixed"]

    with pytest.raises(ValueError, match="different grids"):
        D.build_curves(FAMILIES, curves, stack=True, verbose=False)
