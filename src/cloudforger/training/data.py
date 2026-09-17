"""Diagrams on disk -> persistence images, targets and splits, in memory.

    data/simulation/<family>/manifest.csv           one row per pattern (case_id, theta, nbar, n, ...)
    data/featurization/<family>/<tag>/diagrams.npz  its diagrams, h<d> + h<d>_offsets in manifest order
    data/classical/<family>/<grid>/curves.npz       its L/F/G/J curves (the classical arm)

Two feature sources, one contract: build() rasterizes diagrams, build_curves() reads summary
functions, and both return a Dataset whose `images` maps a channel key to a (N, ...) array with one
encoder per key. Everything after this module is shared, so the arms are comparable by construction.

Rows are ordered by (family, manifest order), and every array here keeps that order, so `case_id`
identifies a row all the way to predictions.npz.

Imagers are fitted on TRAIN rows only, once per (tag, dim), and applied frozen to every row; the
per-channel z-score likewise -- one (mean, std) per filtration channel, pooled over train patterns
and pixels, so the picture is only shifted and rescaled, never reshaped per pattern or per pixel.

Images pass through sqrt before that z-score: a linearly weighted persistence image of H1 is ~93%
empty with a handful of very bright pixels (standardized range [-0.2, +50] on real Thomas dtm_k10
diagrams), which no single affine map can tame. sqrt is variance-stabilizing for that kind of
mass-like quantity, keeps zero at zero, and leaves the imager itself exactly Adams et al. The birth axis exists only where the filtration gives H0 a non-trivial
birth (DTM), so rips/alpha H0 images are 1-D -- see birth_axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch.utils.data

from ..classical.curves import DATA as CLASSICAL
from ..featurization.sweep import DATA as FEATURIZATION, SIMULATION
from ..simulation.split import split_of
from ..vectorization.persistence_images import PersistenceImager, Scaling, fit_imager

FAMILIES = ("poisson", "thomas", "nested", "matern2", "lgcp")


def birth_axis(tag: str, dim: int) -> bool:
    """H1 always has one; H0 only under DTM, where births are 2 f(x) rather than 0."""
    return dim > 0 or tag.startswith("dtm")


@dataclass
class Dataset:
    manifest: pd.DataFrame                  # one row per pattern, with a `split` column
    images: dict                            # channel key -> (N, n_tags, ...) float32; one encoder each
    covariates: np.ndarray                  # (N, 1) float32: log n
    imagers: dict[tuple[str, int], PersistenceImager]
    norm: dict[int, tuple[np.ndarray, np.ndarray]]

    def index(self, split: str) -> np.ndarray:
        return np.flatnonzero((self.manifest["split"] == split).to_numpy())


def load_manifest(families: list[str]) -> pd.DataFrame:
    frames = []
    for family in families:
        m = pd.read_csv(SIMULATION / family / "manifest.csv")
        m["split"] = split_of(m["theta"].to_numpy())
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def load_pairs(family: str, tag: str, dim: int) -> list[np.ndarray]:
    """Per-pattern (m, 2) finite pairs, in manifest order."""
    z = np.load(FEATURIZATION / family / tag / "diagrams.npz")
    flat, offsets = z[f"h{dim}"], z[f"h{dim}_offsets"]
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def _channel(
    pairs: list[np.ndarray], n: np.ndarray, train: np.ndarray, tag: str, dim: int,
    resolution: int, sigma_pixels: float, coverage: float, scaling: Scaling,
) -> tuple[np.ndarray, PersistenceImager]:
    imager = fit_imager(
        [pairs[i] for i in train], n[train], birth_axis=birth_axis(tag, dim),
        resolution=resolution, sigma_pixels=sigma_pixels, coverage=coverage, scaling=scaling,
    )
    out = np.empty((len(pairs), *imager.shape), dtype=np.float32)
    for i, p in enumerate(pairs):
        out[i] = imager.transform(p, int(n[i]))
    return out, imager


def _zscore(images: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One (mean, std) per tag channel, over train pixels: H0 and H1 pixel scales differ by ~10x
    and nothing in the network rescales its input."""
    axes = (0, *range(2, images.ndim))
    block = images[train]
    return (block.mean(axis=axes, dtype=np.float32),
            np.maximum(block.std(axis=axes, dtype=np.float32), 1e-12))


TRANSFORMS = {"sqrt": np.sqrt, "none": lambda x: x}


def build(
    families: list[str],
    tags: list[str],
    dims: list[int],
    resolution: int = 64,
    sigma_pixels: float = 1.0,
    coverage: float = 0.99,
    scaling: Scaling = Scaling(),
    transform: str = "sqrt",
    verbose: bool = True,
) -> Dataset:
    manifest = load_manifest(families)
    n = manifest["n"].to_numpy()
    train = manifest.index[manifest["split"] == "train"].to_numpy()

    images, imagers, norm = {}, {}, {}
    for dim in dims:
        channels = []
        for tag in tags:
            pairs = [p for family in families for p in load_pairs(family, tag, dim)]
            if len(pairs) != len(manifest):
                raise ValueError(f"{tag} H{dim}: {len(pairs)} diagrams for {len(manifest)} patterns")
            channel, imager = _channel(pairs, n, train, tag, dim, resolution, sigma_pixels,
                                       coverage, scaling)
            imagers[(tag, dim)] = imager
            channels.append(channel)
            if verbose:
                print(f"  [image] {tag} H{dim}: {channel.shape[1:]} {imager.params}", flush=True)
        stacked = TRANSFORMS[transform](np.stack(channels, axis=1))
        mean, std = _zscore(stacked, train)
        shape = (1, len(tags)) + (1,) * (stacked.ndim - 2)
        stacked -= mean.reshape(shape)      # in place: the tensor is the bulk of a run's memory
        stacked /= std.reshape(shape)
        images[dim], norm[dim] = stacked, (mean, std)

    return Dataset(manifest, images, _covariates(n, train), imagers, norm)


def _covariates(n: np.ndarray, train: np.ndarray) -> np.ndarray:
    covariates = np.log(n.astype(np.float32)).reshape(-1, 1)
    return (covariates - covariates[train].mean()) / covariates[train].std()


def load_curves(family: str, grid: str, name: str) -> np.ndarray:
    """One family's L/F/G/J curves on one grid, (P, 512) float32 in manifest order."""
    return np.load(CLASSICAL / family / grid / "curves.npz")[name]


def parse_curves(spec: str, default_grid: str) -> list[tuple[str, str]]:
    """'L@fixed,F,G,J' -> [(L, fixed), (F, default), (G, default), (J, default)].

    A curve names its own grid because the functions saturate at different places: F and G are
    distance CDFs that reach 1 near the mean nearest-neighbour distance, so on the fixed r axis
    (r_max = 0.25) most of their 512 samples sit on a flat tail, while L is only comparable to the
    literature on that axis. Per-function grids spend each curve's resolution where it varies.
    """
    out = []
    for item in spec.split(","):
        name, _, grid = item.partition("@")
        out.append((name, grid or default_grid))
    return out


def build_curves(families: list[str], curves: list[tuple[str, str]], stack: bool | None = None,
                 verbose: bool = True) -> Dataset:
    """The classical arm: (curve, grid) pairs as 1-D channels.

    stack puts the curves in as channels of ONE encoder, which is only meaningful when they share a
    grid: then index i means the same radius in every channel and a convolution can compare them
    there. Curves on different grids have nothing to compare at a given index, so they get one
    encoder each and meet only as concatenated embeddings -- the same arrangement H0 and H1 are in
    for the same reason. Default: stack when the grid is shared, separate when it is not.

    Same contract as build() otherwise -- manifest order, statistics fitted on train rows only,
    log n as the covariate -- so everything downstream is shared and the arms stay comparable. No
    sqrt here: these are bounded functions (F, G in [0, 1]), not a sparse measure with a heavy tail.
    """
    grids = {grid for _, grid in curves}
    if stack is None:
        stack = len(grids) == 1
    elif stack and len(grids) > 1:
        raise ValueError(f"curves on different grids ({sorted(grids)}) cannot share an encoder")

    manifest = load_manifest(families)
    train = manifest.index[manifest["split"] == "train"].to_numpy()

    blocks = {}
    for name, grid in curves:
        block = np.concatenate([load_curves(f, grid, name) for f in families])[:, None, :]
        if len(block) != len(manifest):
            raise ValueError(f"{name} on {grid}: {len(block)} curves for {len(manifest)} patterns")
        blocks[f"{name}@{grid}"] = block

    if stack:
        blocks = {"+".join(blocks): np.concatenate(list(blocks.values()), axis=1)}

    images, norm = {}, {}
    for key, block in blocks.items():
        mean, std = _zscore(block, train)          # one (mean, std) per curve, whichever layout
        shape = (1, block.shape[1], 1)
        images[key] = (block - mean.reshape(shape)) / std.reshape(shape)
        norm[key] = (mean, std)
        if verbose:
            print(f"  [curve] {key}: {images[key].shape[1:]}", flush=True)

    return Dataset(manifest, images, _covariates(manifest["n"].to_numpy(), train), {}, norm)


def targets(manifest: pd.DataFrame, columns: list[str], train: np.ndarray) -> tuple[np.ndarray, dict]:
    """Regression targets, log-transformed where strictly positive, then z-scored on train rows.

    The transform is returned so predictions can be put back into parameter units.
    """
    values = manifest[columns].to_numpy(float)
    logged = (values[train] > 0).all(axis=0)
    values = np.where(logged, np.log(np.where(values > 0, values, np.nan)), values)
    mean, std = values[train].mean(axis=0), np.maximum(values[train].std(axis=0), 1e-12)
    norm = {"columns": columns, "log": logged.tolist(), "mean": mean.tolist(), "std": std.tolist()}
    return ((values - mean) / std).astype(np.float32), norm


def invert_targets(standardized: np.ndarray, norm: dict) -> np.ndarray:
    values = standardized * np.asarray(norm["std"]) + np.asarray(norm["mean"])
    return np.where(norm["log"], np.exp(values), values)


# The family's own parameters (Family.model's keys), which is what the manifest stores. nbar stands
# in for LGCP's mu_log (= log nbar - sigma2/2, and the only target that can be negative) and is
# Poisson's only parameter.
TARGETS = {
    "poisson": ["nbar"],
    "thomas": ["kappa", "mu", "sigma"],
    "nested": ["kappa", "mu1", "mu2", "sigma1", "sigma2"],
    "matern2": ["R", "lam_p"],
    "lgcp": ["nbar", "sigma2", "s"],
}


class Rows(torch.utils.data.Dataset):
    """A split's rows as (images per dim, covariates, target) tuples, indexing the full arrays
    rather than slicing them, so a split costs no extra memory."""

    def __init__(self, data: Dataset, targets: np.ndarray, index: np.ndarray):
        self.images = [torch.from_numpy(data.images[dim]) for dim in sorted(data.images)]
        self.covariates = torch.from_numpy(data.covariates)
        self.targets = torch.from_numpy(np.array(targets))   # copy: torch rejects read-only arrays
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        j = self.index[i]
        return [x[j] for x in self.images], self.covariates[j], self.targets[j]
