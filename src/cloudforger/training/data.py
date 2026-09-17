"""Diagrams on disk -> persistence images, targets and splits, in memory.

    data/simulation/<family>/manifest.csv          one row per pattern (case_id, theta, nbar, n, ...)
    data/featurization/<family>/<tag>/diagrams.npz  its diagrams, h<d> + h<d>_offsets in manifest order

Rows are ordered by (family, manifest order), and every array here keeps that order, so `case_id`
identifies a row all the way to predictions.npz.

Imagers are fitted on TRAIN rows only, once per (tag, dim), and applied frozen to every row; the
per-channel z-score likewise. The birth axis exists only where the filtration gives H0 a non-trivial
birth (DTM), so rips/alpha H0 images are 1-D -- see birth_axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch.utils.data

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
    images: dict[int, np.ndarray]           # dim -> (N, n_tags, resolution[, resolution]) float32
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


def build(
    families: list[str],
    tags: list[str],
    dims: list[int],
    resolution: int = 64,
    sigma_pixels: float = 1.0,
    coverage: float = 0.99,
    scaling: Scaling = Scaling(),
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
        stacked = np.stack(channels, axis=1)
        mean, std = _zscore(stacked, train)
        shape = (1, len(tags)) + (1,) * (stacked.ndim - 2)
        stacked -= mean.reshape(shape)      # in place: the tensor is the bulk of a run's memory
        stacked /= std.reshape(shape)
        images[dim], norm[dim] = stacked, (mean, std)

    covariates = np.log(n.astype(np.float32)).reshape(-1, 1)
    covariates = (covariates - covariates[train].mean()) / covariates[train].std()
    return Dataset(manifest, images, covariates, imagers, norm)


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
