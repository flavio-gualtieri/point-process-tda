# src/cloudforger/nn/experiments/ph_combined.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cloudforger.nn.data import BettiCurveDataset, PersistenceImageDataset
from cloudforger.nn.encoders.base import Encoder
from cloudforger.nn.experiments.base import (
    Experiment,
    register,
    n_points_head_extra,
    persistence_entropy_head_extra,
    _load_pickle,
)


class _PiBettiFusionEncoder(Encoder):
    """Two independent encoders (PIEncoder over the pi_01-style 2-channel
    persistence image, SequenceCNNEncoder over the betti_cnn_01-style
    concatenated Betti curve), late-fused by concatenating their embeddings
    -- same "concatenate before the head" fusion MultiModalModel uses, kept
    as a plain Encoder here instead so it drops straight into the
    SingleModalModel + shared Experiment.run() orchestration every other
    method already goes through."""

    def __init__(self, pi_encoder: Encoder, betti_encoder: Encoder):
        super().__init__(embedding_dim=pi_encoder.embedding_dim + betti_encoder.embedding_dim)
        self.pi_encoder = pi_encoder
        self.betti_encoder = betti_encoder

    @property
    def input_modality(self) -> str:
        return "pi+betti_curve"

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        pi_embedding = self.pi_encoder(x["pi"])
        betti_embedding = self.betti_encoder(x["betti"])
        return torch.cat([pi_embedding, betti_embedding], dim=-1)


class _FusedPiBettiDataset(Dataset):
    """Zips a PersistenceImageDataset and a BettiCurveDataset sample-for-
    sample (both built from the same labels, so already the same length and
    order) into one dataset returning ({"pi": image, "betti": curve}, y),
    matching the dict-input convention train.py's _to_device/_unpack_batch
    and _PiBettiFusionEncoder.forward already expect."""

    def __init__(self, pi_dataset: PersistenceImageDataset, betti_dataset: BettiCurveDataset):
        if len(pi_dataset) != len(betti_dataset):
            raise ValueError(
                f"pi and betti datasets must have the same length, "
                f"got {len(pi_dataset)} vs {len(betti_dataset)}"
            )
        self.pi_dataset = pi_dataset
        self.betti_dataset = betti_dataset

    def __len__(self) -> int:
        return len(self.pi_dataset)

    def __getitem__(self, idx: int):
        pi_x, y = self.pi_dataset[idx]
        betti_x, _ = self.betti_dataset[idx]
        return {"pi": pi_x, "betti": betti_x}, y


@register("ph_combined")
class PHCombinedExperiment(Experiment):
    """Fuses all four channels -- pi_0, pi_1, betti_0, betti_1 -- into one
    model: PIEncoder(in_channels=2) over the channel-stacked persistence
    images (same data pi_01 trains on) plus SequenceCNNEncoder over the
    concatenated Betti curves (same data betti_cnn_01 trains on), late-fused
    by concatenating both embeddings before the shared regression head
    (encoder_output_dim = 2 * cfg['embedding_dim']).

    Experiment.run() only loads one dataset_path itself, so this method is
    driven with the images file as dataset_path (same file pi_0/pi_1/pi_01
    use) and build_dataset loads the sibling betti file by filename
    convention -- compute_features.py always writes the "images_..." and
    "betti_..." files together, in lockstep, from the same
    diagrams/seeds/labels (see compute_for_split in
    dtm_experiment/compute_features.py), so same-index samples line up
    across the two files exactly the way n_points_head_extra already
    assumes clouds.pkl lines up with any features file by relying on the
    shared "seeds" list. Uses the plain (unweighted) Betti curves --
    weight_by_persistence=True was tried (see the orphaned
    combined_all_weighted results) and measured worse (0.0969 vs 0.0938
    test_loss on the same architecture), so this reverts to betti_0/betti_1
    rather than betti_weighted_0/betti_weighted_1.
    """

    file_key = "images"

    @property
    def subdir(self) -> str:
        return "ph_combined"

    @property
    def extra_meta(self) -> dict:
        meta: dict[str, Any] = {
            "hom_dim": [0, 1],
            "channels": ["pi_0", "pi_1", "betti_0", "betti_1"],
        }
        norm = getattr(self, "_n_points_norm", None)
        if norm is not None:
            meta["n_points_log_mean"] = norm["mean"]
            meta["n_points_log_std"] = norm["std"]
        return meta

    @property
    def encoder_output_dim(self) -> int:
        return 2 * self.cfg["embedding_dim"]

    @staticmethod
    def _sibling_betti_path(dataset_path: Path) -> Path:
        name = dataset_path.name.replace("images", "betti", 1)
        if name == dataset_path.name:
            raise ValueError(
                f"ph_combined expects dataset_path's filename to contain "
                f"'images' so the sibling betti file can be derived "
                f"from it, got {dataset_path.name!r}"
            )
        return dataset_path.parent / name

    def build_dataset(self, payload, labels):
        betti_path = self._sibling_betti_path(self._dataset_path)
        if not betti_path.exists():
            raise FileNotFoundError(
                f"ph_combined needs the sibling betti file {betti_path} "
                f"(derived from {self._dataset_path}) -- run compute_features.py first."
            )
        betti_payload = _load_pickle(betti_path)

        # --- pi branch: per-dim z-score, same as PersistenceImageExperiment ---
        if not hasattr(self, "_image_norm"):
            self._image_norm = {}

        image_tensors: dict[int, np.ndarray] = {}
        for dim in (0, 1):
            raw = np.asarray(payload["image_tensors"][dim], dtype=np.float64)
            if dim not in self._image_norm:
                mean = float(raw.mean())
                std = float(raw.std())
                self._image_norm[dim] = {"mean": mean, "std": std if std > 0 else 1.0}
            norm = self._image_norm[dim]
            image_tensors[dim] = ((raw - norm["mean"]) / norm["std"]).astype(np.float32)

        pi_dataset = PersistenceImageDataset(
            {"image_tensors": image_tensors}, labels, homology_dims=[0, 1], return_dict=False,
        )

        # --- betti branch: per-dim z-score, same as BettiCurveCNNExperiment ---
        if not hasattr(self, "_curve_norm"):
            self._curve_norm = {}

        curve_matrices: dict[str, np.ndarray] = {}
        for dim in (0, 1):
            matrix_key = f"betti{dim}_matrix"
            raw = np.asarray(betti_payload[matrix_key], dtype=np.float64)
            if dim not in self._curve_norm:
                mean = float(raw.mean())
                std = float(raw.std())
                self._curve_norm[dim] = {"mean": mean, "std": std if std > 0 else 1.0}
            norm = self._curve_norm[dim]
            curve_matrices[matrix_key] = ((raw - norm["mean"]) / norm["std"]).astype(np.float32)

        betti_dataset = BettiCurveDataset(curve_matrices, labels, homology_dims=[0, 1])

        return _FusedPiBettiDataset(pi_dataset, betti_dataset)

    def extract_head_extra(self, payload, dataset_path: Path) -> np.ndarray:
        n_x = n_points_head_extra(self, payload, dataset_path)
        return persistence_entropy_head_extra(self, payload, (0, 1), n_x)

    def build_encoder(self, dataset):
        from cloudforger.nn.encoders.persistence_image import PIEncoder
        from cloudforger.nn.encoders.sequence_cnn import SequenceCNNEncoder

        embedding_dim = self.cfg["embedding_dim"]
        pi_encoder = PIEncoder(in_channels=2, embedding_dim=embedding_dim)
        betti_encoder = SequenceCNNEncoder(
            input_dim=dataset.betti_dataset.input_dim,
            embedding_dim=embedding_dim,
            pool_size=5,
        )
        return _PiBettiFusionEncoder(pi_encoder, betti_encoder)
