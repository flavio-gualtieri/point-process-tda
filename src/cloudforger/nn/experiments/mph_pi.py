# src/cloudforger/nn/experiments/mph_pi.py
"""CNN for the multiparameter-persistent-homology image feature
(scripts/featurize_bifiltration.py's mph_image, built from a Bifiltration in
BIFILTRATION_REGISTRY -- e.g. mph_dtm's (rips_radius, dtm_codensity)
bifiltration -- via multipers.signed_measure + build_signed_measure_imagers).

Same architecture as pi_multik's per-branch design: CoordConvPIEncoder
(cloudforger.nn.encoders.coordconv_pi) over the (H_dim0, H_dim1, ...) image
stack, concatenated with [log N] and passed through a ParameterEstimator MLP
head (nn.experiments.base.Experiment.run() wires that composition via
SingleModalModel). There is no k/scale axis to fuse across here -- a
bifiltration's signed measure already folds "scale" into its second
filtration parameter, so one joint multiparameter image pair IS the whole
feature, unlike pi_multik's several single-parameter images at different k.
Architecturally this is exactly PIMultiK(n_k=1, use_fusion=False): with a
single k, pi_multik's per-k CoordConv branch + flat-concat fusion reduces to
one encoder call feeding the head directly, which is what subclassing
Experiment (rather than MultiSourceExperiment) gives for free.

The one real format difference from persistence_image.py's PersistenceImageExperiment
(pi, which this otherwise mirrors): pi's payload["image_tensors"] is already a
{homology_dim: (N, R, R)} dict, built that way by the persistence_image
featurizer. mph_image's payload["image_tensors"] is a plain (N, D, R, R)
ndarray (see featurize_bifiltration.py's stage_b) -- D stacked in
homology_dims order, with no per-channel dim label attached to the array
itself. payload["imager_params"] (dict-keyed by literal homology dim, built
by the SAME homology_dims iteration -- see
vectorizers.signed_measure_image.build_signed_measure_imagers) recovers that
order, so build_dataset below uses its key order to relabel channels before
handing off to the same PersistenceImageDataset pi uses.

Training-recipe note: this subclasses Experiment, so it trains via
base.py's shared _train_and_eval (plain Adam, fixed weight_decay=1e-4, no
early stopping) rather than pi_multik's own bespoke run() (AdamW,
configurable weight_decay, early_stopping_patience). That's a training-loop
difference, not an architecture one -- the model graph is identical to
PIMultiK(n_k=1). Promote to pi_multik's own loop later if exact optimizer
parity turns out to matter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cloudforger.nn.data import PersistenceImageDataset
from cloudforger.nn.experiments.base import Experiment, register, n_points_head_extra


@register("mph_pi")
class MPHImageExperiment(Experiment):
    file_key = "mph_image"

    @property
    def subdir(self) -> str:
        return "mph_pi"

    @property
    def extra_meta(self) -> dict:
        meta: dict = {"hom_dim": list(getattr(self, "_dims", ()))}
        norm = getattr(self, "_n_points_norm", None)
        if norm is not None:
            meta["n_points_log_mean"] = norm["mean"]
            meta["n_points_log_std"] = norm["std"]
        return meta

    def build_dataset(self, payload, labels):
        # Persistence image pixels are raw Gaussian-kernel sums with a huge
        # dynamic range -- z-score per channel before this reaches the CNN,
        # same fit-once/freeze-for-adversarial pattern as pi/pi_multik.
        raw = np.asarray(payload["image_tensors"], dtype=np.float64)  # (N, D, R, R)

        available = list(payload["imager_params"].keys())  # channel axis order -- see module docstring
        if len(available) != raw.shape[1]:
            raise ValueError(
                f"imager_params has {len(available)} dims but image_tensors has {raw.shape[1]} "
                "channels -- payload is internally inconsistent."
            )
        requested = tuple(self.cfg.get("homology_dims", available))
        missing = [d for d in requested if d not in available]
        if missing:
            raise KeyError(
                f"homology_dims {requested} requests dims {missing} not present in this bundle "
                f"(available: {available}) -- these must have been computed by featurize_bifiltration.py "
                "(bifiltration.params.homology_dims)."
            )
        self._dims = requested

        if not hasattr(self, "_image_norm"):
            self._image_norm = {}

        image_tensors: dict[int, np.ndarray] = {}
        for dim in self._dims:
            channel = raw[:, available.index(dim)]
            if dim not in self._image_norm:
                mean = float(channel.mean())
                std = float(channel.std())
                self._image_norm[dim] = {"mean": mean, "std": std if std > 0 else 1.0}
            norm = self._image_norm[dim]
            image_tensors[dim] = ((channel - norm["mean"]) / norm["std"]).astype(np.float32)

        return PersistenceImageDataset(
            {"image_tensors": image_tensors}, labels, homology_dims=list(self._dims), return_dict=False
        )

    def extract_head_extra(self, payload, dataset_path: Path) -> np.ndarray:
        return n_points_head_extra(self, payload, dataset_path)

    def build_encoder(self, dataset):
        from cloudforger.nn.encoders.coordconv_pi import CoordConvPIEncoder

        return CoordConvPIEncoder(
            in_channels=len(self._dims),
            embedding_dim=self.cfg["embedding_dim"],
            conv_channels=tuple(self.cfg.get("conv_channels", (32, 64, 128))),
            dropout=self.cfg.get("dropout", 0.2),
            pool_type=str(self.cfg.get("pool_type", "max")),
        )
