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
order -- select_mph_channels below uses its key order to relabel channels.

select_mph_channels/build_mph_tensor/load_mph_split are factored out (rather
than inlined in build_dataset, as an earlier version of this module had them)
for the same reason pi_multik.py factors out load_multik_split/build_pi_tensor:
mph_fusion.py's vihrs+mph_pi fusion needs the identical load-from-path +
per-channel-zscore logic, just joined against vihrs's L(r)-r population by
seed instead of handed straight to a Dataset -- see that module's docstring.

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
from typing import Any

import numpy as np

from cloudforger.core.io import load_pickle
from cloudforger.nn.data import PersistenceImageDataset
from cloudforger.nn.experiments.base import Experiment, register, n_points_head_extra


def select_mph_channels(
    payload: dict[str, Any], homology_dims: tuple[int, ...] | None = None
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Validate + select the requested homology-dim channels from an
    mph_image payload's (N, D, R, R) image_tensors, using imager_params's
    key order to recover which literal dim each channel-axis position is
    (see module docstring). homology_dims=None selects every dim the bundle
    has, in its own (imager_params) order. Returns (raw (N, len(dims), R, R)
    float64, the dims actually selected, in that order)."""
    raw = np.asarray(payload["image_tensors"], dtype=np.float64)  # (N, D, R, R)
    available = list(payload["imager_params"].keys())  # channel axis order
    if len(available) != raw.shape[1]:
        raise ValueError(
            f"imager_params has {len(available)} dims but image_tensors has {raw.shape[1]} "
            "channels -- payload is internally inconsistent."
        )
    requested = tuple(homology_dims) if homology_dims is not None else tuple(available)
    missing = [d for d in requested if d not in available]
    if missing:
        raise KeyError(
            f"homology_dims {requested} requests dims {missing} not present in this bundle "
            f"(available: {available}) -- these must have been computed by featurize_bifiltration.py "
            "(bifiltration.params.homology_dims)."
        )
    idx = [available.index(d) for d in requested]
    return raw[:, idx], requested


def build_mph_tensor(
    raw: np.ndarray, channel_norms: list[dict[str, float]] | None = None
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Per-channel (fit-once/freeze) z-score -- same rationale as
    pi_multik.build_pi_tensor: PI pixels are raw Gaussian-kernel sums with a
    huge dynamic range. raw: (N, D, R, R). channel_norms=None fits (train);
    pass the returned list back in to apply frozen (val/test/adversarial).
    Position-indexed (not dim-keyed), matching pi_multik.build_pi_tensor --
    callers must keep channel order identical across fit/apply calls, which
    every caller here does (same homology_dims resolution both times)."""
    fit = channel_norms is None
    if fit:
        channel_norms = []
    normed = np.empty(raw.shape, dtype=np.float32)
    for i in range(raw.shape[1]):
        channel = raw[:, i]
        if fit:
            mean = float(channel.mean())
            std = float(channel.std())
            channel_norms.append({"mean": mean, "std": std if std > 0 else 1.0})
        norm = channel_norms[i]
        normed[:, i] = (channel - norm["mean"]) / norm["std"]
    return normed, channel_norms


def load_mph_split(
    image_path: Path,
    clouds_path: Path,
    label_names: tuple[str, ...],
    tag: str,
    homology_dims: tuple[int, ...] | None = None,
) -> dict[str, Any] | None:
    """Load one mph_image.pkl split (train or adversarial) + join n(x) from
    the sibling clouds.pkl by seed. Single-source counterpart of
    pi_multik.load_multik_split -- mph_pi has no k-axis to intersect across,
    just one bifiltration's image file."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"  [{tag}] missing {image_path} -- skipping split.")
        return None
    payload = load_pickle(image_path)
    raw, dims = select_mph_channels(payload, homology_dims)

    tda_label_names = list(payload["label_names"])
    missing = [name for name in label_names if name not in tda_label_names]
    if missing:
        raise KeyError(f"[{tag}] label_names {tda_label_names} is missing {missing} from {label_names}")
    col_idx = [tda_label_names.index(name) for name in label_names]
    targets = np.asarray(payload["labels"], dtype=float)[:, col_idx]

    clouds = load_pickle(Path(clouds_path))
    n_points_by_seed = {int(c["seed"]): c["n_points"] for c in clouds}
    seeds = np.asarray(payload["seeds"])
    n_points = np.array([n_points_by_seed[int(s)] for s in seeds], dtype=np.float64)

    return {
        "image_tensors": raw,  # (N, D, R, R) float64, D = len(dims)
        "dims": dims,
        "n_points": n_points,
        "targets": targets,
        "seeds": seeds,
        "label_names": list(label_names),
    }


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
        requested = self.cfg.get("homology_dims")
        raw, self._dims = select_mph_channels(payload, tuple(requested) if requested else None)

        if not hasattr(self, "_channel_norms"):
            self._channel_norms = None
        normed, self._channel_norms = build_mph_tensor(raw, self._channel_norms)

        image_tensors = {dim: normed[:, i] for i, dim in enumerate(self._dims)}
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
