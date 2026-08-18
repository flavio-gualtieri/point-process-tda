# src/cloudforger/experiments/pi_multik/vectorized_multik.py
"""Generalizes pi_multik.py's late-fusion-across-k design across a
`vectorization` axis (persistence_image / landscape / silhouette /
persistence_statistics) and an `encoder_path` axis (native CNN / shared
flatten-MLP), registered as method: vec_multik. Reuses, rather than
reimplements, everything vectorization-agnostic:

  - pi_multik.load_multik_split for the diagram side (k-dependent seed
    intersection, target alignment, n(x) join) -- filtration/vectorization-
    agnostic already, same reuse betti_multik.py/betti_cnn.py rely on.
  - pi_multik.build_extra for the [log N(x), optional per-(dim,k) entropy]
    side-vector -- likewise agnostic to what the main per-k tensor holds.
  - encoder_bank.EncoderBank's shared/independent k-batch-folding, via its
    encoder_factory escape hatch (see that module) -- so "fold the k-axis
    into batch, shared weights, flat-concat" is the SAME code for every
    vectorization x encoder_path combination here, not reimplemented per
    arm.
  - cloudforger.encoders.scaleconv_pi.ConvFusion for the optional conv
    fusion_mode, exactly as pi_multik.py/betti_multik.py use it.

vectorization="persistence_image", encoder_path="native" -- i.e. what
method: pi_multik already does -- is deliberately NOT reimplemented here:
run() delegates that one combination straight to
pi_multik.PIMultiKExperiment.run(), so it is byte-identical to method:
pi_multik for the same seed (same class, same code path), not merely
numerically close. See tests/test_pipeline_e2e.py's parity test.

Every other combination builds its own (N, n_k, C, ...) tensor via one of
the build_<vectorization>_tensor functions below (all following the same
fit-on-train-idx/apply-frozen convention as pi_multik.build_pi_tensor/
betti_multik.build_betti_tensor -- see those functions), then feeds it
through an EncoderBank whose per-k encoder is chosen by _build_bank:

  persistence_image + mlp   -> build_pi_tensor (reused) + FlattenMLPEncoder
  landscape + native        -> build_landscape_tensor + CoordConvPIEncoder
                                (landscape's (K, G) raster IS a 2-D image,
                                 same conv stack pi_multik.py already uses)
  landscape + mlp           -> build_landscape_tensor + FlattenMLPEncoder
  silhouette + native       -> build_silhouette_tensor + SilhouetteConv1DEncoder
  silhouette + mlp          -> build_silhouette_tensor + FlattenMLPEncoder
  persistence_statistics    -> build_stats_tensor + FlattenMLPEncoder
    (+ mlp only -- no raster/curve for a native CNN, see run())

p (silhouette's amplitude-weighting exponent, see vectorization/landscapes/
tent.py's silhouette_from_tents) is an experimental axis, not a tuned
hyperparameter: separate arms of a sweep, each its own results row/subdir
(see subdir below), never stacked as channels and never used to pick a
"winning" p."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset, TensorDataset

from cloudforger.baselines import vihrs
from cloudforger.core.splits import train_val_test_indices
from cloudforger.encoders.encoder_bank import EncoderBank
from cloudforger.encoders.flatten_mlp import FlattenMLPEncoder
from cloudforger.encoders.scaleconv_pi import ConvFusion
from cloudforger.encoders.silhouette_conv import SilhouetteConv1DEncoder
from cloudforger.experiments.base import register
from cloudforger.experiments.common import MultiSourceExperiment, prepare_device, save_results
from cloudforger.experiments.pi_multik import pi_multik
from cloudforger.models.heads.paramest import ParameterEstimator
from cloudforger.training.train import evaluate, evaluate_per_target, train_one_epoch
from cloudforger.vectorization.landscapes.calibrated import build_calibrated_landscape, build_calibrated_silhouette
from cloudforger.vectorization.scalar_features import REGISTRY as FEATURE_REGISTRY

VECTORIZATIONS = ("persistence_image", "landscape", "silhouette", "persistence_statistics")
ENCODER_PATHS = ("native", "mlp")


# ---------------------------------------------------------------------------
# Tensor builders -- one per vectorization, same fit(train_idx)/apply(fitted)
# convention as pi_multik.build_pi_tensor / betti_multik.build_betti_tensor.
# ---------------------------------------------------------------------------

def build_landscape_tensor(
    split: dict[str, Any],
    k_values: list[int],
    homology_dims: tuple[int, ...],
    G: int,
    K: int | dict[int, int] | None,
    q: float,
    pad_factor: float,
    K_coverage: float = 0.99,
    K_cap: int = 16,
    K_rel_threshold: float = 0.01,
    train_idx: np.ndarray | None = None,
    fitted: list | None = None,
) -> tuple[np.ndarray, list]:
    """(N, n_k, C, K, G) float32 -- C = len(homology_dims) landscape
    channels, K = landscape layers (see build_calibrated_landscape's own K
    argument; NOT the DTM neighborhood scale k_values iterates over -- the
    two are unrelated axes that happen to share a letter). No per-diagram
    amplitude normalization (see landscapes/calibrated.py's module
    docstring): raw tent heights reach the encoder unchanged, same as
    build_pi_tensor's raw persistence-image pixels."""
    fit = fitted is None
    if fit:
        if train_idx is None:
            raise ValueError("build_landscape_tensor: train_idx is required when fitting (fitted=None).")
        fitted = []

    per_k: list[np.ndarray] = []
    for ki, k in enumerate(k_values):
        diagrams_k = split["diagrams_per_k"][k]
        if fit:
            calibration_diagrams = [diagrams_k[i] for i in train_idx]
            landscape = build_calibrated_landscape(
                calibration_diagrams, homology_dims=homology_dims, G=G, K=K, q=q, pad_factor=pad_factor,
                K_coverage=K_coverage, K_cap=K_cap, K_rel_threshold=K_rel_threshold, verbose=False,
            )
            fitted.append(landscape)
        else:
            landscape = fitted[ki]
        per_diagram = [landscape.transform(d) for d in diagrams_k]  # list[{dim: (K, G)}]
        channel_list = [np.stack([pd[dim] for pd in per_diagram]) for dim in homology_dims]  # each (N, K, G)
        per_k.append(np.stack(channel_list, axis=1))  # (N, C, K, G)

    return np.stack(per_k, axis=1).astype(np.float32), fitted  # (N, n_k, C, K, G)


def build_silhouette_tensor(
    split: dict[str, Any],
    k_values: list[int],
    homology_dims: tuple[int, ...],
    G: int,
    p: float,
    q: float,
    pad_factor: float,
    train_idx: np.ndarray | None = None,
    fitted: list | None = None,
) -> tuple[np.ndarray, list]:
    """(N, n_k, 2*C, G) float32 -- for each dim in homology_dims, TWO
    channels: the normalized silhouette phi^(p) and its unnormalized
    numerator (see tent.silhouette_from_tents' docstring for why both are
    kept -- the denominator discards feature count, which likely matters
    for parameter recovery). Channel order per k: [dim0_norm, dim0_unnorm,
    dim1_norm, dim1_unnorm, ...]. No per-diagram amplitude normalization,
    same reasoning as build_landscape_tensor."""
    fit = fitted is None
    if fit:
        if train_idx is None:
            raise ValueError("build_silhouette_tensor: train_idx is required when fitting (fitted=None).")
        fitted = []

    per_k: list[np.ndarray] = []
    for ki, k in enumerate(k_values):
        diagrams_k = split["diagrams_per_k"][k]
        if fit:
            calibration_diagrams = [diagrams_k[i] for i in train_idx]
            silhouette = build_calibrated_silhouette(
                calibration_diagrams, homology_dims=homology_dims, G=G, p=p, q=q, pad_factor=pad_factor, verbose=False,
            )
            fitted.append(silhouette)
        else:
            silhouette = fitted[ki]
        per_diagram = [silhouette.transform(d) for d in diagrams_k]  # list[{dim: {"silhouette":.., "silhouette_unnormalized":..}}]
        channel_list = []
        for dim in homology_dims:
            channel_list.append(np.stack([pd[dim]["silhouette"] for pd in per_diagram]))               # (N, G)
            channel_list.append(np.stack([pd[dim]["silhouette_unnormalized"] for pd in per_diagram]))   # (N, G)
        per_k.append(np.stack(channel_list, axis=1))  # (N, 2*C, G)

    return np.stack(per_k, axis=1).astype(np.float32), fitted  # (N, n_k, 2*C, G)


def _fit_col_zscore(values: np.ndarray) -> dict[str, np.ndarray]:
    """Per-column (not global-scalar) fit/apply-frozen zscore -- unlike
    experiments/common.py's fit_zscore/apply_zscore (a single mean/std over
    the whole array), persistence_statistics' 18 columns are heterogeneous
    in scale (a raw count next to a filtration-value quantile next to an
    entropy), and this feeds a plain MLP with no BatchNorm/conv-bias to
    absorb that -- unlike build_pi_tensor/build_landscape_tensor/
    build_silhouette_tensor, which are deliberately left unnormalized (see
    landscapes/calibrated.py's module docstring)."""
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return {"mean": mean, "std": std}


def _apply_col_zscore(values: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return (values - norm["mean"]) / norm["std"]


def build_stats_tensor(
    split: dict[str, Any],
    k_values: list[int],
    homology_dims: tuple[int, ...],
    train_idx: np.ndarray | None = None,
    fitted: list | None = None,
) -> tuple[np.ndarray, list]:
    """(N, n_k, C, 18) float32, C = len(homology_dims), 18 =
    len(persistence_statistics.STAT_NAMES) -- calibration-free (a pure
    per-diagram function, same as persistence_entropy -- see
    persistence_statistics.py's docstring), but per-column z-scored,
    fit-on-train-idx/apply-frozen like every other population statistic in
    this pipeline (label_norm, n(x), entropy)."""
    fit = fitted is None
    if fit:
        if train_idx is None:
            raise ValueError("build_stats_tensor: train_idx is required when fitting (fitted=None).")
        fitted = []

    feature = FEATURE_REGISTRY.build("persistence_statistics", homology_dims=homology_dims)
    per_k: list[np.ndarray] = []
    for ki, k in enumerate(k_values):
        diagrams_k = split["diagrams_per_k"][k]
        per_diagram = [feature.compute(d) for d in diagrams_k]  # list[{dim: (18,)}]
        channel_list = [np.stack([pd[dim] for pd in per_diagram]) for dim in homology_dims]  # each (N, 18)
        raw = np.stack(channel_list, axis=1)  # (N, C, 18)
        if fit:
            norm = _fit_col_zscore(raw[train_idx])
            fitted.append(norm)
        else:
            norm = fitted[ki]
        per_k.append(_apply_col_zscore(raw, norm))

    return np.stack(per_k, axis=1).astype(np.float32), fitted  # (N, n_k, C, 18)


def _build_tensor(
    split: dict[str, Any],
    k_values: list[int],
    homology_dims: tuple[int, ...],
    vectorization: str,
    cfg: dict[str, Any],
    train_idx: np.ndarray | None,
    fitted: list | None,
) -> tuple[np.ndarray, list]:
    q = float(cfg.get("pd_calibration_coverage", 0.99))
    pad_factor = float(cfg.get("pad_factor", cfg.get("pad", 1.05)))

    if vectorization == "landscape":
        return build_landscape_tensor(
            split, k_values, homology_dims,
            G=int(cfg.get("G", 128)), K=cfg.get("K"), q=q, pad_factor=pad_factor,
            K_coverage=float(cfg.get("K_coverage", 0.99)), K_cap=int(cfg.get("K_cap", 16)),
            K_rel_threshold=float(cfg.get("K_rel_threshold", 0.01)),
            train_idx=train_idx, fitted=fitted,
        )
    if vectorization == "silhouette":
        return build_silhouette_tensor(
            split, k_values, homology_dims,
            G=int(cfg.get("G", 128)), p=float(cfg.get("p", 1.0)), q=q, pad_factor=pad_factor,
            train_idx=train_idx, fitted=fitted,
        )
    if vectorization == "persistence_statistics":
        return build_stats_tensor(split, k_values, homology_dims, train_idx=train_idx, fitted=fitted)
    if vectorization == "persistence_image":
        # encoder_path == "mlp" only ever reaches here -- native delegates
        # to pi_multik.PIMultiKExperiment entirely (see run() below).
        return pi_multik.build_pi_tensor(
            split, k_values, homology_dims=homology_dims,
            resolution=int(cfg.get("resolution", 64)), sigma_pixels=float(cfg.get("sigma_pixels", 0.5)),
            coverage=q, train_idx=train_idx, imagers=fitted,
        )
    raise ValueError(f"vec_multik: unknown vectorization {vectorization!r}. Choices: {VECTORIZATIONS}")


def _build_bank(cfg: dict[str, Any], vectorization: str, encoder_path: str, n_k: int, in_channels: int, per_k_shape: tuple[int, ...]) -> EncoderBank:
    """The one place `encoder_path` actually picks a per-k encoder --
    everything else (train loop, tensor shape, fusion/head) is identical
    regardless of which one gets built here. per_k_shape is one sample's
    (C, *raster) shape (i.e. the built tensor's shape[2:])."""
    mode = str(cfg.get("encoder_mode", "shared"))

    if encoder_path == "mlp":
        input_dim = int(np.prod(per_k_shape))
        hidden_dim = int(cfg.get("mlp_hidden_dim", 512))
        dropout = float(cfg.get("mlp_dropout", 0.1))
        factory: Callable[[], nn.Module] = lambda: FlattenMLPEncoder(
            input_dim=input_dim, embedding_dim=128, hidden_dim=hidden_dim, dropout=dropout,
        )
        # Output is fixed at 128 by FlattenMLPEncoder's own construction
        # (see that class's docstring) -- embedding_dim passed to the bank
        # here must match, not read from cfg["embedding_dim"].
        return EncoderBank(mode=mode, n_k=n_k, in_channels=in_channels, embedding_dim=128, encoder_factory=factory)

    embedding_dim = int(cfg.get("embedding_dim", 128))
    dropout = float(cfg.get("dropout", 0.2))
    conv_channels = tuple(cfg.get("conv_channels", (32, 64, 128)))

    if vectorization == "landscape":
        K_layers = per_k_shape[1]  # (C, K, G)
        if K_layers < 4:
            raise ValueError(
                f"vec_multik: landscape native encoder needs K >= 4 (two 2x2 pools in CoordConvPIEncoder), got K={K_layers}."
            )
        pool_type = str(cfg.get("pool_type", "max"))
        return EncoderBank(
            mode=mode, n_k=n_k, in_channels=in_channels, embedding_dim=embedding_dim,
            conv_channels=conv_channels, dropout=dropout, pool_type=pool_type,
        )
    if vectorization == "silhouette":
        kernel_size = int(cfg.get("kernel_size", 3))
        factory = lambda: SilhouetteConv1DEncoder(
            in_channels=in_channels, embedding_dim=embedding_dim, conv_channels=conv_channels,
            kernel_size=kernel_size, dropout=dropout,
        )
        return EncoderBank(mode=mode, n_k=n_k, in_channels=in_channels, embedding_dim=embedding_dim, encoder_factory=factory)

    raise ValueError(f"vec_multik: vectorization={vectorization!r} has no native encoder path.")


def _channel_names(vectorization: str, k_values: list[int], homology_dims: tuple[int, ...]) -> list[str]:
    if vectorization == "silhouette":
        return [f"k{k}_h{d}_{variant}" for k in k_values for d in homology_dims for variant in ("norm", "unnorm")]
    return [f"k{k}_h{d}" for k in k_values for d in homology_dims]


def _serialize_fitted(vectorization: str, fitted: list) -> Any:
    """Best-effort JSON-friendly snapshot of the per-k fitted
    calibration/normalization, for cfg_meta -- mirrors pi_multik.py's
    imager_params / betti_multik.py's betti_params, so a run's exact
    calibration is recoverable from results.json alone."""
    if vectorization == "persistence_statistics":
        return [{"mean": norm["mean"].tolist(), "std": norm["std"].tolist()} for norm in fitted]
    return [f.params for f in fitted]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VectorizedMultiK(nn.Module):
    """Structurally identical to pi_multik.py's PIMultiK / betti_multik.py's
    BettiMultiK (bank -> optional ConvFusion/flat-concat -> shared
    regression head) -- the only difference is that `bank` is passed in
    already built, instead of constructed from one hardcoded per-k encoder
    class, so the same forward()/fusion/head code serves every
    vectorization x encoder_path combination _build_bank can produce."""

    def __init__(
        self,
        bank: EncoderBank,
        embedding_dim: int,
        n_k: int,
        n_extra: int,
        n_targets: int,
        fusion_mode: str = "concat",
        fusion_pool: str = "avg",
        fusion_dropout: float = 0.0,
        scale_fusion_hidden: int = 128,
        scale_fusion_out_dim: int = 128,
        scale_fusion_kernel_size: int = 3,
        scale_fusion_dropout: float = 0.0,
        head_hidden_dims: tuple[int, ...] = (64, 32),
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.bank = bank
        self.n_k = n_k
        self.fusion_mode = fusion_mode
        if fusion_mode == "concat":
            self.fusion = None
            fused_dim = n_k * embedding_dim
        elif fusion_mode == "conv":
            self.fusion = ConvFusion(
                n_k=n_k, embedding_dim=embedding_dim, hidden=scale_fusion_hidden,
                out_dim=scale_fusion_out_dim, kernel_size=scale_fusion_kernel_size,
                dropout=scale_fusion_dropout, pool=fusion_pool,
            )
            fused_dim = self.fusion.out_dim
        else:
            raise ValueError(f"VectorizedMultiK: fusion_mode must be 'concat' or 'conv', got {fusion_mode!r}.")
        self.fusion_dropout = nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity()

        self.head = ParameterEstimator(
            embedding_dim=fused_dim + n_extra, n_params=n_targets,
            hidden_dims=head_hidden_dims, dropout=head_dropout,
        )

    def forward(self, x: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        seq = self.bank(x)  # (B, K, E)
        if self.fusion_mode == "concat":
            fused = seq.reshape(seq.shape[0], -1)
        else:
            fused = self.fusion(seq)
        fused = self.fusion_dropout(fused)
        return self.head(torch.cat([fused, extra], dim=1))


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

@register("vec_multik")
class VectorizedMultiKExperiment(MultiSourceExperiment):
    file_keys = ("clouds", "images")

    @property
    def subdir(self) -> str:
        vectorization = self.cfg.get("vectorization", "persistence_image")
        encoder_path = self.cfg.get("encoder_path", "native")
        tag = f"vec_multik_{vectorization}_{encoder_path}"
        if vectorization == "silhouette":
            tag += f"_p{float(self.cfg.get('p', 1.0)):g}"
        return tag

    def run(
        self,
        dataset_paths: dict[str, Any],
        output_dir: Path,
        adversarial_paths: dict[str, Any] | None = None,
    ) -> dict:
        vectorization = self.cfg.get("vectorization", "persistence_image")
        encoder_path = self.cfg.get("encoder_path", "native")
        if vectorization not in VECTORIZATIONS:
            raise ValueError(f"vec_multik: vectorization must be one of {VECTORIZATIONS}, got {vectorization!r}.")
        if encoder_path not in ENCODER_PATHS:
            raise ValueError(f"vec_multik: encoder_path must be one of {ENCODER_PATHS}, got {encoder_path!r}.")
        if vectorization == "persistence_statistics" and encoder_path != "mlp":
            raise ValueError(
                "vec_multik: persistence_statistics only supports encoder_path='mlp' "
                "(no raster/curve for a native CNN)."
            )

        if vectorization == "persistence_image" and encoder_path == "native":
            # Exact parity path -- see module docstring. Delegates entirely
            # to the unmodified pi_multik.py implementation; output_dir is
            # already this experiment's own subdir (vec_multik_persistence_
            # image_native), computed from `exp.subdir` by scripts/train.py
            # before .run() is ever called, so results land in their own
            # directory rather than colliding with method: pi_multik's.
            return pi_multik.PIMultiKExperiment(self.cfg).run(
                dataset_paths, output_dir, adversarial_paths=adversarial_paths
            )

        label_names = tuple(self.cfg.get("target_label_names"))
        k_values = list(self.cfg["k_values"])
        homology_dims = tuple(self.cfg.get("homology_dims", (0, 1)))
        include_entropy = bool(self.cfg.get("include_entropy", False))
        seed = self.cfg["seed"]
        device = prepare_device(seed)

        train_split = pi_multik.load_multik_split(
            k_values, list(dataset_paths["images"]), Path(dataset_paths["clouds"]), label_names, tag="train_test",
            homology_dims=homology_dims,
        )
        if train_split is None:
            raise FileNotFoundError(f"diagrams missing for some k in {k_values} under {dataset_paths['images']}.")

        adv_split = None
        if adversarial_paths is not None:
            adv_split = pi_multik.load_multik_split(
                k_values, list(adversarial_paths["images"]), Path(adversarial_paths["clouds"]),
                label_names, tag="adversarial", homology_dims=homology_dims,
            )

        n = len(train_split["targets"])
        train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

        label_norm = vihrs.fit_log_zscore(train_split["targets"][train_idx])
        targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)

        main_tensor, fitted = _build_tensor(
            train_split, k_values, homology_dims, vectorization, self.cfg, train_idx=train_idx, fitted=None,
        )
        extra, n_norm, entropy_norms = pi_multik.build_extra(train_split, train_idx, include_entropy=include_entropy)

        full_dataset = TensorDataset(
            torch.from_numpy(main_tensor), torch.from_numpy(extra), torch.from_numpy(targets_std),
        )

        def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            return DataLoader(Subset(full_dataset, idx), batch_size=self.cfg["batch_size"], shuffle=shuffle)

        train_loader, val_loader, test_loader = _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)

        per_k_shape = main_tensor.shape[2:]  # (C, *raster)
        in_channels = per_k_shape[0]
        feature_dim = int(np.prod(per_k_shape[1:])) if len(per_k_shape) > 1 else int(per_k_shape[0])

        bank = _build_bank(self.cfg, vectorization, encoder_path, n_k=len(k_values), in_channels=in_channels, per_k_shape=per_k_shape)

        model = VectorizedMultiK(
            bank=bank,
            embedding_dim=bank.embedding_dim,
            n_k=len(k_values),
            n_extra=extra.shape[1],
            n_targets=len(label_names),
            fusion_mode=str(self.cfg.get("fusion_mode", "concat")),
            fusion_pool=str(self.cfg.get("fusion_pool", "avg")),
            fusion_dropout=float(self.cfg.get("fusion_dropout", 0.0)),
            scale_fusion_hidden=int(self.cfg.get("scale_fusion_hidden", 128)),
            scale_fusion_out_dim=int(self.cfg.get("scale_fusion_out_dim", 128)),
            scale_fusion_kernel_size=int(self.cfg.get("scale_fusion_kernel_size", 3)),
            scale_fusion_dropout=float(self.cfg.get("scale_fusion_dropout", 0.0)),
            head_hidden_dims=tuple(self.cfg.get("head_hidden_dims", (64, 32))),
            head_dropout=self.cfg.get("head_dropout", 0.1),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.get("lr", 1e-3), weight_decay=self.cfg.get("weight_decay", 1e-4))
        loss_fn = nn.MSELoss()

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss, best_state = float("inf"), None
        n_epochs = self.cfg["n_epochs"]
        patience = self.cfg.get("early_stopping_patience")
        epochs_no_improve = 0

        for epoch in range(1, n_epochs + 1):
            train_loss, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss, _ = evaluate(model, val_loader, loss_fn, device)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epoch == 1 or epoch % 25 == 0 or epoch == n_epochs:
                print(f"[{self.tag} seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")
            if patience is not None and epochs_no_improve >= patience:
                print(f"[{self.tag} seed={seed}] early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break

        model.load_state_dict(best_state)
        test_loss, _ = evaluate(model, test_loader, loss_fn, device)
        test_loss_per_target = dict(zip(label_names, evaluate_per_target(model, test_loader, device).tolist()))
        print(f"\n[{self.tag} seed={seed}] test loss {test_loss:.4f}")

        adversarial_loss = None
        adversarial_loss_per_target = None
        if adv_split is not None:
            adv_targets_std = vihrs.apply_log_zscore(adv_split["targets"], label_norm).astype(np.float32)
            adv_tensor, _ = _build_tensor(adv_split, k_values, homology_dims, vectorization, self.cfg, train_idx=None, fitted=fitted)
            adv_extra, _, _ = pi_multik.build_extra(
                adv_split, None, n_norm=n_norm, entropy_norms=entropy_norms, include_entropy=include_entropy,
            )
            adv_ds = TensorDataset(
                torch.from_numpy(adv_tensor), torch.from_numpy(adv_extra), torch.from_numpy(adv_targets_std),
            )
            adv_loader = DataLoader(adv_ds, batch_size=self.cfg["batch_size"], shuffle=False)
            adversarial_loss, _ = evaluate(model, adv_loader, loss_fn, device)
            adversarial_loss_per_target = dict(
                zip(label_names, evaluate_per_target(model, adv_loader, device).tolist())
            )
            print(f"[{self.tag} seed={seed}] adversarial loss {adversarial_loss:.4f}")

        cfg_meta = {
            **self.cfg,
            "vectorization": vectorization,
            "encoder_path": encoder_path,
            "p": float(self.cfg.get("p", 1.0)) if vectorization == "silhouette" else None,
            "feature_dim": feature_dim,
            "channels": _channel_names(vectorization, k_values, homology_dims),
            "vectorizer_params": _serialize_fitted(vectorization, fitted),
        }
        save_results(
            output_dir, model=model, best_state=best_state, history=history, cfg=cfg_meta,
            test_loss=test_loss, label_names=list(label_names), label_norm=label_norm,
            test_loss_per_target=test_loss_per_target, adversarial_loss=adversarial_loss,
            adversarial_loss_per_target=adversarial_loss_per_target,
            adversarial_path=adversarial_paths.get("clouds") if adversarial_paths else None,
        )

        result: dict[str, Any] = {"seed": seed, "test_loss": test_loss, "test_loss_per_target": test_loss_per_target}
        if adversarial_loss is not None:
            result["adversarial_loss"] = adversarial_loss
            result["adversarial_loss_per_target"] = adversarial_loss_per_target
        return result
