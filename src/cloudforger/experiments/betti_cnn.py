# src/cloudforger/experiments/betti_cnn.py
"""One filtration, one homology dimension: pick a filtration (the config's
single-entry filtration: list) and a homology dimension (method.params.
hom_dim), compute that one Betti curve beta_{hom_dim}(t), sample it evenly
at grid_size points, run the resulting vector through a Conv1d-then-pool
stack (SequenceCNNEncoder -- Vihrs (2022)'s own conv shape, see that
class's docstring) to obtain a feature vector, then feed that (plus the
n(x)/entropy side-vector) to the shared regression head. This is exactly
the literal spec: no multi-k late fusion, no Euler-characteristic channel
-- see betti_multik.py in the pi_multik/ subpackage for that fancier,
multi-filtration/multi-channel sibling, and its module docstring for how
the two differ.

Same name (betti_cnn) and same results-subdir convention
(betti_cnn_<hom_dim>) as the deleted single-filtration betti_cnn.py this
replaces, but NOT the same implementation: the old one read a precomputed
betti_curve.pkl calibrated (build_calibrated_betti_curves) against the
ENTIRE train_test population in scripts/featurize.py, before any
train/val/test split existed -- the same leakage bug pi_multik.py's module
docstring documents and fixes for persistence images, never fixed for
Betti curves until betti_multik.py. This file gets the same fix: calibration
(build_calibrated_betti) is fit fresh per seed, on that seed's train_idx
rows only, computed on the fly from diagrams.pkl (no precomputed features
file). hom_dim is now a method.params field rather than a dimensioned
method-name suffix (betti_cnn_0/betti_cnn_1) -- this file is a
MultiSourceExperiment (it needs train_idx before it can build its
per-sample tensor, which Experiment.run()'s generic build_dataset(payload,
labels) hook has no way to receive -- the same reason pi_multik.py had to
leave Experiment for its own bespoke run()), and MultiSourceExperiment
subclasses are constructed via cls(cfg) with no hom_dim argument (see
experiments/base.py's build_experiment)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset, TensorDataset

from cloudforger.baselines import vihrs
from cloudforger.core.records import load_diagrams
from cloudforger.core.splits import train_val_test_indices
from cloudforger.encoders.sequence_cnn import SequenceCNNEncoder
from cloudforger.experiments.base import register
from cloudforger.experiments.common import MultiSourceExperiment, prepare_device, save_results
from cloudforger.experiments.pi_multik import pi_multik
from cloudforger.models.heads.paramest import ParameterEstimator
from cloudforger.training.train import evaluate, evaluate_per_target, train_one_epoch
from cloudforger.vectorization.scalar_features import REGISTRY as FEATURE_REGISTRY
from cloudforger.vectorization.scalar_features.betti_curve import BettiCurve
from cloudforger.vectorization.scalar_features.calibrated import build_calibrated_betti


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_betti_cnn_split(
    diagram_path: Path,
    clouds_path: Path,
    label_names: tuple[str, ...] | None,
    hom_dim: int,
    tag: str,
) -> dict[str, Any] | None:
    """Load one filtration's diagrams.pkl (no cross-file seed intersection
    needed -- unlike pi_multik.load_multik_split, there's only ever one
    filtration here) and join n(x)/persistence entropy for hom_dim, same
    calibration-free reasoning as load_multik_split's own entropy column
    (a pure per-diagram function, no population fit, so no leakage
    concern -- persistence IMAGES/CURVES are deliberately not built here,
    that's build_betti_curve's job once a train/val/test split exists)."""
    if not Path(diagram_path).exists():
        print(f"  [{tag}] missing {diagram_path} -- skipping split.")
        return None
    diagrams, bundle = load_diagrams(diagram_path)

    tda_label_names = list(bundle["label_names"])
    if label_names is None:
        label_names = tuple(tda_label_names)
    missing = [name for name in label_names if name not in tda_label_names]
    if missing:
        raise KeyError(f"[{tag}] label_names {tda_label_names} is missing {missing} from {label_names}")
    col_idx = [tda_label_names.index(name) for name in label_names]
    targets = np.asarray(bundle["labels"], dtype=float)[:, col_idx]

    entropy_feature = FEATURE_REGISTRY.build("persistence_entropy", homology_dims=(hom_dim,))
    entropy = np.array([entropy_feature.compute(d)[hom_dim] for d in diagrams], dtype=np.float64)

    clouds = _load_pickle(clouds_path)
    n_points_by_seed = {int(c["seed"]): c["n_points"] for c in clouds}
    seeds = np.asarray(bundle["seeds"])
    n_points = np.array([n_points_by_seed[int(s)] for s in seeds], dtype=np.float64)

    return {
        "diagrams": diagrams,
        "entropy_cols": {f"entropy{hom_dim}": entropy},
        "n_points": n_points,
        "targets": targets,
        "seeds": seeds,
        "label_names": list(label_names),
    }


def build_betti_curve(
    split: dict[str, Any],
    hom_dim: int,
    grid_size: int,
    coverage: float,
    range_pad: float = 1.1,
    weight_by_persistence: bool = False,
    train_idx: np.ndarray | None = None,
    betti_feature: BettiCurve | None = None,
) -> tuple[np.ndarray, BettiCurve]:
    """(N, grid_size) float32 -- beta_{hom_dim}(t) evenly sampled at
    grid_size points on a per-seed-calibrated filtration-value grid (see
    build_calibrated_betti). Fit-once/apply-frozen, same convention as
    pi_multik.py's build_pi_tensor / betti_multik.py's build_betti_tensor:
      - Fit (betti_feature=None): train_idx selects which rows of
        split["diagrams"] calibrate the BettiCurve -- the main population's
        train-only fit. Every row of split (train AND val/test) is still
        transformed and returned, just not used to fit.
      - Apply (betti_feature=BettiCurve, train_idx unused/None): every row
        of split is transformed through the given, already-fit BettiCurve
        -- the adversarial-population call."""
    fit = betti_feature is None
    if fit:
        if train_idx is None:
            raise ValueError("build_betti_curve: train_idx is required when fitting (betti_feature=None).")
        calibration_diagrams = [split["diagrams"][i] for i in train_idx]
        betti_feature = build_calibrated_betti(
            calibration_diagrams, homology_dims=(hom_dim,), grid_size=grid_size,
            coverage=coverage, range_pad=range_pad, weight_by_persistence=weight_by_persistence,
        )
    curves = np.stack([betti_feature.compute(d).curves[hom_dim] for d in split["diagrams"]])
    return curves.astype(np.float32), betti_feature


class BettiCNN(nn.Module):
    """One Conv1d-then-pool stack (SequenceCNNEncoder) over a single
    beta_{hom_dim}(t) curve, concatenated with the extra scalar features
    before the shared regression head. The single-filtration/single-channel
    special case of betti_multik.py's BettiMultiK (n_k=1, in_channels=1, no
    Euler-characteristic branch) -- see that module if you want several
    filtrations and/or H0+H1+euler fused together instead of just one
    curve."""

    def __init__(
        self,
        grid_size: int,
        embedding_dim: int,
        n_extra: int,
        n_targets: int,
        n_filters: int = 64,
        kernel_size: int = 7,
        pool_size: int = 2,
        dropout: float = 0.3,
        head_hidden_dims: tuple[int, ...] = (64, 32),
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = SequenceCNNEncoder(
            input_dim=grid_size, embedding_dim=embedding_dim, n_filters=n_filters,
            kernel_size=kernel_size, pool_size=pool_size, dropout=dropout,
        )
        self.head = ParameterEstimator(
            embedding_dim=embedding_dim + n_extra, n_params=n_targets,
            hidden_dims=head_hidden_dims, dropout=head_dropout,
        )

    def forward(self, curve: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(curve)
        return self.head(torch.cat([emb, extra], dim=1))


@register("betti_cnn")
class BettiCNNExperiment(MultiSourceExperiment):
    file_keys = ("clouds", "diagrams")

    @property
    def subdir(self) -> str:
        return f"betti_cnn_{self.cfg['hom_dim']}"

    def run(
        self,
        dataset_paths: dict[str, Any],
        output_dir: Path,
        adversarial_paths: dict[str, Any] | None = None,
    ) -> dict:
        # fetch configs -- hom_dim has no default: "pick a homology
        # dimension" is a deliberate choice this config must make explicit.
        label_names = tuple(self.cfg.get("target_label_names"))
        hom_dim = int(self.cfg["hom_dim"])
        include_entropy = bool(self.cfg.get("include_entropy", False))
        grid_size = int(self.cfg.get("grid_size", 128))
        coverage = float(self.cfg.get("pd_calibration_coverage", 0.99))
        range_pad = float(self.cfg.get("range_pad", 1.1))
        weight_by_persistence = bool(self.cfg.get("weight_by_persistence", False))
        seed = self.cfg["seed"]
        device = prepare_device(seed)

        train_split = load_betti_cnn_split(
            Path(dataset_paths["diagrams"]), Path(dataset_paths["clouds"]), label_names, hom_dim, tag="train_test",
        )
        if train_split is None:
            raise FileNotFoundError(f"{dataset_paths['diagrams']} missing.")

        adv_split = None
        if adversarial_paths is not None:
            adv_split = load_betti_cnn_split(
                Path(adversarial_paths["diagrams"]), Path(adversarial_paths["clouds"]), label_names, hom_dim,
                tag="adversarial",
            )

        n = len(train_split["targets"])
        train_idx, val_idx, test_idx = train_val_test_indices(n, seed)

        label_norm = vihrs.fit_log_zscore(train_split["targets"][train_idx])
        targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)
        curve, betti_feature = build_betti_curve(
            train_split, hom_dim, grid_size=grid_size, coverage=coverage, range_pad=range_pad,
            weight_by_persistence=weight_by_persistence, train_idx=train_idx,
        )
        # build_extra is pi_multik's own helper -- reused here rather than
        # reimplemented, same as betti_multik.py does: it operates on any
        # dict with "n_points"/"entropy_cols" keys, which is exactly what
        # load_betti_cnn_split above returns.
        extra, n_norm, entropy_norms = pi_multik.build_extra(train_split, train_idx, include_entropy=include_entropy)

        full_dataset = TensorDataset(
            torch.from_numpy(curve), torch.from_numpy(extra), torch.from_numpy(targets_std),
        )

        def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
            return DataLoader(Subset(full_dataset, idx), batch_size=self.cfg["batch_size"], shuffle=shuffle)

        train_loader, val_loader, test_loader = _loader(train_idx, True), _loader(val_idx, False), _loader(test_idx, False)

        # BettiCNN.forward(curve, extra) lines up with
        # cloudforger.training.train's (inputs, covariates, labels) 3-tuple
        # batch convention, so the shared train loop applies as-is.
        model = BettiCNN(
            grid_size=grid_size,
            embedding_dim=self.cfg["embedding_dim"],
            n_extra=extra.shape[1],
            n_targets=len(label_names),
            n_filters=int(self.cfg.get("n_filters", 64)),
            kernel_size=int(self.cfg.get("kernel_size", 7)),
            pool_size=int(self.cfg.get("pool_size", 2)),
            dropout=float(self.cfg.get("dropout", 0.3)),
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
            adv_curve, _ = build_betti_curve(
                adv_split, hom_dim, grid_size=grid_size, coverage=coverage, range_pad=range_pad,
                weight_by_persistence=weight_by_persistence, betti_feature=betti_feature,
            )
            adv_extra, _, _ = pi_multik.build_extra(
                adv_split, None, n_norm=n_norm, entropy_norms=entropy_norms, include_entropy=include_entropy,
            )
            adv_ds = TensorDataset(
                torch.from_numpy(adv_curve), torch.from_numpy(adv_extra), torch.from_numpy(adv_targets_std),
            )
            adv_loader = DataLoader(adv_ds, batch_size=self.cfg["batch_size"], shuffle=False)
            adversarial_loss, _ = evaluate(model, adv_loader, loss_fn, device)
            adversarial_loss_per_target = dict(
                zip(label_names, evaluate_per_target(model, adv_loader, device).tolist())
            )
            print(f"[{self.tag} seed={seed}] adversarial loss {adversarial_loss:.4f}")

        cfg_meta = {
            **self.cfg,
            "channels": [f"beta{hom_dim}"],
            # Calibrated grid_range (and the rest of BettiCurve.params), so
            # a seed's exact calibration is recoverable from results.json
            # alone -- mirrors pi_multik.py's imager_params/betti_multik.py's
            # betti_params.
            "betti_params": betti_feature.params,
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
