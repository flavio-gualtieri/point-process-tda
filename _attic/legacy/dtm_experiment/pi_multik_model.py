# dtm_experiment/pi_multik_model.py
"""
Multi-k persistence-image model: stacks the 2-channel (H0, H1) persistence
image computed at EACH DTM k in K_VALUES into one (2*len(K_VALUES))-channel
image, fed through a single PIEncoder. This is the "k sweep, PI branch only"
model from the k-sweep discussion -- unlike fusion_model.py's L(r)-r+TDA
fusion (several different MODALITIES, late-fused via concatenation), this
fuses the SAME modality (persistence images) at several density SCALES (k)
by stacking them as channels into ONE encoder, so a single conv kernel can
see all k's jointly rather than only meeting them at a final concat layer.

Channel order (fixed, documented here since nothing in the tensor itself
labels it): for k in K_VALUES, in order: [pi0(k), pi1(k)]. E.g. for
K_VALUES = [5, 10, 15]:
    channel 0: H0 image at k=5    channel 1: H1 image at k=5
    channel 2: H0 image at k=10   channel 3: H1 image at k=10
    channel 4: H0 image at k=15   channel 5: H1 image at k=15

Not built on cloudforger.nn.experiments.Experiment, for the same reason
fusion_model.py isn't (see that file's docstring): Experiment.run() loads
exactly one dataset_path per call, with no hook for joining several files by
seed. That join is unavoidable here because compute_features.py's DTM
empty-diagram filter (see its _is_broken docstring) drops a different,
k-dependent set of clouds at each k -- images_dtm_k5.pkl, images_dtm_k10.pkl
and images_dtm_k15.pkl each survive filtering with a DIFFERENT subset of
seeds, so the three files must be intersected by seed before their channels
can be stacked, not just concatenated index-for-index.

Assumes, for each k in K_VALUES:
    <data_dir>/images_dtm_k<k>.pkl (+ adversarial_images_dtm_k<k>.pkl)
already exist (from `compute_features.py --process <process> --k <k>`).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "runners" / "params"))

import run_vihrs as vihrs
from cloudforger.nn.encoders.persistence_image import PIEncoder
from cloudforger.nn.heads.paramest import ParameterEstimator
from cloudforger.nn.models.single_modal import SingleModalModel
from cloudforger.nn.train import train_one_epoch, evaluate, evaluate_per_target

DATA_DIR = ROOT / "data" / "params" / "2d" / "thomas"
RESULTS_DIR = Path(__file__).resolve().parent / "results_k5" / "pi_multik"

K_VALUES = [5, 10, 15]  # DTM k's to stack as channels; override before calling main()/run_one_seed for other sweeps

SEEDS = [9371, 9372, 9373]
N_EPOCHS = 500
BATCH_SIZE = 32  # matches fusion/betti_cnn/pi's batch size (not vihrs's 100 -- see fusion_model.py's own note)
LR = 0.001
EMBEDDING_DIM = 64


# ══════════════════════════════════════════════════════════════════════════════
# Plain (non-log) zscore -- persistence entropy can be exactly 0, and
# persistence-image pixels can be 0 for an all-empty region, where
# run_vihrs.fit_log_zscore's log() would be undefined. Same fit/apply-frozen
# convention as fusion_model.py's own fit_zscore/apply_zscore.
# ══════════════════════════════════════════════════════════════════════════════

def fit_zscore(values: np.ndarray) -> dict[str, float]:
    std = float(values.std())
    return {"mean": float(values.mean()), "std": std if std else 1.0}


def apply_zscore(values: np.ndarray, norm: dict[str, float]) -> np.ndarray:
    return (values - norm["mean"]) / norm["std"]


# ══════════════════════════════════════════════════════════════════════════════
# Data: load images at every k, intersect by seed, stack channels
# ══════════════════════════════════════════════════════════════════════════════

def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _intersect_seeds(seed_arrays: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """Each k's images_dtm_k<k>.pkl survives compute_features.py's DTM
    empty-diagram filter with a different subset of seeds (see this file's
    module docstring). Returns (idx_per_k, common_seeds) so that
    idx_per_k[j] indexes INTO the j-th input array at the same common
    cloud, for every entry, in seed_arrays[0]'s own relative order."""
    common = set(int(s) for s in seed_arrays[0].tolist())
    for arr in seed_arrays[1:]:
        common &= set(int(s) for s in arr.tolist())

    common_seeds = np.array([int(s) for s in seed_arrays[0].tolist() if int(s) in common], dtype=np.int64)

    idx_per_k = []
    for arr in seed_arrays:
        pos = {int(s): i for i, s in enumerate(arr)}
        idx_per_k.append(np.array([pos[s] for s in common_seeds], dtype=np.int64))
    return idx_per_k, common_seeds


def load_multik_split(k_values: list[int], data_dir: Path, tag: str) -> dict[str, Any] | None:
    prefix = "adversarial_" if tag == "adversarial" else ""

    payloads = []
    for k in k_values:
        path = data_dir / f"{prefix}images_dtm_k{k}.pkl"
        if not path.exists():
            print(f"  [{tag}] missing {path} -- skipping split.")
            return None
        payloads.append(_load_pickle(path))

    seed_arrays = [np.asarray(p["seeds"]) for p in payloads]
    idx_per_k, common_seeds = _intersect_seeds(seed_arrays)
    per_k_totals = {k: len(s) for k, s in zip(k_values, seed_arrays)}
    print(f"  [{tag}] {len(common_seeds)} clouds common to all k in {k_values} (per-k totals: {per_k_totals}).")

    # Select the real target columns by name, same reasoning as
    # fusion_model.py's own col_idx lookup: label_names carries a spurious
    # "edge_buffer" column alongside the real targets.
    tda_label_names = list(payloads[0]["label_names"])
    missing = [name for name in vihrs.LABEL_NAMES if name not in tda_label_names]
    if missing:
        raise KeyError(f"[{tag}] label_names {tda_label_names} is missing {missing} from vihrs's LABEL_NAMES")
    col_idx = [tda_label_names.index(name) for name in vihrs.LABEL_NAMES]

    targets = np.asarray(payloads[0]["labels"], dtype=float)[np.ix_(idx_per_k[0], col_idx)]
    for k, payload, idx in zip(k_values[1:], payloads[1:], idx_per_k[1:]):
        targets_k = np.asarray(payload["labels"], dtype=float)[np.ix_(idx, col_idx)]
        assert np.allclose(targets, targets_k), (
            f"[{tag}] target mismatch between k={k_values[0]} and k={k} after seed alignment -- alignment bug."
        )

    # Channel order: for k in k_values, [pi0(k), pi1(k)] -- see module docstring.
    pi_channels: list[np.ndarray] = []
    entropy_cols: dict[str, np.ndarray] = {}
    for k, payload, idx in zip(k_values, payloads, idx_per_k):
        pi_channels.append(np.asarray(payload["image_tensors"][0], dtype=np.float64)[idx])
        pi_channels.append(np.asarray(payload["image_tensors"][1], dtype=np.float64)[idx])
        entropy_cols[f"entropy0_k{k}"] = np.asarray(payload["persistence_entropy"][0], dtype=np.float64)[idx]
        entropy_cols[f"entropy1_k{k}"] = np.asarray(payload["persistence_entropy"][1], dtype=np.float64)[idx]

    # n(x) doesn't depend on k (same underlying cloud) -- joined once from
    # the sibling clouds.pkl by seed, same convention as
    # cloudforger.nn.experiments.base.n_points_head_extra.
    clouds_path = data_dir / f"{prefix}clouds.pkl"
    clouds = _load_pickle(clouds_path)
    n_points_by_seed = {int(c["seed"]): c["n_points"] for c in clouds}
    n_points = np.array([n_points_by_seed[int(s)] for s in common_seeds], dtype=np.float64)

    return {
        "pi_channels": pi_channels,
        "entropy_cols": entropy_cols,
        "n_points": n_points,
        "targets": targets,
        "seeds": common_seeds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tensor assembly: per-channel z-score, fit on train_split / frozen for adv
# ══════════════════════════════════════════════════════════════════════════════

def _build_pi_tensor(
    split: dict[str, Any], channel_norms: list[dict[str, float]] | None
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """(N, 2*len(K_VALUES), H, W) float32 tensor, one independent z-score per
    channel -- PI pixels are raw Gaussian-kernel sums with a huge dynamic
    range (see cloudforger.nn.experiments.persistence_image.py's own
    comment), and different k's have different raw magnitude scales on top
    of that, so per-channel (not global) normalization matters here more
    than it would for a single k."""
    fit = channel_norms is None
    if fit:
        channel_norms = []
    channels = []
    for i, raw in enumerate(split["pi_channels"]):
        norm = fit_zscore(raw) if fit else channel_norms[i]
        if fit:
            channel_norms.append(norm)
        channels.append(apply_zscore(raw, norm))
    return np.stack(channels, axis=1).astype(np.float32), channel_norms


def _build_extra(
    split: dict[str, Any], n_norm: dict, entropy_norms: dict[str, dict] | None, k_values: list[int]
) -> tuple[np.ndarray, dict[str, dict]]:
    fit = entropy_norms is None
    if fit:
        entropy_norms = {}
    n_std = vihrs.apply_log_zscore(split["n_points"], n_norm).astype(np.float32)
    cols = [n_std]
    for k in k_values:
        for dim in (0, 1):
            name = f"entropy{dim}_k{k}"
            values = split["entropy_cols"][name]
            norm = fit_zscore(values) if fit else entropy_norms[name]
            if fit:
                entropy_norms[name] = norm
            cols.append(apply_zscore(values, norm).astype(np.float32))
    return np.stack(cols, axis=1), entropy_norms


# ══════════════════════════════════════════════════════════════════════════════
# Train / eval for one seed
# ══════════════════════════════════════════════════════════════════════════════

def run_one_seed(seed: int, train_split: dict[str, Any], adv_split: dict[str, Any] | None) -> dict[str, Any]:
    output_dir = RESULTS_DIR / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    k_values = K_VALUES
    n = len(train_split["targets"])
    label_norm = vihrs.fit_log_zscore(train_split["targets"])
    n_norm = vihrs.fit_log_zscore(train_split["n_points"])

    targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)
    pi_img, channel_norms = _build_pi_tensor(train_split, channel_norms=None)
    extra, entropy_norms = _build_extra(train_split, n_norm, entropy_norms=None, k_values=k_values)

    # Single shared dataset sliced via Subset, not fancy-indexed copies --
    # same rationale as fusion_model.py's own full_dataset (avoids doubling
    # the PI tensor's memory footprint).
    full_dataset = TensorDataset(
        torch.from_numpy(pi_img), torch.from_numpy(extra), torch.from_numpy(targets_std),
    )
    train_idx, val_idx, test_idx = vihrs.train_val_test_indices(n, seed)

    def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(Subset(full_dataset, idx), batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = _loader(train_idx, True)
    val_loader = _loader(val_idx, False)
    test_loader = _loader(test_idx, False)

    # SingleModalModel(encoder, head).forward(x, covariates) lines up exactly
    # with cloudforger.nn.train's 3-tuple batch convention (inputs,
    # covariates, labels), so train_one_epoch/evaluate/evaluate_per_target
    # can be reused as-is instead of hand-rolling a training loop.
    model = SingleModalModel(
        encoder=PIEncoder(in_channels=pi_img.shape[1], embedding_dim=EMBEDDING_DIM),
        head=ParameterEstimator(embedding_dim=EMBEDDING_DIM + extra.shape[1], n_params=len(vihrs.LABEL_NAMES)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss, best_state = float("inf"), None

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _ = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0 or epoch == N_EPOCHS:
            print(f"[pi_multik seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

    model.load_state_dict(best_state)
    test_loss, _ = evaluate(model, test_loader, loss_fn, device)
    test_loss_per_target = dict(zip(vihrs.LABEL_NAMES, evaluate_per_target(model, test_loader, device).tolist()))
    print(f"\n[pi_multik seed={seed}] test loss {test_loss:.4f}")

    adversarial_loss = None
    adversarial_loss_per_target = None
    if adv_split is not None:
        adv_targets_std = vihrs.apply_log_zscore(adv_split["targets"], label_norm).astype(np.float32)
        adv_pi_img, _ = _build_pi_tensor(adv_split, channel_norms=channel_norms)
        adv_extra, _ = _build_extra(adv_split, n_norm, entropy_norms=entropy_norms, k_values=k_values)
        adv_ds = TensorDataset(
            torch.from_numpy(adv_pi_img), torch.from_numpy(adv_extra), torch.from_numpy(adv_targets_std),
        )
        adv_loader = DataLoader(adv_ds, batch_size=BATCH_SIZE, shuffle=False)
        adversarial_loss, _ = evaluate(model, adv_loader, loss_fn, device)
        adversarial_loss_per_target = dict(
            zip(vihrs.LABEL_NAMES, evaluate_per_target(model, adv_loader, device).tolist())
        )
        print(f"[pi_multik seed={seed}] adversarial loss {adversarial_loss:.4f}")

    cfg = {
        "task": "params", "process": "thomas", "method": "pi_multik",
        "k_values": k_values, "channels": [f"k{k}_h{d}" for k in k_values for d in (0, 1)],
        "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS, "lr": LR,
        "embedding_dim": EMBEDDING_DIM, "seed": seed,
    }
    torch.save(
        {
            "model_state": best_state, "history": history, "config": cfg, "test_loss": test_loss,
            "test_loss_per_target": test_loss_per_target,
            "label_names": list(vihrs.LABEL_NAMES), "label_norm": label_norm,
            "label_log_mean": label_norm["mean"], "label_log_std": label_norm["std"],
            "label_transforms": ["log"] * len(vihrs.LABEL_NAMES),
            "adversarial_loss": adversarial_loss,
            "adversarial_loss_per_target": adversarial_loss_per_target,
        },
        output_dir / "results.pt",
    )
    torch.save(model.cpu(), output_dir / "model.pt")

    import json
    json_payload: dict[str, Any] = {
        "task": "params", "method": "pi_multik", "test_loss": test_loss,
        "test_loss_per_target": test_loss_per_target, "seed": seed,
    }
    if adversarial_loss is not None:
        json_payload["adversarial_loss"] = adversarial_loss
        json_payload["adversarial_loss_per_target"] = adversarial_loss_per_target
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    return {"seed": seed, "test_loss": test_loss, "adversarial_loss": adversarial_loss}


def main() -> None:
    print(f"Loading persistence images for k in {K_VALUES} (K_VALUES) ...")
    train_split = load_multik_split(K_VALUES, DATA_DIR, tag="train_test")
    if train_split is None:
        raise FileNotFoundError(
            f"images_dtm_k<k>.pkl missing for some k in {K_VALUES} under {DATA_DIR} -- "
            "run compute_features.py --k <k> for each first."
        )
    adv_split = load_multik_split(K_VALUES, DATA_DIR, tag="adversarial")

    for seed in SEEDS:
        output_dir = RESULTS_DIR / f"seed_{seed}"
        if (output_dir / "results.pt").exists():
            print(f"\npi_multik | seed {seed}: already done, skipping.")
            continue
        print(f"\n{'#' * 90}\n### pi_multik | seed {seed}\n{'#' * 90}")
        run_one_seed(seed, train_split, adv_split)

    print(f"\nAll seeds done. Results under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
