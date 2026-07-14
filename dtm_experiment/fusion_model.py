# dtm_experiment/fusion_model.py
"""
L(r)-r + TDA fusion: feeds new_feature's L(r)-r/n(x) representation and
combined_all's topological representation (pi_0+pi_1 stacked image,
betti_0+betti_1 concatenated curve, n(x), persistence entropy) into ONE
model, late-fused by concatenating all branch embeddings before a shared
regression head -- same "concatenate before the head" pattern combined_all
and MultiModalModel already use.

Not built on cloudforger.nn.experiments.Experiment: new_feature itself lives
outside that framework (its own model/training loop, see
scripts/runners/params/run_new_feature.py), and fusing it with the TDA
features needs an explicit seed-based join the Experiment framework has no
hook for -- L(r)-r covers every cloud in clouds.pkl, but the TDA features
only cover the subset that survived the DTM empty-diagram filter in
compute_features.py (~28,541 of 28,730 for k=5), so the two populations are
different sizes and must be intersected by seed, not just index-aligned.

Assumes, for the K below:
    data/params/2d/thomas/images_dtm_k<K>.pkl (+ adversarial_)
    data/params/2d/thomas/betti_dtm_k<K>.pkl (+ adversarial_)
    data/params/2d/thomas/clouds.pkl (+ adversarial_clouds.pkl)
already exist (the first two from compute_features.py, matching whichever K
that was last run with).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "runners" / "params"))

import run_new_feature as nf
from cloudforger.nn.encoders.persistence_image import PIEncoder
from cloudforger.nn.encoders.sequence_cnn import SequenceCNNEncoder

DATA_DIR = ROOT / "data" / "params" / "2d" / "thomas"
RESULTS_DIR = Path(__file__).resolve().parent / "results_k5" / "fusion"

K = 5  # which DTM k's features to fuse against; matches whichever compute_features.py run you're using

SEEDS = [9371, 9372, 9373]
N_EPOCHS = 500
BATCH_SIZE = 32  # matches betti_cnn/pi/combined_all's batch size, not new_feature's 100 -- see run_one_seed docstring
LR = 0.001
EMBEDDING_DIM = 64


# ══════════════════════════════════════════════════════════════════════════════
# Plain (non-log) zscore -- persistence entropy can be exactly 0, where
# run_new_feature.fit_log_zscore's log() would be undefined. Mirrors
# zscore_fit_once in src/cloudforger/nn/experiments/base.py, but as an
# explicit fit/apply pair (this file has no persistent "experiment" object
# to hang fit-once state off of, matching run_new_feature.py's own
# fit_log_zscore/apply_log_zscore convention instead).
# ══════════════════════════════════════════════════════════════════════════════

def fit_zscore(values: np.ndarray) -> dict[str, float]:
    std = float(values.std())
    return {"mean": float(values.mean()), "std": std if std else 1.0}


def apply_zscore(values: np.ndarray, norm: dict[str, float]) -> np.ndarray:
    return (values - norm["mean"]) / norm["std"]


# ══════════════════════════════════════════════════════════════════════════════
# Data: load TDA (images + betti) and L(r)-r separately, align by seed
# ══════════════════════════════════════════════════════════════════════════════

def _load_pickle(path: Path) -> dict[str, Any]:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def _align_tda_to_lr(tda_seeds: np.ndarray, lr_seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """tda_seeds is the DTM-filtered (smaller) population; lr_seeds is
    L(r)-r's population (every cloud in clouds.pkl, never filtered). Returns
    (tda_idx, lr_idx) so tda_seeds[tda_idx[i]] == lr_seeds[lr_idx[i]] for
    every i, in tda_seeds's own order, restricted to the intersection."""
    lr_pos = {int(s): i for i, s in enumerate(lr_seeds)}
    tda_idx, lr_idx = [], []
    for i, s in enumerate(tda_seeds):
        s = int(s)
        if s in lr_pos:
            tda_idx.append(i)
            lr_idx.append(lr_pos[s])
    if len(tda_idx) != len(tda_seeds):
        missing = len(tda_seeds) - len(tda_idx)
        print(f"  ! {missing}/{len(tda_seeds)} TDA seeds have no matching L(r)-r cloud -- dropping them.")
    return np.asarray(tda_idx, dtype=np.int64), np.asarray(lr_idx, dtype=np.int64)


def load_fusion_split(images_path: Path, betti_path: Path, lr_features: dict[str, np.ndarray], tag: str) -> dict[str, np.ndarray] | None:
    if not images_path.exists() or not betti_path.exists():
        print(f"  [{tag}] missing {images_path if not images_path.exists() else betti_path} -- skipping split.")
        return None

    images = _load_pickle(images_path)
    betti = _load_pickle(betti_path)

    assert list(images["seeds"]) == list(betti["seeds"]), (
        f"[{tag}] images/betti seed order mismatch -- both should come from the same "
        "compute_features.py::compute_for_split call and share identical 'seeds' order."
    )

    # Select the 3 real parameters by name rather than assuming the payload
    # has exactly 3 columns in exactly this order -- clouds.pkl's "params"
    # dict also carries "edge_buffer" (NeymanScottProcess.params exposes it
    # for simulation bookkeeping; it's deterministically 4*cluster_scale, not
    # a free parameter), which leaks into images["labels"]/["label_names"] as
    # a spurious 4th column via pipeline_lib.records.build_labels. Indexing
    # by name is robust to that regardless of how many extra columns show up
    # or in what order.
    tda_label_names = list(images["label_names"])
    missing = [name for name in nf.LABEL_NAMES if name not in tda_label_names]
    if missing:
        raise KeyError(f"[{tag}] TDA label_names {tda_label_names} is missing {missing} from new_feature's LABEL_NAMES")
    col_idx = [tda_label_names.index(name) for name in nf.LABEL_NAMES]

    tda_idx, lr_idx = _align_tda_to_lr(np.asarray(images["seeds"]), lr_features["cloud_seeds"])
    print(f"  [{tag}] {len(tda_idx)}/{len(images['seeds'])} TDA clouds matched to L(r)-r clouds.")

    # Sanity check: independently-loaded targets must agree at the aligned indices.
    tda_targets = np.asarray(images["labels"], dtype=float)[np.ix_(tda_idx, col_idx)]
    lr_targets = lr_features["targets"][lr_idx]
    assert np.allclose(tda_targets, lr_targets), f"[{tag}] target mismatch after seed alignment -- alignment bug."

    pi0 = np.asarray(images["image_tensors"][0], dtype=np.float64)[tda_idx]
    pi1 = np.asarray(images["image_tensors"][1], dtype=np.float64)[tda_idx]
    b0 = np.asarray(betti["betti0_matrix"], dtype=np.float64)[tda_idx]
    b1 = np.asarray(betti["betti1_matrix"], dtype=np.float64)[tda_idx]
    entropy0 = np.asarray(images["persistence_entropy"][0], dtype=np.float64)[tda_idx]
    entropy1 = np.asarray(images["persistence_entropy"][1], dtype=np.float64)[tda_idx]

    return {
        "pi0": pi0, "pi1": pi1, "b0": b0, "b1": b1,
        "entropy0": entropy0, "entropy1": entropy1,
        "lr_seq": lr_features["l_minus_r"][lr_idx],
        "n_points": lr_features["n_points"][lr_idx],
        "targets": tda_targets,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model: three branches (L(r)-r, PI, Betti), late-fused, plus scalar extras
# ══════════════════════════════════════════════════════════════════════════════

class FusionCNN(nn.Module):
    def __init__(self, lr_seq_len: int, betti_seq_len: int, embedding_dim: int, n_extra: int, n_targets: int):
        super().__init__()
        # L(r)-r branch -- identical architecture to VihrsCNN's conv stack.
        self.lr_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7), nn.ReLU(),
        )
        with torch.no_grad():
            lr_flat_dim = self.lr_conv(torch.zeros(1, 1, lr_seq_len)).flatten(1).shape[1]
        self.lr_head = nn.Linear(lr_flat_dim, embedding_dim)

        # TDA branches -- identical architecture/setup to combined_all's.
        self.pi_encoder = PIEncoder(in_channels=2, embedding_dim=embedding_dim)
        self.betti_encoder = SequenceCNNEncoder(input_dim=betti_seq_len, embedding_dim=embedding_dim, pool_size=5)

        merge_dim = 3 * embedding_dim + n_extra
        self.head = nn.Sequential(
            nn.Linear(merge_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, n_targets),
        )

    def forward(self, lr_seq: torch.Tensor, pi_img: torch.Tensor, betti_seq: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        lr_emb = self.lr_head(self.lr_conv(lr_seq.unsqueeze(1)).flatten(start_dim=1))
        pi_emb = self.pi_encoder(pi_img)
        betti_emb = self.betti_encoder(betti_seq)
        merged = torch.cat([lr_emb, pi_emb, betti_emb, extra], dim=1)
        return self.head(merged)


# ══════════════════════════════════════════════════════════════════════════════
# Train / eval loops (mirrors run_new_feature.py's, adapted for 4 inputs)
# ══════════════════════════════════════════════════════════════════════════════

def _train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total, count = 0.0, 0
    for lr_seq, pi_img, betti_seq, extra, y in loader:
        lr_seq, pi_img, betti_seq, extra, y = (t.to(device) for t in (lr_seq, pi_img, betti_seq, extra, y))
        optimizer.zero_grad()
        pred = model(lr_seq, pi_img, betti_seq, extra)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


@torch.no_grad()
def _evaluate_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for lr_seq, pi_img, betti_seq, extra, y in loader:
        lr_seq, pi_img, betti_seq, extra, y = (t.to(device) for t in (lr_seq, pi_img, betti_seq, extra, y))
        pred = model(lr_seq, pi_img, betti_seq, extra)
        loss = loss_fn(pred, y)
        total += loss.item() * len(y)
        count += len(y)
    return total / count


def _build_extra(split: dict[str, np.ndarray], n_norm: dict, entropy0_norm: dict, entropy1_norm: dict) -> np.ndarray:
    n_std = nf.apply_log_zscore(split["n_points"], n_norm).astype(np.float32)
    e0_std = apply_zscore(split["entropy0"], entropy0_norm).astype(np.float32)
    e1_std = apply_zscore(split["entropy1"], entropy1_norm).astype(np.float32)
    return np.stack([n_std, e0_std, e1_std], axis=1)


def run_one_seed(seed: int, train_split: dict[str, np.ndarray], adv_split: dict[str, np.ndarray] | None) -> dict[str, Any]:
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

    n = len(train_split["targets"])
    label_norm = nf.fit_log_zscore(train_split["targets"])
    n_norm = nf.fit_log_zscore(train_split["n_points"])
    entropy0_norm = fit_zscore(train_split["entropy0"])
    entropy1_norm = fit_zscore(train_split["entropy1"])

    targets_std = nf.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)
    lr_seq = train_split["lr_seq"].astype(np.float32)
    pi_img = np.stack([train_split["pi0"], train_split["pi1"]], axis=1).astype(np.float32)
    betti_seq = np.concatenate([train_split["b0"], train_split["b1"]], axis=1).astype(np.float32)
    extra = _build_extra(train_split, n_norm, entropy0_norm, entropy1_norm)

    train_idx, val_idx, test_idx = nf.train_val_test_indices(n, seed)

    def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(lr_seq[idx]), torch.from_numpy(pi_img[idx]),
            torch.from_numpy(betti_seq[idx]), torch.from_numpy(extra[idx]),
            torch.from_numpy(targets_std[idx]),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = _loader(train_idx, True)
    val_loader = _loader(val_idx, False)
    test_loader = _loader(test_idx, False)

    model = FusionCNN(
        lr_seq_len=lr_seq.shape[1], betti_seq_len=betti_seq.shape[1],
        embedding_dim=EMBEDDING_DIM, n_extra=extra.shape[1], n_targets=len(nf.LABEL_NAMES),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss, best_state = float("inf"), None

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = _evaluate_loss(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0 or epoch == N_EPOCHS:
            print(f"[fusion seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

    model.load_state_dict(best_state)
    test_loss = _evaluate_loss(model, test_loader, loss_fn, device)
    print(f"\n[fusion seed={seed}] test loss {test_loss:.4f}")

    adversarial_loss = None
    if adv_split is not None:
        adv_targets_std = nf.apply_log_zscore(adv_split["targets"], label_norm).astype(np.float32)
        adv_lr_seq = adv_split["lr_seq"].astype(np.float32)
        adv_pi_img = np.stack([adv_split["pi0"], adv_split["pi1"]], axis=1).astype(np.float32)
        adv_betti_seq = np.concatenate([adv_split["b0"], adv_split["b1"]], axis=1).astype(np.float32)
        adv_extra = _build_extra(adv_split, n_norm, entropy0_norm, entropy1_norm)
        adv_ds = TensorDataset(
            torch.from_numpy(adv_lr_seq), torch.from_numpy(adv_pi_img),
            torch.from_numpy(adv_betti_seq), torch.from_numpy(adv_extra),
            torch.from_numpy(adv_targets_std),
        )
        adv_loader = DataLoader(adv_ds, batch_size=BATCH_SIZE, shuffle=False)
        adversarial_loss = _evaluate_loss(model, adv_loader, loss_fn, device)
        print(f"[fusion seed={seed}] adversarial loss {adversarial_loss:.4f}")

    cfg = {
        "task": "params", "process": "thomas", "method": "fusion",
        "channels": ["l_minus_r", "n_points", "pi_0", "pi_1", "betti_0", "betti_1", "entropy_0", "entropy_1"],
        "k": K, "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS, "lr": LR,
        "embedding_dim": EMBEDDING_DIM, "seed": seed,
    }
    torch.save(
        {
            "model_state": best_state, "history": history, "config": cfg, "test_loss": test_loss,
            "label_names": list(nf.LABEL_NAMES), "label_norm": label_norm,
            "label_log_mean": label_norm["mean"], "label_log_std": label_norm["std"],
            "label_transforms": ["log"] * len(nf.LABEL_NAMES),
            "adversarial_loss": adversarial_loss,
        },
        output_dir / "results.pt",
    )
    torch.save(model.cpu(), output_dir / "model.pt")

    import json
    json_payload: dict[str, Any] = {"task": "params", "method": "fusion", "test_loss": test_loss, "seed": seed}
    if adversarial_loss is not None:
        json_payload["adversarial_loss"] = adversarial_loss
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    return {"seed": seed, "test_loss": test_loss, "adversarial_loss": adversarial_loss}


def main() -> None:
    print(f"Loading L(r)-r features (K={K} for the TDA side) ...")
    lr_data = nf.prepare_data(DATA_DIR / "clouds.pkl", DATA_DIR / "adversarial_clouds.pkl")

    images_path = DATA_DIR / f"images_dtm_k{K}.pkl"
    betti_path = DATA_DIR / f"betti_dtm_k{K}.pkl"
    adv_images_path = DATA_DIR / f"adversarial_images_dtm_k{K}.pkl"
    adv_betti_path = DATA_DIR / f"adversarial_betti_dtm_k{K}.pkl"

    print("\nAligning train_test TDA features to L(r)-r by seed ...")
    train_split = load_fusion_split(images_path, betti_path, lr_data["train_features"], tag="train_test")
    if train_split is None:
        raise FileNotFoundError(f"{images_path} / {betti_path} missing -- run compute_features.py first.")

    adv_split = None
    if lr_data["adversarial_features"] is not None:
        print("Aligning adversarial TDA features to L(r)-r by seed ...")
        adv_split = load_fusion_split(adv_images_path, adv_betti_path, lr_data["adversarial_features"], tag="adversarial")

    for seed in SEEDS:
        output_dir = RESULTS_DIR / f"seed_{seed}"
        if (output_dir / "results.pt").exists():
            print(f"\nfusion | seed {seed}: already done, skipping.")
            continue
        print(f"\n{'#' * 90}\n### fusion | seed {seed}\n{'#' * 90}")
        run_one_seed(seed, train_split, adv_split)

    print(f"\nAll seeds done. Results under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
