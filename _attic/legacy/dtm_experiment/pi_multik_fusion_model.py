# dtm_experiment/pi_multik_fusion_model.py
"""
Fusion of vihrs's L(r)-r + n(x) branch with pi_multik's k=5/10/15 stacked
persistence-image branch, late-fused by concatenating both branch
embeddings before a shared regression head -- same "concatenate before the
head" pattern fusion_model.py's own (lr + pi + betti) FusionCNN uses, minus
the betti branch. Motivated by vihrs (checkpointed -- see
vihrs_checkpointed.py) reaching a lower best-val-loss than pi_multik on its
own: if L(r)-r genuinely carries information pi_multik's images don't,
fusing the two branches should do at least as well as the better of the two
alone, not just average them out.

Reuses, rather than reimplements:
  - pi_multik_model.load_multik_split for the image side, so it inherits
    that function's k-dependent seed-intersection handling (see that file's
    module docstring for why the intersection is needed at all).
  - fusion_model._align_tda_to_lr for the join against vihrs's L(r)-r
    population (every cloud in clouds.pkl, never DTM-filtered) -- the same
    join fusion_model.py itself needs, for the same reason (different,
    filtered population sizes).
  - pi_multik_model._build_pi_tensor / _build_extra for per-channel image
    z-scoring and the n(x)+entropy scalar side-channel -- both operate on
    any dict with "pi_channels"/"entropy_cols"/"n_points" keys, which is
    exactly the schema load_fusion_split below returns.

Assumes data/params/2d/<process>/clouds.pkl (+ adversarial_clouds.pkl) and
images_dtm_k<k>.pkl for each k in pi_multik_model.K_VALUES already exist.
"""

from __future__ import annotations

import json
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

import fusion_model
import pi_multik_model

DATA_DIR = ROOT / "data" / "params" / "2d" / "thomas"
RESULTS_DIR = Path(__file__).resolve().parent / "results_k5" / "pi_multik_fusion"

SEEDS = [9371, 9372, 9373]
N_EPOCHS = 500
BATCH_SIZE = 32
LR = 0.001
EMBEDDING_DIM = 64


# ══════════════════════════════════════════════════════════════════════════════
# Data: pi_multik's split, joined against vihrs's L(r)-r population by seed
# ══════════════════════════════════════════════════════════════════════════════

def load_fusion_split(
    k_values: list[int], data_dir: Path, lr_features: dict[str, np.ndarray], tag: str
) -> dict[str, Any] | None:
    multik_split = pi_multik_model.load_multik_split(k_values, data_dir, tag=tag)
    if multik_split is None:
        return None

    multik_idx, lr_idx = fusion_model._align_tda_to_lr(multik_split["seeds"], lr_features["cloud_seeds"])
    print(f"  [{tag}] {len(multik_idx)}/{len(multik_split['seeds'])} pi_multik clouds matched to L(r)-r clouds.")

    targets_multik = multik_split["targets"][multik_idx]
    targets_lr = lr_features["targets"][lr_idx]
    assert np.allclose(targets_multik, targets_lr), f"[{tag}] target mismatch after seed alignment -- alignment bug."

    return {
        "pi_channels": [ch[multik_idx] for ch in multik_split["pi_channels"]],
        "entropy_cols": {name: values[multik_idx] for name, values in multik_split["entropy_cols"].items()},
        "n_points": multik_split["n_points"][multik_idx],
        "targets": targets_multik,
        "lr_seq": lr_features["l_minus_r"][lr_idx],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model: two branches (L(r)-r, stacked multi-k PI), late-fused, plus scalar extras
# ══════════════════════════════════════════════════════════════════════════════

class VihrsPIMultiKFusion(nn.Module):
    def __init__(self, lr_seq_len: int, in_channels: int, embedding_dim: int, n_extra: int, n_targets: int):
        super().__init__()
        # L(r)-r branch -- identical architecture to VihrsCNN's/fusion_model
        # FusionCNN's own conv stack.
        self.lr_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7), nn.ReLU(), nn.MaxPool1d(5),
            nn.Conv1d(64, 64, kernel_size=7), nn.ReLU(),
        )
        with torch.no_grad():
            lr_flat_dim = self.lr_conv(torch.zeros(1, 1, lr_seq_len)).flatten(1).shape[1]
        self.lr_head = nn.Linear(lr_flat_dim, embedding_dim)

        # Multi-k persistence-image branch -- identical to pi_multik_model's own.
        self.pi_encoder = PIEncoder(in_channels=in_channels, embedding_dim=embedding_dim)

        merge_dim = 2 * embedding_dim + n_extra
        self.head = nn.Sequential(
            nn.Linear(merge_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, n_targets),
        )

    def forward(self, lr_seq: torch.Tensor, pi_img: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        lr_emb = self.lr_head(self.lr_conv(lr_seq.unsqueeze(1)).flatten(start_dim=1))
        pi_emb = self.pi_encoder(pi_img)
        merged = torch.cat([lr_emb, pi_emb, extra], dim=1)
        return self.head(merged)


# ══════════════════════════════════════════════════════════════════════════════
# Train / eval loops (3 model inputs, doesn't fit cloudforger.nn.train's
# 2-or-3-tuple batch convention, so hand-rolled -- same reason
# fusion_model.py's own _train_one_epoch/_evaluate_loss are)
# ══════════════════════════════════════════════════════════════════════════════

def _train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total, count = 0.0, 0
    for lr_seq, pi_img, extra, y in loader:
        lr_seq, pi_img, extra, y = (t.to(device) for t in (lr_seq, pi_img, extra, y))
        optimizer.zero_grad()
        pred = model(lr_seq, pi_img, extra)
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
    for lr_seq, pi_img, extra, y in loader:
        lr_seq, pi_img, extra, y = (t.to(device) for t in (lr_seq, pi_img, extra, y))
        pred = model(lr_seq, pi_img, extra)
        loss = loss_fn(pred, y)
        total += loss.item() * len(y)
        count += len(y)
    return total / count


@torch.no_grad()
def _evaluate_per_target(model, loader, device) -> np.ndarray:
    model.eval()
    total_sq_err, count = None, 0
    for lr_seq, pi_img, extra, y in loader:
        lr_seq, pi_img, extra, y = (t.to(device) for t in (lr_seq, pi_img, extra, y))
        pred = model(lr_seq, pi_img, extra)
        sq_err = (pred - y).pow(2).sum(dim=0)
        total_sq_err = sq_err if total_sq_err is None else total_sq_err + sq_err
        count += len(y)
    return (total_sq_err / count).cpu().numpy()


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

    k_values = pi_multik_model.K_VALUES
    n = len(train_split["targets"])
    label_norm = vihrs.fit_log_zscore(train_split["targets"])
    n_norm = vihrs.fit_log_zscore(train_split["n_points"])
    lr_norm = vihrs.fit_zscore_global(train_split["lr_seq"])

    targets_std = vihrs.apply_log_zscore(train_split["targets"], label_norm).astype(np.float32)
    lr_seq = vihrs.apply_zscore_global(train_split["lr_seq"], lr_norm).astype(np.float32)
    pi_img, channel_norms = pi_multik_model._build_pi_tensor(train_split, channel_norms=None)
    extra, entropy_norms = pi_multik_model._build_extra(train_split, n_norm, entropy_norms=None, k_values=k_values)

    full_dataset = TensorDataset(
        torch.from_numpy(lr_seq), torch.from_numpy(pi_img), torch.from_numpy(extra), torch.from_numpy(targets_std),
    )
    train_idx, val_idx, test_idx = vihrs.train_val_test_indices(n, seed)

    def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(Subset(full_dataset, idx), batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = _loader(train_idx, True)
    val_loader = _loader(val_idx, False)
    test_loader = _loader(test_idx, False)

    model = VihrsPIMultiKFusion(
        lr_seq_len=lr_seq.shape[1], in_channels=pi_img.shape[1], embedding_dim=EMBEDDING_DIM,
        n_extra=extra.shape[1], n_targets=len(vihrs.LABEL_NAMES),
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
            print(f"[pi_multik_fusion seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

    model.load_state_dict(best_state)
    test_loss = _evaluate_loss(model, test_loader, loss_fn, device)
    test_loss_per_target = dict(zip(vihrs.LABEL_NAMES, _evaluate_per_target(model, test_loader, device).tolist()))
    print(f"\n[pi_multik_fusion seed={seed}] test loss {test_loss:.4f}")

    adversarial_loss = None
    adversarial_loss_per_target = None
    if adv_split is not None:
        adv_targets_std = vihrs.apply_log_zscore(adv_split["targets"], label_norm).astype(np.float32)
        adv_lr_seq = vihrs.apply_zscore_global(adv_split["lr_seq"], lr_norm).astype(np.float32)
        adv_pi_img, _ = pi_multik_model._build_pi_tensor(adv_split, channel_norms=channel_norms)
        adv_extra, _ = pi_multik_model._build_extra(adv_split, n_norm, entropy_norms=entropy_norms, k_values=k_values)
        adv_ds = TensorDataset(
            torch.from_numpy(adv_lr_seq), torch.from_numpy(adv_pi_img),
            torch.from_numpy(adv_extra), torch.from_numpy(adv_targets_std),
        )
        adv_loader = DataLoader(adv_ds, batch_size=BATCH_SIZE, shuffle=False)
        adversarial_loss = _evaluate_loss(model, adv_loader, loss_fn, device)
        adversarial_loss_per_target = dict(
            zip(vihrs.LABEL_NAMES, _evaluate_per_target(model, adv_loader, device).tolist())
        )
        print(f"[pi_multik_fusion seed={seed}] adversarial loss {adversarial_loss:.4f}")

    cfg = {
        "task": "params", "process": "thomas", "method": "pi_multik_fusion",
        "k_values": k_values, "channels": ["l_minus_r", "n_points"] + [f"k{k}_h{d}" for k in k_values for d in (0, 1)],
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

    json_payload: dict[str, Any] = {
        "task": "params", "method": "pi_multik_fusion", "test_loss": test_loss,
        "test_loss_per_target": test_loss_per_target, "seed": seed,
    }
    if adversarial_loss is not None:
        json_payload["adversarial_loss"] = adversarial_loss
        json_payload["adversarial_loss_per_target"] = adversarial_loss_per_target
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    return {"seed": seed, "test_loss": test_loss, "adversarial_loss": adversarial_loss}


def main() -> None:
    print(f"Loading L(r)-r features for the fusion's vihrs side ...")
    lr_data = vihrs.prepare_data(DATA_DIR / "clouds.pkl", DATA_DIR / "adversarial_clouds.pkl")

    print(f"\nAligning train_test pi_multik (k={pi_multik_model.K_VALUES}) features to L(r)-r by seed ...")
    train_split = load_fusion_split(pi_multik_model.K_VALUES, DATA_DIR, lr_data["train_features"], tag="train_test")
    if train_split is None:
        raise FileNotFoundError(
            f"images_dtm_k<k>.pkl missing for some k in {pi_multik_model.K_VALUES} under {DATA_DIR} -- "
            "run compute_features.py --k <k> for each first."
        )

    adv_split = None
    if lr_data["adversarial_features"] is not None:
        print("Aligning adversarial pi_multik features to L(r)-r by seed ...")
        adv_split = load_fusion_split(
            pi_multik_model.K_VALUES, DATA_DIR, lr_data["adversarial_features"], tag="adversarial",
        )

    for seed in SEEDS:
        output_dir = RESULTS_DIR / f"seed_{seed}"
        if (output_dir / "results.pt").exists():
            print(f"\npi_multik_fusion | seed {seed}: already done, skipping.")
            continue
        print(f"\n{'#' * 90}\n### pi_multik_fusion | seed {seed}\n{'#' * 90}")
        run_one_seed(seed, train_split, adv_split)

    print(f"\nAll seeds done. Results under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
