# dtm_experiment/vihrs_checkpointed.py
"""
Best-val-checkpoint variant of vihrs (scripts/runners/params/run_vihrs.py).

run_vihrs.py deliberately has NO checkpoint selection -- it's a literal
replication of Vihrs (2022)'s own recipe: train a fixed number of epochs, no
early stopping, evaluate whatever the network looks like after the LAST
epoch (see that file's module docstring). Its val_loss curve is tracked only
as a diagnostic; the lowest point on it is never turned into a saved model
or checked against the test set. That makes vihrs's "best val loss" number
incomparable to every other method in this repo, which all select the best
val checkpoint and report ITS test loss -- an honest, unpeeked-at estimate
of what that checkpoint actually generalizes to.

This file isolates that one variable. It reuses run_vihrs's data pipeline,
VihrsCNN model, and train_one_epoch/evaluate_loss/evaluate_per_target_loss
functions completely unchanged, and only replaces the training loop: track
best val loss, reload that state before the final test/adversarial
evaluation -- the same pattern cloudforger.nn.experiments.base.
_train_and_eval, fusion_model.run_one_seed, and pi_multik_model.run_one_seed
already use.

Deliberately NOT changed: batch_size=100, lr=0.001, no weight_decay, no
dropout -- every other paper-faithful choice stays as-is, so checkpoint
selection is the only thing being tested. No mincontrast comparison or
paper scatter/boxplot plots either (see run_vihrs.py for those) -- this
file exists purely to answer "what is vihrs's honest test loss if it gets
the same fair treatment every other method here gets," not to replace
run_vihrs.py as the paper-fidelity baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "runners" / "params"))

import run_vihrs as vihrs

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results_k5" / "vihrs_checkpointed"


def run_one_seed(
    seed: int,
    *,
    train_features: dict[str, np.ndarray],
    adversarial_features: dict[str, np.ndarray] | None,
    adversarial_path: Path | None,
    r_grid: np.ndarray,
    output_root: Path,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device_pref: str | None = None,
) -> dict[str, Any]:
    output_dir = output_root / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if device_pref:
        device = device_pref
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    n = len(train_features["targets"])
    train_idx, val_idx, test_idx = vihrs.train_val_test_indices(n, seed)

    # Fit stats on the train split only, same protocol fix run_vihrs.py's
    # own run_one_seed already documents/applies.
    label_norm = vihrs.fit_log_zscore(train_features["targets"][train_idx])
    n_norm = vihrs.fit_log_zscore(train_features["n_points"][train_idx])
    lr_norm = vihrs.fit_zscore_global(train_features["l_minus_r"][train_idx])

    targets_std = vihrs.apply_log_zscore(train_features["targets"], label_norm).astype(np.float32)
    n_std = vihrs.apply_log_zscore(train_features["n_points"], n_norm).astype(np.float32)
    seq = vihrs.apply_zscore_global(train_features["l_minus_r"], lr_norm).astype(np.float32)

    def _loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(seq[idx]), torch.from_numpy(n_std[idx]), torch.from_numpy(targets_std[idx]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _loader(train_idx, True)
    val_loader = _loader(val_idx, False)
    test_loader = _loader(test_idx, False)

    model = vihrs.VihrsCNN(seq_len=len(r_grid), n_targets=len(vihrs.LABEL_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # no weight_decay -- see module docstring
    loss_fn = nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss, best_state, best_epoch = float("inf"), None, 0

    for epoch in range(1, n_epochs + 1):
        train_loss = vihrs.train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = vihrs.evaluate_loss(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0 or epoch == n_epochs:
            print(f"[vihrs_ckpt seed={seed}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

    model.load_state_dict(best_state)
    print(f"[vihrs_ckpt seed={seed}] best checkpoint: epoch {best_epoch}/{n_epochs}, val_loss {best_val_loss:.4f}")

    test_loss = vihrs.evaluate_loss(model, test_loader, loss_fn, device)
    test_loss_per_target = dict(
        zip(vihrs.LABEL_NAMES, vihrs.evaluate_per_target_loss(model, test_loader, device).tolist())
    )
    print(f"[vihrs_ckpt seed={seed}] test loss {test_loss:.4f}")

    adversarial_loss = None
    adversarial_loss_per_target = None
    if adversarial_features is not None:
        adv_targets_std = vihrs.apply_log_zscore(adversarial_features["targets"], label_norm).astype(np.float32)
        adv_n_std = vihrs.apply_log_zscore(adversarial_features["n_points"], n_norm).astype(np.float32)
        adv_seq = vihrs.apply_zscore_global(adversarial_features["l_minus_r"], lr_norm).astype(np.float32)
        adv_ds = TensorDataset(
            torch.from_numpy(adv_seq), torch.from_numpy(adv_n_std), torch.from_numpy(adv_targets_std),
        )
        adv_loader = DataLoader(adv_ds, batch_size=batch_size, shuffle=False)
        adversarial_loss = vihrs.evaluate_loss(model, adv_loader, loss_fn, device)
        adversarial_loss_per_target = dict(
            zip(vihrs.LABEL_NAMES, vihrs.evaluate_per_target_loss(model, adv_loader, device).tolist())
        )
        print(f"[vihrs_ckpt seed={seed}] adversarial loss {adversarial_loss:.4f}")

    cfg = {
        "task": "params", "process": "nested_thomas", "method": "vihrs_checkpointed",
        "batch_size": batch_size, "n_epochs": n_epochs, "lr": lr,
        "embedding_dim": model.merge_dim, "seed": seed,
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
    }

    torch.save(
        {
            "model_state": best_state, "history": history, "config": cfg,
            "test_loss": test_loss, "test_loss_per_target": test_loss_per_target,
            "label_names": list(vihrs.LABEL_NAMES), "label_norm": label_norm,
            "label_log_mean": label_norm["mean"], "label_log_std": label_norm["std"],
            "label_transforms": ["log"] * len(vihrs.LABEL_NAMES),
            "n_points_norm": n_norm, "lr_norm": lr_norm,
            "adversarial_loss": adversarial_loss, "adversarial_loss_per_target": adversarial_loss_per_target,
            "adversarial_path": str(adversarial_path) if adversarial_path else None,
        },
        output_dir / "results.pt",
    )
    torch.save(model.cpu(), output_dir / "model.pt")

    json_payload: dict[str, Any] = {
        "task": "params", "method": "vihrs_checkpointed", "test_loss": test_loss,
        "test_loss_per_target": test_loss_per_target, "seed": seed,
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
    }
    if adversarial_loss is not None:
        json_payload["adversarial_loss"] = adversarial_loss
        json_payload["adversarial_loss_per_target"] = adversarial_loss_per_target
        json_payload["adversarial_path"] = str(adversarial_path)
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    return {"seed": seed, "test_loss": test_loss, "adversarial_loss": adversarial_loss, "best_epoch": best_epoch}
