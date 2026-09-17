"""Train one model: AdamW, early stopping on validation loss, best state restored.

Batches are (images, covariates, targets), where images is the per-homology-dimension list PHNet
takes. Every metric here is the loss the task was trained on; anything per-regime is computed later
from the saved per-pattern predictions, never here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _move(batch, device):
    images, covariates, targets = batch
    return [x.to(device) for x in images], covariates.to(device), targets.to(device)


@torch.no_grad()
def evaluate(model, loader: DataLoader, loss_fn: nn.Module, device) -> tuple[float, float]:
    """(mean loss, accuracy). Accuracy is nan unless the targets are class indices."""
    model.eval()
    total, correct, seen = 0.0, None, 0
    for batch in loader:
        images, covariates, targets = _move(batch, device)
        out = model(images, covariates)
        total += loss_fn(out, targets).item() * len(targets)
        if targets.dim() == 1:   # class indices; regression targets are (B, n_targets)
            correct = (correct or 0) + (out.argmax(dim=-1) == targets).sum().item()
        seen += len(targets)
    return total / seen, (correct / seen if correct is not None else float("nan"))


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device,
    lr: float = 1e-3,
    weight_decay: float = 3e-4,
    epochs: int = 200,
    patience: int = 30,
    tag: str = "",
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            images, covariates, targets = _move(batch, device)
            optimizer.zero_grad()
            loss = loss_fn(model(images, covariates), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
            seen += len(targets)
        train_loss = total / seen
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}"
              f"{f'  acc {val_acc:.4f}' if val_acc == val_acc else ''}"
              f"{'  *' if epoch == best_epoch else ''}", flush=True)
        if epoch - best_epoch >= patience:
            print(f"[{tag}] no improvement in {patience} epochs, stopping", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"[{tag}] validation loss was never finite; nothing to restore")
    model.load_state_dict(best_state)
    return {"best_val_loss": best_loss, "best_epoch": best_epoch, "epochs_run": len(history),
            "history": history}


@torch.no_grad()
def predict(model, loader: DataLoader, device) -> np.ndarray:
    """Raw outputs in loader order: standardized predictions for regression, logits for a classifier."""
    model.eval()
    parts = [model(*_move(batch, device)[:2]).cpu().numpy() for batch in loader]
    return np.concatenate(parts, axis=0)
