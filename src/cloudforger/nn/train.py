# src/pointforge/nn/train.py

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _to_device(inputs, device):
    if isinstance(inputs, dict):
        return {k: v.to(device) for k, v in inputs.items()}
    return inputs.to(device)


def _batch_size(labels: torch.Tensor) -> int:
    return labels.size(0)


def train_one_epoch(
        model,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for inputs, labels in loader:
        inputs = _to_device(inputs, device)
        labels = labels.to(device)
        n = _batch_size(labels)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * n
        if labels.dim() == 1:
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_seen += n

    acc = total_correct / total_seen if total_seen > 0 else float("nan")

    return total_loss / total_seen, acc


@torch.no_grad()
def evaluate(model, loader: DataLoader, loss_fn: nn.Module, device: torch.device):
    model.eval()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for inputs, labels in loader:
        inputs = _to_device(inputs, device)
        labels = labels.to(device)
        n = _batch_size(labels)

        logits = model(inputs)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * n
        if labels.dim() == 1:
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_seen += n

    acc = total_correct / total_seen if total_seen > 0 else float("nan")
    
    return total_loss / total_seen, acc


def train_and_eval(model, loaders, cfg: dict, device: str, tag: str):
    train_loader, val_loader, test_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss, best_state = float("inf"), None

    for epoch in range(1, cfg["n_epochs"] + 1):
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _ = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{tag}] Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

    model.load_state_dict(best_state)
    test_loss, _ = evaluate(model, test_loader, loss_fn, device)
    print(f"\n[{tag}] Test loss: {test_loss:.4f}")
    return history, best_state, test_loss