# src/pointforge/nn/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def _to_device(inputs, device):
    if isinstance(inputs, dict):
        return {k: v.to(device) for k, v in inputs.items()}
    return inputs.to(device)


def _batch_size(labels: torch.Tensor) -> int:
    return labels.size(0)


def train_one_epoch(model, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device):
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
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_seen += n

    return total_loss / total_seen, total_correct / total_seen


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
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_seen += n

    return total_loss / total_seen, total_correct / total_seen