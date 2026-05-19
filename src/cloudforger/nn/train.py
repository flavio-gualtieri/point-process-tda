# src/pointforge/nn/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_seen += inputs.size(0)

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * inputs.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_seen += inputs.size(0)

    return total_loss / total_seen, total_correct / total_seen