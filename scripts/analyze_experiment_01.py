# scripts/analyze_experiment_01.py
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import matplotlib.pyplot as plt

import pathlib
torch.serialization.add_safe_globals([pathlib.PosixPath])
results = torch.load("results/experiment_01/results.pt", weights_only=True)
history = results["history"]
label_names = results["label_names"]

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
axes[0].legend()
axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["val_acc"], label="val")
axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
axes[1].legend()
plt.tight_layout()
plt.savefig("results/experiment_01/training_curves.png", dpi=150)

print(f"Final test accuracy: {results['test_acc']:.3f}")