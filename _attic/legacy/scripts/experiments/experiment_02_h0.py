# scripts/experiment_02_persistence_image.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from cloudforger.nn.data import PersistenceImageDataset  # NOT YET BUILT — see note
from cloudforger.nn.encoders.persistence_image import PIEncoder
from cloudforger.nn.heads.classifier import ClassificationHead
from cloudforger.nn.models.single_modal import SingleModalModel
from cloudforger.nn.train_old import train_one_epoch, evaluate
from cloudforger.nn.splits import train_val_test_split

# --- Config ---
CONFIG = {
    "dataset_path": Path("data/images.pkl"),
    "batch_size": 32,
    "n_epochs": 500,
    "lr": 1e-3,
    "embedding_dim": 128,
    "seed": 0,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "output_dir": Path("results/experiment_02_pi"),
}
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

# --- Reproducibility ---
torch.manual_seed(CONFIG["seed"])

# --- Data ---
with open(CONFIG["dataset_path"], "rb") as f:
    data = pickle.load(f)

images = data["images"]          # list of {0: (R,R) array, 1: (R,R) array}
labels = data["labels"]
label_names = data["label_names"]
n_classes = len(label_names)
dims = [0]  # e.g. [0, 1]
print(f"Loaded {len(images)} samples, {n_classes} classes, homology dims {dims}")

# --- Visualize: persistence diagrams (H0 + H1) ---
with open("data/diagrams.pkl", "rb") as f:
    diag_bundle = pickle.load(f)
dgms = diag_bundle["diagrams"]
dgm_labels = diag_bundle["labels"]

# Collect one sample per class for dims 0 and 1
sample_pairs = {}
for cls_idx in range(n_classes):
    idx = int(np.where(dgm_labels == cls_idx)[0][0])
    sample_pairs[cls_idx] = {
        dim: dgms[idx].diagrams[dim] for dim in [0, 1]
    }

# Global axis limits: use finite points across all classes and both dims
all_finite = []
for cls_idx in range(n_classes):
    for dim in [0, 1]:
        pairs = sample_pairs[cls_idx][dim]
        finite = pairs[np.isfinite(pairs[:, 1])]
        if len(finite):
            all_finite.append(finite)
all_finite = np.vstack(all_finite) if all_finite else np.array([[0.0, 1.0]])
global_max = float(all_finite[:, 1].max())
axis_lim = (0.0, global_max * 1.05)

fig, axes = plt.subplots(2, n_classes, figsize=(4 * n_classes, 8))
colors = {0: "C0", 1: "C1"}
for cls_idx, cls_name in enumerate(label_names):
    for row, dim in enumerate([0, 1]):
        pairs = sample_pairs[cls_idx][dim]
        finite = pairs[np.isfinite(pairs[:, 1])]
        ax = axes[row, cls_idx]
        ax.scatter(finite[:, 0], finite[:, 1], s=6, alpha=0.6, color=colors[dim])
        ax.plot(axis_lim, axis_lim, "k--", lw=0.8)
        ax.set_xlim(axis_lim)
        ax.set_ylim(axis_lim)
        ax.set_aspect("equal")
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        ax.text(0.05, 0.91, f"H{dim}", transform=ax.transAxes,
                fontsize=9, color=colors[dim], fontweight="bold")
        if row == 0:
            ax.set_title(cls_name)

fig.suptitle("Persistence diagrams — H0 (top) / H1 (bottom), shared axes")
fig.tight_layout()
fig.savefig(CONFIG["output_dir"] / "h01_persistence_diagrams.png", dpi=150)
plt.close(fig)
print("Saved H0+H1 persistence diagrams")

# --- Visualize: persistence images (H0 + H1) ---
# Shared colormap limits across both dims and all classes
all_img_vals = []
for cls_idx in range(n_classes):
    sample_idx = int(np.where(np.asarray(labels) == cls_idx)[0][0])
    for dim in [0, 1]:
        all_img_vals.append(np.asarray(images[sample_idx][dim]).ravel())
all_img_vals = np.concatenate(all_img_vals)
vmin, vmax = float(all_img_vals.min()), float(all_img_vals.max())

fig, axes = plt.subplots(2, n_classes, figsize=(4 * n_classes, 8))
for cls_idx, cls_name in enumerate(label_names):
    sample_idx = int(np.where(np.asarray(labels) == cls_idx)[0][0])
    for row, dim in enumerate([0, 1]):
        img = np.asarray(images[sample_idx][dim])
        ax = axes[row, cls_idx]
        im = ax.imshow(img, origin="lower", aspect="equal", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel("birth")
        ax.set_ylabel("persistence")
        ax.text(0.05, 0.91, f"H{dim}", transform=ax.transAxes,
                fontsize=9, color="white", fontweight="bold")
        if row == 0:
            ax.set_title(cls_name)
        fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle("Persistence images — H0 (top) / H1 (bottom), shared color scale")
fig.tight_layout()
fig.savefig(CONFIG["output_dir"] / "h01_persistence_images.png", dpi=150)
plt.close(fig)
print("Saved H0+H1 persistence images")

class _SingleDimDataset(torch.utils.data.Dataset):
    """Extracts a single homology-dim tensor from a PersistenceImageDataset."""
    def __init__(self, base, key: str):
        self.base, self.key = base, key
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x[self.key], y

dataset = _SingleDimDataset(PersistenceImageDataset(images, labels), key="h0")
train_ds, val_ds, test_ds = train_val_test_split(dataset, seed=CONFIG["seed"])
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])
test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])

# --- Model ---
encoder = PIEncoder(in_channels=1, embedding_dim=CONFIG["embedding_dim"])
head = ClassificationHead(embedding_dim=CONFIG["embedding_dim"], n_classes=n_classes)
model = SingleModalModel(encoder=encoder, head=head).to(CONFIG["device"])

# --- Training setup ---
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
criterion = nn.CrossEntropyLoss()

# --- Training loop ---
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc, best_state = 0.0, None

for epoch in range(1, CONFIG["n_epochs"] + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, CONFIG["device"]
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, CONFIG["device"])
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(
        f"Epoch {epoch:3d} | "
        f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
        f"val loss {val_loss:.4f} acc {val_acc:.3f}"
    )

# --- Final test (best-val checkpoint) ---
model.load_state_dict(best_state)
test_loss, test_acc = evaluate(model, test_loader, criterion, CONFIG["device"])
print(f"\nTest accuracy (best-val checkpoint): {test_acc:.3f}")

# --- Save results ---
torch.save({
    "model_state": best_state,
    "history": history,
    "config": CONFIG,
    "test_acc": test_acc,
    "label_names": label_names,
}, CONFIG["output_dir"] / "h0_results.pt")