# scripts/experiment_02_persistence_image.py
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from cloudforger.nn.data import PersistenceImageDataset  # NOT YET BUILT — see note
from cloudforger.nn.encoders.persistence_image import PIEncoder
from cloudforger.nn.heads.classifier import ClassificationHead
from cloudforger.nn.models.multi_modal import MultiModalModel
from cloudforger.nn.train import train_one_epoch, evaluate
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
dims = sorted(images[0].keys())  # e.g. [0, 1]
print(f"Loaded {len(images)} samples, {n_classes} classes, homology dims {dims}")

dataset = PersistenceImageDataset(images, labels)
train_ds, val_ds, test_ds = train_val_test_split(dataset, seed=CONFIG["seed"])
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])
test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])

# --- Model: one PIEncoder per homology dimension, fused late ---
encoders = {
    f"h{d}": PIEncoder(in_channels=1, embedding_dim=CONFIG["embedding_dim"])
    for d in dims
}
head = ClassificationHead(
    # Late fusion: head sees concatenated embeddings, one per encoder.
    embedding_dim=CONFIG["embedding_dim"] * len(encoders),
    n_classes=n_classes,
)
model = MultiModalModel(encoders=encoders, head=head).to(CONFIG["device"])

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
}, CONFIG["output_dir"] / "results.pt")