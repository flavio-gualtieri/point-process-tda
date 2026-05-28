# scripts/experiment_04_torus.py
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cloudforger.nn.data import (
    PointCloudDataset,
    PersistenceImageDataset,
    CorrelationFeatureDataset,
)
from cloudforger.nn.encoders.point_cloud import PointNetEncoder
from cloudforger.nn.encoders.persistence_image import PIEncoder
from cloudforger.nn.encoders.stats import StatsEncoder
from cloudforger.nn.heads.classifier import ClassificationHead
from cloudforger.nn.models.single_modal import SingleModalModel
from cloudforger.nn.models.multi_modal import MultiModalModel
from cloudforger.nn.train import train_one_epoch, evaluate
from cloudforger.nn.splits import train_val_test_split


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    "batch_size": 32,
    "n_epochs": 500,
    "lr": 1e-3,
    "n_points": 500,
    "embedding_dim": 128,
    "hidden_dims": (64, 128, 256),
    "stats_hidden_dims": (128, 128),
    "seed": 0,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "output_dir": Path("results/experiment_04_torus"),
    "cloud_path": Path("data/clouds_torus.pkl"),
    "pi_path": Path("data/images_torus.pkl"),
    "edge_path": Path("data/features.pkl"),
}

CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
torch.manual_seed(CONFIG["seed"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SingleDimDataset(Dataset):
    """Extracts a single homology-dim tensor from a PersistenceImageDataset."""
    def __init__(self, base, key: str):
        self.base, self.key = base, key

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x[self.key], y


def make_loaders(dataset):
    """Standard train/val/test split + DataLoaders."""
    train_ds, val_ds, test_ds = train_val_test_split(dataset, seed=CONFIG["seed"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])
    return train_loader, val_loader, test_loader


def run_experiment(name, model, loaders, label_names):
    """Train `model` on `loaders=(train, val, test)`, save best-val checkpoint."""
    train_loader, val_loader, test_loader = loaders
    device = CONFIG["device"]
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None

    print(f"\n=== Running experiment: {name} ===")
    for epoch in range(1, CONFIG["n_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{name}] Epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val   loss {val_loss:.4f} acc {val_acc:.3f}"
        )

    # --- Final test (best-val checkpoint) ---
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"[{name}] Test accuracy (best-val checkpoint): {test_acc:.3f}")

    # --- Save ---
    out_path = CONFIG["output_dir"] / f"results_{name}.pt"
    torch.save({
        "model_state": best_state,
        "history": history,
        "config": CONFIG,
        "test_acc": test_acc,
        "label_names": label_names,
    }, out_path)
    print(f"[{name}] Saved to {out_path}")

    return test_acc


# ---------------------------------------------------------------------------
# Experiment 1: Point cloud (PointNet)
# ---------------------------------------------------------------------------

with open(CONFIG["cloud_path"], "rb") as f:
    cloud_data = pickle.load(f)

clouds = cloud_data["clouds"]
labels = cloud_data["labels"]
label_names = cloud_data["label_names"]
n_classes = len(label_names)
point_dim = clouds[0].dimension
print(f"[pc] Loaded {len(clouds)} samples, {n_classes} classes, point dim {point_dim}")

pc_dataset = PointCloudDataset(clouds, labels, n_points=CONFIG["n_points"])
pc_loaders = make_loaders(pc_dataset)

pc_model = SingleModalModel(
    encoder=PointNetEncoder(
        input_dim=point_dim,
        embedding_dim=CONFIG["embedding_dim"],
        hidden_dims=CONFIG["hidden_dims"],
    ),
    head=ClassificationHead(
        embedding_dim=CONFIG["embedding_dim"],
        n_classes=n_classes,
    ),
)
run_experiment("pc", pc_model, pc_loaders, label_names)


# ---------------------------------------------------------------------------
# Experiments 2-4: Persistence images (H0, H1, H0+H1)
# ---------------------------------------------------------------------------

with open(CONFIG["pi_path"], "rb") as f:
    pi_data = pickle.load(f)

images = pi_data["images"]
labels = pi_data["labels"]
label_names = pi_data["label_names"]
n_classes = len(label_names)
dims = sorted(images[0].keys())  # e.g. [0, 1]
print(f"[pi] Loaded {len(images)} samples, {n_classes} classes, homology dims {dims}")

pi_base = PersistenceImageDataset(images, labels)

# --- H0 only ---
h0_loaders = make_loaders(_SingleDimDataset(pi_base, key="h0"))
h0_model = SingleModalModel(
    encoder=PIEncoder(in_channels=1, embedding_dim=CONFIG["embedding_dim"]),
    head=ClassificationHead(embedding_dim=CONFIG["embedding_dim"], n_classes=n_classes),
)
run_experiment("h0", h0_model, h0_loaders, label_names)

# --- H1 only ---
h1_loaders = make_loaders(_SingleDimDataset(pi_base, key="h1"))
h1_model = SingleModalModel(
    encoder=PIEncoder(in_channels=1, embedding_dim=CONFIG["embedding_dim"]),
    head=ClassificationHead(embedding_dim=CONFIG["embedding_dim"], n_classes=n_classes),
)
run_experiment("h1", h1_model, h1_loaders, label_names)

# --- H0 + H1 (late fusion) ---
full_pi_loaders = make_loaders(pi_base)
full_pi_model = MultiModalModel(
    encoders={
        f"h{d}": PIEncoder(in_channels=1, embedding_dim=CONFIG["embedding_dim"])
        for d in dims
    },
    head=ClassificationHead(
        embedding_dim=CONFIG["embedding_dim"] * len(dims),
        n_classes=n_classes,
    ),
)
run_experiment("full_PI", full_pi_model, full_pi_loaders, label_names)


# ---------------------------------------------------------------------------
# Experiment 5: Edge / correlation statistics
# ---------------------------------------------------------------------------

with open(CONFIG["edge_path"], "rb") as f:
    edge_data = pickle.load(f)

features = edge_data["features"]
labels = edge_data["labels"]
label_names = edge_data["label_names"]
n_classes = len(label_names)

edge_dataset = CorrelationFeatureDataset(features, labels, statistic_names=None)
print(f"[edge] Loaded {len(edge_dataset)} samples, {n_classes} classes, "
      f"feature dim {edge_dataset.input_dim}")

edge_loaders = make_loaders(edge_dataset)
edge_model = SingleModalModel(
    encoder=StatsEncoder(
        input_dim=edge_dataset.input_dim,
        embedding_dim=CONFIG["embedding_dim"],
        hidden_dims=CONFIG["stats_hidden_dims"],
    ),
    head=ClassificationHead(
        embedding_dim=CONFIG["embedding_dim"],
        n_classes=n_classes,
    ),
)
run_experiment("edge", edge_model, edge_loaders, label_names)