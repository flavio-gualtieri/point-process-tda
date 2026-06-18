# scripts/experiment_02_persistence_image.py  (PARAMETER ESTIMATION, H0)

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from cloudforger.nn.data import PersistenceImageDataset
from cloudforger.nn.encoders.persistence_image import PIEncoder
from cloudforger.nn.heads.paramest import ParameterEstimator     # CHANGED: regression head
from cloudforger.nn.models.single_modal import SingleModalModel
from cloudforger.nn.train import train_one_epoch, evaluate
from cloudforger.nn.splits import train_val_test_split

# --- Config ---
CONFIG = {
    "dataset_path": Path("data/images_params.pkl"),
    "batch_size": 32,
    "n_epochs": 500,
    "n_params": 2,                       # CHANGED: was n_classes
    "lr": 1e-3,
    "embedding_dim": 128,
    "hidden_dims": (64, 128, 256),       # CHANGED: ParameterEstimator expects hidden_dims
    "seed": 0,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "output_dir": Path("results/experiment_04_params_matern_h0"),   # CHANGED
}
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

# --- Reproducibility ---
torch.manual_seed(CONFIG["seed"])

# --- Data ---
with open(CONFIG["dataset_path"], "rb") as f:
    data = pickle.load(f)

images = data["images"]                                  # list of {0: (R,R), 1: (R,R)}
labels = np.asarray(data["labels"], dtype=np.float32)    # CHANGED: float regression targets (N, 2)
label_names = data.get("label_names", ["parent_intensity", "hardcore_radius"])
n_params = labels.shape[1]                               # CHANGED
print(f"Loaded {len(images)} samples, {n_params} params {label_names}, using H0")

# CHANGED: removed the classification-only visualization block (per-class
# persistence diagrams and images). It indexed one sample per discrete class
# via np.where(labels == cls_idx), which does not apply to continuous targets.


class _SingleDimDataset(torch.utils.data.Dataset):
    """Extracts a single homology-dim tensor from a PersistenceImageDataset."""
    def __init__(self, base, key: str):
        self.base, self.key = base, key

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x[self.key], y


pi_dataset = PersistenceImageDataset(images, labels)
# CHANGED: PersistenceImageDataset casts labels to long (class indices). Override
# to float for regression. (Cleaner: add a dtype= param mirroring PointCloudDataset.)
pi_dataset.labels = torch.as_tensor(labels, dtype=torch.float32)

dataset = _SingleDimDataset(pi_dataset, key="h0")
train_ds, val_ds, test_ds = train_val_test_split(dataset, seed=CONFIG["seed"])

# --- CHANGED: train-split target normalization (no leakage) ---
def _label_collate(batch):
    return torch.stack([b[1] for b in batch]).float()

_stats_loader = DataLoader(train_ds, batch_size=256, collate_fn=_label_collate)
_labels = torch.cat([b for b in _stats_loader], dim=0)
target_mean = _labels.mean(0)
target_std = _labels.std(0).clamp_min(1e-8)
print(f"Train target mean: {target_mean.tolist()}  std: {target_std.tolist()}")

def make_collate(t_mean, t_std):
    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])            # (B, 1, R, R)
        lbls = torch.stack([b[1] for b in batch]).float()
        lbls = (lbls - t_mean) / t_std                       # normalized targets
        return imgs, lbls
    return collate

collate = make_collate(target_mean, target_std)
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], collate_fn=collate)

# --- Model ---
encoder = PIEncoder(in_channels=1, embedding_dim=CONFIG["embedding_dim"])
head = ParameterEstimator(                                   # CHANGED
    embedding_dim=CONFIG["embedding_dim"],
    n_params=CONFIG["n_params"],
    hidden_dims=CONFIG["hidden_dims"],
)
model = SingleModalModel(encoder=encoder, head=head).to(CONFIG["device"])

# --- Training setup ---
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
loss_fn = nn.MSELoss()                                       # CHANGED: was CrossEntropyLoss

# --- Training loop ---
history = {"train_loss": [], "val_loss": []}                 # CHANGED: dropped acc
best_val_loss, best_state = float("inf"), None               # CHANGED: track min loss
for epoch in range(1, CONFIG["n_epochs"] + 1):
    train_loss, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, CONFIG["device"])
    val_loss, _ = evaluate(model, val_loader, loss_fn, CONFIG["device"])
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    if val_loss < best_val_loss:                             # CHANGED
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

# --- Final test (best-val checkpoint) ---
model.load_state_dict(best_state)
test_loss, _ = evaluate(model, test_loader, loss_fn, CONFIG["device"])
print(f"\nTest MSE (normalized space, best-val checkpoint): {test_loss:.4f}")

# CHANGED: per-parameter metrics in ORIGINAL units
device = CONFIG["device"]
t_mean, t_std = target_mean.to(device), target_std.to(device)
model.eval()
abs_err = torch.zeros(n_params, device=device)
sq_err = torch.zeros(n_params, device=device)
seen = 0
with torch.no_grad():
    for inputs, lbls in test_loader:
        inputs, lbls = inputs.to(device), lbls.to(device)
        pred = model(inputs) * t_std + t_mean
        true = lbls * t_std + t_mean
        abs_err += (pred - true).abs().sum(0)
        sq_err += ((pred - true) ** 2).sum(0)
        seen += lbls.shape[0]
mae = (abs_err / seen).tolist()
rmse = (sq_err / seen).sqrt().tolist()
print("\nTest metrics in original units:")
for nm, a, r in zip(label_names, mae, rmse):
    print(f"  {nm:18s} MAE={a:.4f}  RMSE={r:.4f}")

# --- Save results ---
torch.save({
    "model_state": best_state,
    "history": history,
    "config": CONFIG,
    "test_loss_normalized": test_loss,                       # CHANGED: was test_acc
    "test_mae_original": dict(zip(label_names, mae)),
    "test_rmse_original": dict(zip(label_names, rmse)),
    "target_mean": target_mean,
    "target_std": target_std,
    "label_names": label_names,
}, CONFIG["output_dir"] / "h0_results.pt")
print(f"\nSaved results to {CONFIG['output_dir'] / 'h0_results.pt'}")