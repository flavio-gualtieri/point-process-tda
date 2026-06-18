# scripts/experiment_01_pointcloud_only.py

import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from cloudforger.nn.data import PointCloudDataset
from cloudforger.nn.encoders.point_cloud import PointNetEncoder
from cloudforger.nn.heads.paramest import ParameterEstimator
from cloudforger.nn.train import train_one_epoch, evaluate
from cloudforger.nn.splits import train_val_test_split

# --- Config ---
CONFIG = {
    "dataset_path": Path("data/clouds_params.pkl"),
    "batch_size": 32,
    "n_epochs": 500,
    "n_params": 2,
    "lr": 1e-3,
    "n_points": None,                  # full variable-length clouds
    "use_cardinality_feature": True,   # feed normalized log-count to the head
    "embedding_dim": 128,
    "hidden_dims": (64, 128, 256),
    "seed": 0,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "output_dir": Path("results/experiment_04_pointcloud"),
}
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

# --- Reproducibility ---
torch.manual_seed(CONFIG["seed"])

# --- Data ---
with open(CONFIG["dataset_path"], "rb") as f:
    data = pickle.load(f)
clouds = data["clouds"]
params = data["labels"]
dim = clouds[0].dimension

dataset = PointCloudDataset(clouds, params, n_points=CONFIG["n_points"], dtype=torch.float32)
n = len(dataset)
print(f"Dataset size: {n} samples")

train_ds, val_ds, test_ds = train_val_test_split(dataset, seed=CONFIG["seed"])

# --- Train-split normalization statistics (targets + log-count) ---
# Computed on TRAIN ONLY (no leakage). Uses a DataLoader so it works whether
# the split returns torch Subsets or custom datasets.
def _stats_collate(batch):
    clouds_b, labels_b = zip(*batch)
    lengths = torch.tensor([c.shape[0] for c in clouds_b], dtype=torch.float32)
    labels = torch.stack(labels_b).float()
    return lengths, labels

_stats_loader = DataLoader(train_ds, batch_size=256, collate_fn=_stats_collate)
_lengths, _labels = [], []
for lengths, labels in _stats_loader:
    _lengths.append(lengths)
    _labels.append(labels)
_lengths = torch.cat(_lengths)
_labels = torch.cat(_labels, dim=0)

target_mean = _labels.mean(0)
target_std = _labels.std(0).clamp_min(1e-8)
log_counts = torch.log1p(_lengths)
logcount_mean = log_counts.mean()
logcount_std = log_counts.std().clamp_min(1e-8)

print(f"Train target mean: {target_mean.tolist()}  std: {target_std.tolist()}")
print(f"Train cloud size: min={int(_lengths.min())}  "
      f"median={int(_lengths.median())}  max={int(_lengths.max())}")

# --- Collate: pad ragged clouds, carry lengths, normalize targets ---
def make_collate(t_mean, t_std):
    def collate(batch):
        clouds_b, labels_b = zip(*batch)
        lengths = torch.tensor([c.shape[0] for c in clouds_b], dtype=torch.long)
        max_n = int(lengths.max())
        d = clouds_b[0].shape[1]
        padded = torch.zeros(len(clouds_b), max_n, d, dtype=torch.float32)
        for i, c in enumerate(clouds_b):
            padded[i, : c.shape[0]] = c
        labels = torch.stack(labels_b).float()
        labels = (labels - t_mean) / t_std            # normalized targets
        return {"points": padded, "lengths": lengths}, labels
    return collate

collate = make_collate(target_mean, target_std)
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], collate_fn=collate)
test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], collate_fn=collate)

# --- Model ---
encoder = PointNetEncoder(
    input_dim=dim,
    embedding_dim=CONFIG["embedding_dim"],
    hidden_dims=CONFIG["hidden_dims"],
)
extra = 1 if CONFIG["use_cardinality_feature"] else 0
head = ParameterEstimator(
    embedding_dim=CONFIG["embedding_dim"] + extra,   # +1 for the log-count feature
    n_params=CONFIG["n_params"],
    hidden_dims=CONFIG["hidden_dims"],
)


class PointCloudRegressor(nn.Module):
    """
    Replaces SingleModalModel. Encodes each cloud on its REAL points only
    (no padding fed to the encoder), optionally concatenates a normalized
    log-count feature, then applies the head.
    """
    def __init__(self, encoder, head, embedding_dim, use_card, c_mean, c_std):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.embedding_dim = embedding_dim
        self.use_card = use_card
        self.register_buffer("c_mean", torch.tensor(float(c_mean)))
        self.register_buffer("c_std", torch.tensor(float(c_std)))
        self._checked = False

    def forward(self, inputs):
        points = inputs["points"]      # (B, N, D)  -- padded container
        lengths = inputs["lengths"]    # (B,)
        embs = []
        for i in range(points.shape[0]):
            m = int(lengths[i])
            if m == 0:
                embs.append(points.new_zeros(self.embedding_dim))
                continue
            out = self.encoder(points[i, :m].unsqueeze(0))   # (1, m, D) -> expect (1, E)
            emb = out.reshape(-1)
            if not self._checked:
                if emb.numel() != self.embedding_dim:
                    raise RuntimeError(
                        f"Encoder returned shape {tuple(out.shape)} (flattened "
                        f"{emb.numel()}); expected embedding_dim={self.embedding_dim}. "
                        f"This wrapper assumes PointNetEncoder((1,m,D)) -> (1,E). "
                        f"Adjust forward() to match the real signature."
                    )
                self._checked = True
            embs.append(emb)
        emb = torch.stack(embs, dim=0)                       # (B, E)
        if self.use_card:
            log_c = (torch.log1p(lengths.float()) - self.c_mean) / self.c_std
            emb = torch.cat([emb, log_c.unsqueeze(1)], dim=1)  # (B, E+1)
        return self.head(emb)


model = PointCloudRegressor(
    encoder=encoder,
    head=head,
    embedding_dim=CONFIG["embedding_dim"],
    use_card=CONFIG["use_cardinality_feature"],
    c_mean=logcount_mean,
    c_std=logcount_std,
).to(CONFIG["device"])

# --- Training setup ---
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
loss_fn = nn.MSELoss()   # on normalized targets

# --- Training loop ---
history = {"train_loss": [], "val_loss": []}
best_val_loss, best_state = float("inf"), None
for epoch in range(1, CONFIG["n_epochs"] + 1):
    train_loss, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, CONFIG["device"])
    val_loss, _ = evaluate(model, val_loader, loss_fn, CONFIG["device"])
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

# --- Final test (best-val checkpoint) ---
model.load_state_dict(best_state)
test_loss, _ = evaluate(model, test_loader, loss_fn, CONFIG["device"])
print(f"\nTest MSE (normalized space, best-val checkpoint): {test_loss:.4f}")

# Per-parameter metrics in ORIGINAL units
device = CONFIG["device"]
t_mean, t_std = target_mean.to(device), target_std.to(device)
model.eval()
abs_err = torch.zeros(CONFIG["n_params"], device=device)
sq_err = torch.zeros(CONFIG["n_params"], device=device)
seen = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        pred = model(inputs) * t_std + t_mean      # de-normalize predictions
        true = labels * t_std + t_mean             # de-normalize targets
        abs_err += (pred - true).abs().sum(0)
        sq_err += ((pred - true) ** 2).sum(0)
        seen += labels.shape[0]
mae = (abs_err / seen).tolist()
rmse = (sq_err / seen).sqrt().tolist()
names = data.get("label_names", [f"param_{i}" for i in range(CONFIG["n_params"])])
print("\nTest metrics in original units:")
for nm, a, r in zip(names, mae, rmse):
    print(f"  {nm:18s} MAE={a:.4f}  RMSE={r:.4f}")

# --- Save results ---
torch.save({
    "model_state": best_state,
    "history": history,
    "config": CONFIG,
    "test_loss_normalized": test_loss,
    "test_mae_original": dict(zip(names, mae)),
    "test_rmse_original": dict(zip(names, rmse)),
    "target_mean": target_mean,
    "target_std": target_std,
    "logcount_mean": float(logcount_mean),
    "logcount_std": float(logcount_std),
}, CONFIG["output_dir"] / "results.pt")
print(f"\nSaved results to {CONFIG['output_dir'] / 'results.pt'}")