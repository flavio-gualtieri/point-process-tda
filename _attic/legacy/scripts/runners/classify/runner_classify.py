import json
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from cloudforger.nn.splits import train_val_test_split
from cloudforger.nn.train_old import train_one_epoch, evaluate
from cloudforger.nn.heads.classifier import ClassificationHead
from cloudforger.nn.models.single_modal import SingleModalModel


def run(cfg: dict):
    dispatch = {
        "raw_pc": _run_raw_pc,
        "pi": _run_pi,
    }
    method = cfg["method"]
    if method not in dispatch:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(dispatch)}")
    dispatch[method](cfg)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _setup(cfg: dict):
    """Create output dir, seed, return device."""
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _make_loaders(dataset, cfg: dict):
    train_ds, val_ds, test_ds = train_val_test_split(dataset, seed=cfg["seed"])
    kw = dict(batch_size=cfg["batch_size"])
    return (
        DataLoader(train_ds, **kw, shuffle=True),
        DataLoader(val_ds,   **kw),
        DataLoader(test_ds,  **kw),
    )


def _train_and_eval(model, loaders, cfg: dict, device: str, tag: str):
    """Standard training loop. Returns (history, best_state, test_acc)."""
    train_loader, val_loader, test_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn   = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None

    for epoch in range(1, cfg["n_epochs"] + 1):
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
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{tag}] Epoch {epoch:3d} | "
            f"train acc {train_acc:.3f} | val acc {val_acc:.3f}"
        )

    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"\n[{tag}] Test accuracy: {test_acc:.3f}")
    return history, best_state, test_acc


def _save(cfg: dict, best_state, history, test_acc, label_names, extra_meta: dict):
    output_dir = Path(cfg["output_dir"])
    torch.save(
        {
            "model_state": best_state,
            "history":     history,
            "config":      cfg,
            "test_acc":    test_acc,
            "label_names": label_names,
        },
        output_dir / "results.pt",
    )
    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "task":     cfg["task"],
                "method":   cfg["method"],
                "test_acc": test_acc,
                "seed":     cfg["seed"],
                **extra_meta,
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Method: raw_pc
# ---------------------------------------------------------------------------

def _run_raw_pc(cfg: dict):
    from cloudforger.nn.data import PointCloudDataset
    from cloudforger.nn.encoders.point_cloud import PointNetEncoder

    device = _setup(cfg)

    with open(cfg["dataset_path"], "rb") as f:
        data = pickle.load(f)

    clouds = data["clouds"]
    labels = data["labels"]
    label_names = data["label_names"]
    n_classes = len(label_names)
    dim = clouds[0].dimension

    dataset = PointCloudDataset(clouds, labels, n_points=cfg["n_points"])
    loaders = _make_loaders(dataset, cfg)

    encoder = PointNetEncoder(
        input_dim=dim,
        embedding_dim=cfg["embedding_dim"],
        hidden_dims=cfg["hidden_dims"],
    )
    head  = ClassificationHead(embedding_dim=cfg["embedding_dim"], n_classes=n_classes)
    model = SingleModalModel(encoder=encoder, head=head).to(device)

    tag = f"dim={cfg['ambient_dim']}"
    history, best_state, test_acc = _train_and_eval(model, loaders, cfg, device, tag)
    _save(cfg, best_state, history, test_acc, label_names,
          extra_meta={"ambient_dim": cfg["ambient_dim"]})


# ---------------------------------------------------------------------------
# Method: pi  (persistence image)
# ---------------------------------------------------------------------------

class _SingleDimDataset(torch.utils.data.Dataset):
    """Extracts a single homology-dim array from a PersistenceImageDataset."""

    def __init__(self, base, key: str):
        self.base = base
        self.key  = key

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x[self.key], y


def _run_pi(cfg: dict):
    from cloudforger.nn.data import PersistenceImageDataset
    from cloudforger.nn.encoders.persistence_image import PIEncoder

    device = _setup(cfg)

    with open(cfg["dataset_path"], "rb") as f:
        data = pickle.load(f)

    images = data["images"]   # list of {0: (R,R) array, 1: (R,R) array}
    labels = data["labels"]
    label_names = data["label_names"]
    n_classes = len(label_names)

    hom_dim = cfg["hom_dim"]
    key = f"h{hom_dim}"
    dataset = _SingleDimDataset(PersistenceImageDataset(images, labels), key=key)
    loaders = _make_loaders(dataset, cfg)

    encoder = PIEncoder(in_channels=1, embedding_dim=cfg["embedding_dim"])
    head = ClassificationHead(embedding_dim=cfg["embedding_dim"], n_classes=n_classes)
    model = SingleModalModel(encoder=encoder, head=head).to(device)

    tag = f"H{hom_dim}"
    history, best_state, test_acc = _train_and_eval(model, loaders, cfg, device, tag)
    _save(cfg, best_state, history, test_acc, label_names,
          extra_meta={"hom_dim": hom_dim})