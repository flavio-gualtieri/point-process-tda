# scripts/runners/runner_params.py

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- make the cloudforger package importable when run as a plain script ------
# scripts/runners/<this file>  ->  parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cloudforger.core.cloud import PointCloud
from cloudforger.core.region import Box
from cloudforger.nn.splits import train_val_test_split
from cloudforger.nn.train import train_one_epoch, evaluate
from cloudforger.nn.heads.paramest import ParameterEstimator
from cloudforger.nn.models.single_modal import SingleModalModel


# Per-process number of parameters the estimator head predicts.
N_PARAMS = {"matern": 2, "thomas": 3}

# Method -> file under data/params/<process>/.
METHOD_FILES = {
    "raw_pc": "clouds.pkl",
    "pi": "images.pkl",
    "pairwise": "features.pkl",
    "betti": "betti.pkl",
}
# Order methods run in for method == "all".
ALL_METHODS = ["raw_pc", "pi_0", "pi_1", "pairwise", "betti_0", "betti_1"]

# Persistence tokens carry the homology dim: "pi_0", "pi_1", "betti_0", ...
_PERSIST_TOKEN = re.compile(r"^(pi|betti)_(\d+)$")


def _parse_method(method: str) -> tuple[str, list[int] | None]:
    """Split a method token into (base, hom_dims).

    "pi_0" -> ("pi", [0]); "betti_1" -> ("betti", [1]); "pi" -> ("pi", None);
    "raw_pc" -> ("raw_pc", None). hom_dims is None means "use the default".
    """
    m = _PERSIST_TOKEN.match(method)
    if m:
        return m.group(1), [int(m.group(2))]
    return method, None


def run(cfg: dict):
    method = cfg["method"]

    # Run every method back to back (e.g. a full Thomas sweep).
    if method == "all":
        results = {}
        for m in ALL_METHODS:
            print(f"\n========== method: {m} ==========")
            results[m] = run({**cfg, "method": m})
        print("\nAll methods done.")
        return results

    # Auto-expand "pi" / "betti" (no dim suffix) → run H0 then H1 separately.
    if method in ("pi", "betti"):
        results = {}
        for dim in (0, 1):
            dim_method = f"{method}_{dim}"
            dim_cfg = {**cfg, "method": dim_method}
            if cfg.get("output_dir"):
                dim_cfg["output_dir"] = str(Path(cfg["output_dir"]) / dim_method)
            print(f"\n========== method: {dim_method} ==========")
            results[dim] = run(dim_cfg)
        return results

    base, hom_dims = _parse_method(method)
    if base == "raw_pc":
        return _run_raw_pc(cfg)
    if base == "pairwise":
        return _run_pairwise(cfg)
    if base == "pi":
        return _run_pi(cfg, hom_dims[0])
    if base == "betti":
        return _run_betti(cfg, hom_dims[0])
    raise ValueError(
        f"Unknown method '{method}'. Available: raw_pc, pairwise, pi, betti, "
        f"pi_<dim>, betti_<dim>, all"
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _data_dir(cfg: dict) -> Path:
    """Base directory holding the datasets: data/params/<process>/."""
    base = Path(cfg["data_dir"]) if cfg.get("data_dir") else (PROJECT_ROOT / "data")
    return base / "params" / cfg["process"]


def _dataset_path(cfg: dict, method: str) -> Path:
    base, _ = _parse_method(method)
    if cfg.get("dataset_dir"):  # explicit directory from the launcher/config
        return Path(cfg["dataset_dir"]) / METHOD_FILES[base]
    return _data_dir(cfg) / METHOD_FILES[base]


def _results_dir(cfg: dict) -> Path:
    """Base directory for outputs: results/params/<process>/."""
    base = Path(cfg["results_dir"]) if cfg.get("results_dir") else (PROJECT_ROOT / "results")
    return base / "params" / cfg["process"]


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Cloud-record helpers (clouds.pkl is now a flat list of dict records)
# ---------------------------------------------------------------------------

def _to_pointcloud(cloud) -> PointCloud:
    """Rebuild a PointCloud (with its Box region) from a dict record, or pass a
    PointCloud through unchanged."""
    if not isinstance(cloud, dict):
        return cloud
    region = None
    reg = cloud.get("region")
    if reg is not None:
        try:
            region = Box(low=np.asarray(reg["low"], dtype=float),
                         high=np.asarray(reg["high"], dtype=float))
        except Exception:
            region = None
    return PointCloud(
        points=np.asarray(cloud["points"]),
        generator_name=cloud.get("process", ""),
        generator_params=dict(cloud.get("params", {})),
        seed=cloud.get("seed"),
        region=region,
    )


def _labels_from_clouds(cloud_list) -> tuple[np.ndarray, list[str]]:
    """Derive the (N, P) label array and names from each cloud's params."""
    def params_of(c):
        return dict(c["params"]) if isinstance(c, dict) else dict(c.generator_params)
    params = [params_of(c) for c in cloud_list]
    names = list(params[0].keys())
    for p in params:
        if list(p.keys()) != names:
            raise ValueError("Clouds have inconsistent parameter keys; cannot stack labels.")
    labels = np.array([[p[k] for k in names] for p in params], dtype=float)
    return labels, names


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _setup(cfg: dict):
    """Ensure the output dir exists, seed, return device."""
    base = Path(cfg["output_dir"]) if cfg.get("output_dir") else _results_dir(cfg)
    base.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _normalize_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-transform then z-score each parameter column.

    All Thomas/Matérn parameters are strictly positive and drawn log-uniformly,
    so log-space is the natural scale. Standardizing afterwards gives each
    parameter equal weight in the MSE regardless of its absolute magnitude.

    Returns (normed_labels, log_mean, log_std) — save mean/std to invert at
    inference: original = exp(pred * log_std + log_mean).
    """
    log_labels = np.log(labels)
    log_mean = log_labels.mean(axis=0)
    log_std  = log_labels.std(axis=0)
    log_std  = np.where(log_std == 0, 1.0, log_std)  # guard constant columns
    return (log_labels - log_mean) / log_std, log_mean, log_std


def _make_head(cfg: dict) -> ParameterEstimator:
    process = cfg["process"]
    if process not in N_PARAMS:
        raise ValueError(f"Unknown process '{process}'. Available: {sorted(N_PARAMS)}")
    return ParameterEstimator(embedding_dim=cfg["embedding_dim"], n_params=N_PARAMS[process])


def _make_loaders(dataset, cfg: dict):
    train_ds, val_ds, test_ds = train_val_test_split(dataset, seed=cfg["seed"])
    kw = dict(batch_size=cfg["batch_size"])
    return (
        DataLoader(train_ds, **kw, shuffle=True),
        DataLoader(val_ds, **kw),
        DataLoader(test_ds, **kw),
    )


def _train_and_eval(model, loaders, cfg: dict, device: str, tag: str):
    """Standard training loop. Returns (history, best_state, test_loss)."""
    train_loader, val_loader, test_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss, best_state = float("inf"), None

    for epoch in range(1, cfg["n_epochs"] + 1):
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _   = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{tag}] Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

    model.load_state_dict(best_state)
    test_loss, _ = evaluate(model, test_loader, loss_fn, device)
    print(f"\n[{tag}] Test loss: {test_loss:.4f}")
    return history, best_state, test_loss


def _save(cfg: dict, best_state, history, test_loss, label_names, extra_meta: dict, subdir: str,
          label_log_mean: np.ndarray | None = None, label_log_std: np.ndarray | None = None):
    """Write results.pt / results.json into results/params/<process>/<subdir>/.

    Each method (and each homology dim) gets its own subdir so a run-all does not
    overwrite earlier results.
    """
    # An explicit output_dir (from the launcher/config, already per-method) is
    # the final directory; otherwise fall back to results/params/<process>/<subdir>.
    if cfg.get("output_dir"):
        output_dir = Path(cfg["output_dir"])
    else:
        output_dir = _results_dir(cfg) / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_payload = {
        "model_state": best_state,
        "history": history,
        "config": cfg,
        "test_loss": test_loss,
        "label_names": label_names,
    }
    if label_log_mean is not None:
        pt_payload["label_log_mean"] = label_log_mean
        pt_payload["label_log_std"]  = label_log_std
    torch.save(pt_payload, output_dir / "results.pt")
    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "task": cfg["task"],
                "method": cfg["method"],
                "test_loss": test_loss,
                "seed": cfg["seed"],
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

    raw = _load_pickle(_dataset_path(cfg, "raw_pc"))
    # clouds.pkl is a flat list of dict records (or a {"clouds": [...]} wrapper).
    cloud_list = raw["clouds"] if isinstance(raw, dict) and "clouds" in raw else raw

    clouds = [_to_pointcloud(c) for c in cloud_list]
    labels, label_names = _labels_from_clouds(cloud_list)
    labels, log_mean, log_std = _normalize_labels(labels)
    dim = clouds[0].dimension

    dataset = PointCloudDataset(clouds, labels, n_points=cfg["n_points"], dtype=torch.float32)
    loaders = _make_loaders(dataset, cfg)

    encoder = PointNetEncoder(
        input_dim=dim,
        embedding_dim=cfg["embedding_dim"],
        hidden_dims=cfg["hidden_dims"],
    )
    head = _make_head(cfg)
    model = SingleModalModel(encoder=encoder, head=head).to(device)

    tag = f"raw_pc(dim={dim})"
    history, best_state, test_loss = _train_and_eval(model, loaders, cfg, device, tag)
    _save(cfg, best_state, history, test_loss, label_names,
          extra_meta={"ambient_dim": dim}, subdir="raw_pc",
          label_log_mean=log_mean, label_log_std=log_std)
    return {"history": history, "test_loss": test_loss}


# ---------------------------------------------------------------------------
# Method: pi  (persistence image)  -- H0 and H1 separately
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


def _run_pi(cfg: dict, hom_dim: int):
    from cloudforger.nn.data import PersistenceImageDataset
    from cloudforger.nn.encoders.persistence_image import PIEncoder

    device = _setup(cfg)

    data = _load_pickle(_dataset_path(cfg, "pi"))
    images = data["images"]
    labels = data["labels"]
    label_names = data["label_names"]

    labels, log_mean, log_std = _normalize_labels(labels)
    key = f"h{hom_dim}"
    dataset = _SingleDimDataset(PersistenceImageDataset(images, labels, dtype=torch.float32), key=key)
    loaders = _make_loaders(dataset, cfg)

    encoder = PIEncoder(in_channels=1, embedding_dim=cfg["embedding_dim"])
    head = _make_head(cfg)
    model = SingleModalModel(encoder=encoder, head=head).to(device)

    tag = f"pi_{hom_dim}"
    history, best_state, test_loss = _train_and_eval(model, loaders, cfg, device, tag)
    _save(cfg, best_state, history, test_loss, label_names,
          extra_meta={"hom_dim": hom_dim}, subdir=f"pi_{hom_dim}",
          label_log_mean=log_mean, label_log_std=log_std)
    return {"history": history, "test_loss": test_loss}


# ---------------------------------------------------------------------------
# Method: pairwise  (correlation features)
# ---------------------------------------------------------------------------

def _run_pairwise(cfg: dict):
    from cloudforger.nn.data import CorrelationFeatureDataset
    from cloudforger.nn.encoders.stats import StatsEncoder

    device = _setup(cfg)

    data = _load_pickle(_dataset_path(cfg, "pairwise"))
    features = data["features"]
    labels = data["labels"]
    label_names = data["label_names"]
    labels, log_mean, log_std = _normalize_labels(labels)

    dataset = CorrelationFeatureDataset(features, labels, dtype=torch.float32)
    loaders = _make_loaders(dataset, cfg)

    encoder = StatsEncoder(
        input_dim=dataset.input_dim,
        embedding_dim=cfg["embedding_dim"],
        hidden_dims=cfg["hidden_dims"],
    )
    head = _make_head(cfg)
    model = SingleModalModel(encoder=encoder, head=head).to(device)

    tag = "pairwise"
    history, best_state, test_loss = _train_and_eval(model, loaders, cfg, device, tag)
    _save(cfg, best_state, history, test_loss, label_names,
          extra_meta={}, subdir="pairwise",
          label_log_mean=log_mean, label_log_std=log_std)
    return {"history": history, "test_loss": test_loss}


# ---------------------------------------------------------------------------
# Method: betti  -- H0 and H1 separately
# ---------------------------------------------------------------------------

def _run_betti(cfg: dict, hom_dim: int):
    from cloudforger.nn.data import BettiCurveDataset
    from cloudforger.nn.encoders.stats import StatsEncoder

    device = _setup(cfg)

    data = _load_pickle(_dataset_path(cfg, "betti"))
    betti_curves = data["betti_curves"]
    labels = data["labels"]
    label_names = data["label_names"]
    labels, log_mean, log_std = _normalize_labels(labels)

    dataset = BettiCurveDataset(betti_curves, labels, homology_dims=[hom_dim])
    loaders = _make_loaders(dataset, cfg)

    encoder = StatsEncoder(
        input_dim=dataset.input_dim,
        embedding_dim=cfg["embedding_dim"],
        hidden_dims=cfg["hidden_dims"],
    )
    head = _make_head(cfg)
    model = SingleModalModel(encoder=encoder, head=head).to(device)

    tag = f"betti_{hom_dim}"
    history, best_state, test_loss = _train_and_eval(model, loaders, cfg, device, tag)
    _save(cfg, best_state, history, test_loss, label_names,
          extra_meta={"hom_dim": hom_dim}, subdir=f"betti_{hom_dim}",
          label_log_mean=log_mean, label_log_std=log_std)
    return {"history": history, "test_loss": test_loss}