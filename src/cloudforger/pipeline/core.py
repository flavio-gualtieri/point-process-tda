"""What every pipeline step shares: config, the row table, targets, weights and the prediction contract.

Rows are every bank pattern of every configured family, in (config family order, manifest order) --
the order the network builders use too -- and are identified by case_id everywhere.

Prediction contract (one predictions.npz per trained unit, keyed by case_id, never by row order):
    classifier   case_id, split, posterior (P, K) float32, classes (K,)
    estimator    case_id, split, theta_hat (P, T) float64 (natural scale), targets (T,), family
Classifiers cover val + test rows, plus out-of-fold train rows when the learner supports it.
Estimators cover val + test rows of EVERY family, so a pipeline can look up whatever the classifier
routes to them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..paths import BANK, read_config
from ..simulation.split import split_of
from .units import estimated_families, run_dir, unit_dir  # noqa: F401  (re-exported: the scripts' one import)

# The family's own parameters. Poisson has none to learn: nbar_hat = n is the exact MLE.
TARGETS = {
    "poisson": ["nbar"],
    "thomas": ["kappa", "mu", "sigma"],
    "nested": ["kappa", "mu1", "mu2", "sigma1", "sigma2"],
    "matern2": ["R", "lam_p"],
    "lgcp": ["nbar", "sigma2", "s"],
    "ring": ["kappa", "mu", "rho", "sigma"],
    "matern1": ["R", "lam_p"],
    "cell": ["nbar", "k"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------- config

def load_config(path: str | Path | None = None) -> dict:
    """The run's config (default configs/pipeline.yaml), CLOUDFORGER_CONFIGS overrides merged in."""
    return read_config(path or "pipeline.yaml")


def config_arg(parser) -> None:
    parser.add_argument("--config", help="default: configs/pipeline.yaml")


def save_config(cfg: dict, out: Path) -> None:
    """The config as run (overrides merged in), next to what it produced."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


# ------------------------------------------------------------------------------------------ rows

def rows(cfg: dict) -> pd.DataFrame:
    """Every bank row of every configured family, indexed by case_id, with `split` and parameters."""
    parts = [pd.read_csv(BANK / f / "manifest.csv") for f in cfg["families"]]
    out = pd.concat(parts, ignore_index=True)
    out["split"] = split_of(out.theta.to_numpy())
    out = out[out.theta % cfg.get("thin", 1) == 0]               # every k-th theta, in every split
    return out.set_index("case_id")


def family_weights(cfg: dict) -> dict:
    """The prior over families, summing to 1."""
    p = cfg["prior"]
    w = {f: 1.0 for f in cfg["families"]} if p == "equal" else {f: float(p[f]) for f in cfg["families"]}
    total = sum(w.values())
    return {f: v / total for f, v in w.items()}


def sample_weights(family: pd.Series | np.ndarray, cfg: dict) -> np.ndarray:
    """Per-row weights turning the rows' family counts into the prior; mean 1 over the rows given."""
    family = pd.Series(np.asarray(family))
    prior, counts = family_weights(cfg), family.value_counts()
    w = family.map(lambda f: prior[f] / counts[f]).to_numpy(float)
    return w * len(w) / w.sum()


# ----------------------------------------------------------------------------------- predictions

def save_predictions(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez(tmp, **{k: v.astype(str) if v.dtype == object else v for k, v in arrays.items()})   # loadable without pickle
    tmp.replace(path)


def load_classifier(cfg: dict, model: str, seed: int | None = None) -> pd.DataFrame | None:
    """case_id-indexed frame: split, one posterior column per class; None if not trained yet."""
    path = unit_dir(cfg, "classify", model, seed=seed) / "predictions.npz"
    if not path.exists():
        return None
    z = np.load(path)
    df = pd.DataFrame(z["posterior"], columns=list(z["classes"]), index=pd.Index(z["case_id"], name="case_id"))
    df.insert(0, "split", z["split"])
    return df


def load_estimator(cfg: dict, family: str, model: str, seed: int | None = None) -> pd.DataFrame | None:
    """case_id-indexed frame of theta_hat, one column per target; None if not trained yet."""
    path = unit_dir(cfg, "estimate", model, family, seed) / "predictions.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return pd.DataFrame(z["theta_hat"], columns=list(z["targets"]), index=pd.Index(z["case_id"], name="case_id"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
