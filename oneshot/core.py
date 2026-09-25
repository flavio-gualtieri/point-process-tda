"""What every oneshot step shares: config, the row table, targets, regime coordinates, weights and
the prediction contract.

Rows are every bank pattern of every configured family, in (config family order, manifest order) --
the order cloudforger's network builders use too -- and are identified by case_id everywhere.

Prediction contract (one predictions.npz per trained unit, keyed by case_id, never by row order):
    classifier   case_id, split, posterior (P, K) float32, classes (K,)
    estimator    case_id, split, theta_hat (P, T) float64 (natural scale), targets (T,), family
Classifiers cover val + test rows, plus out-of-fold train rows when the learner supports it.
Estimators cover val + test rows of EVERY family, so a pipeline can look up whatever the classifier
routes to them. Module names avoid cascade/'s, since both directories sit on sys.path.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cascade"))

from cloudforger.simulation.bank import DATA as BANK        # noqa: E402
from cloudforger.simulation.split import split_of           # noqa: E402

DEFAULT_CONFIG = ROOT / "oneshot" / "configs" / "default.yaml"
RESULTS = ROOT / "oneshot" / "results"
DATA = ROOT / "data" / "oneshot"

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

# Regime coordinates beyond cascade/regime.py's DERIVED (which covers the original five families).
COORDINATES = {
    "ring": {"omega": lambda r: r.rho * np.sqrt(r.kappa),              # ring overlap
             "jitter": lambda r: r.sigma / r.rho},
    "matern1": {"core": lambda r: r.R * np.sqrt(r.nbar),
                "zeta": lambda r: r.R * r.nbar},
    "cell": {"lattice": lambda r: 1 - 1 / (r.k - 1)},                 # P(exactly one point per cell)
}


# ---------------------------------------------------------------------------------------- config

def load_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())


def config_arg(parser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="default: %(default)s")


def run_dir(cfg: dict, *parts: str) -> Path:
    return RESULTS.joinpath(cfg["name"], *parts)


def unit_dir(cfg: dict, task: str, model: str, family: str | None = None) -> Path:
    """oneshot/results/<run>/classify/<model>/ or .../estimate/<family>/<model>/."""
    return run_dir(cfg, task, *([family] if family else []), model)


def save_config(cfg_path: str | Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out / "config.yaml")


def estimated_families(cfg: dict) -> list[str]:
    """Families with a learned estimator (everything but poisson)."""
    return [f for f in cfg["families"] if f != "poisson"]


# ------------------------------------------------------------------------------------------ rows

def rows(cfg: dict) -> pd.DataFrame:
    """Every bank row of every configured family, indexed by case_id, with `split` and parameters."""
    parts = [pd.read_csv(BANK / f / "manifest.csv") for f in cfg["families"]]
    out = pd.concat(parts, ignore_index=True)
    out["split"] = split_of(out.theta.to_numpy())
    out = out[out.theta % cfg.get("thin", 1) == 0]               # every k-th theta, in every split
    return out.set_index("case_id")


def coordinate(r: pd.DataFrame, family: str, name: str) -> np.ndarray:
    """A named regime coordinate of one family's rows: COORDINATES, cascade's DERIVED, or a column."""
    from regime import DERIVED                                  # cascade/regime.py
    fn = COORDINATES.get(family, {}).get(name) or DERIVED.get(family, {}).get(name)
    return np.asarray(fn(r) if fn else r[name], float)


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


def load_classifier(cfg: dict, model: str) -> pd.DataFrame | None:
    """case_id-indexed frame: split, one posterior column per class; None if not trained yet."""
    path = unit_dir(cfg, "classify", model) / "predictions.npz"
    if not path.exists():
        return None
    z = np.load(path)
    df = pd.DataFrame(z["posterior"], columns=list(z["classes"]), index=pd.Index(z["case_id"], name="case_id"))
    df.insert(0, "split", z["split"])
    return df


def load_estimator(cfg: dict, family: str, model: str) -> pd.DataFrame | None:
    """case_id-indexed frame of theta_hat, one column per target; None if not trained yet."""
    path = unit_dir(cfg, "estimate", model, family) / "predictions.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return pd.DataFrame(z["theta_hat"], columns=list(z["targets"]), index=pd.Index(z["case_id"], name="case_id"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
