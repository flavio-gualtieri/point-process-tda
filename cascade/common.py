"""What every stage shares: the config, the data table, the prior, the models and cross-fitting.

A stage's output is always predictions.npz under cascade/results/<run>/<stage>/, keyed by
`case_id` and carrying `split`, so any later stage (or a notebook) joins on case_id and never on
row order.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.simulation.bank import DATA as BANK        # noqa: E402
from cloudforger.simulation.split import split_of           # noqa: E402

import features                                              # noqa: E402

RESULTS = ROOT / "cascade" / "results"
DEFAULT_CONFIG = ROOT / "cascade" / "configs" / "default.yaml"

CLASSES = ("poisson", "clustered", "repulsive")
REGIME = {"poisson": "poisson", "thomas": "clustered", "nested": "clustered", "lgcp": "clustered",
          "matern2": "repulsive"}
GROUPS = {c: [f for f, g in REGIME.items() if g == c] for c in CLASSES}
# The family's own parameters, as cloudforger.training.data.TARGETS (copied rather than imported,
# which would pull in torch). nbar stands in for LGCP's mu_log; see simulate.sampler_kwargs.
TARGETS = {
    "poisson": ["nbar"],
    "thomas": ["kappa", "mu", "sigma"],
    "nested": ["kappa", "mu1", "mu2", "sigma1", "sigma2"],
    "matern2": ["R", "lam_p"],
    "lgcp": ["nbar", "sigma2", "s"],
}


# ---------------------------------------------------------------------------------------- config

def load_config(path: str | Path | None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())


def run_dir(cfg: dict, stage: str, config_path: str | Path | None = None) -> Path:
    """cascade/results/<run>/<stage>/, with the config copied next to the stage's outputs."""
    d = RESULTS / cfg["name"] / stage
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path or DEFAULT_CONFIG, d / "config.yaml")
    return d


def config_arg(parser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="default: %(default)s")


# ------------------------------------------------------------------------------------------ data

def manifest() -> pd.DataFrame:
    """Every bank row of every family, indexed by case_id, with the parameter columns."""
    return pd.concat([pd.read_csv(BANK / f / "manifest.csv") for f in REGIME],
                     ignore_index=True).set_index("case_id")


@dataclass
class Data:
    case_id: np.ndarray
    X: np.ndarray
    columns: list[str]
    rows: pd.DataFrame          # manifest rows aligned with X
    split: np.ndarray           # train | val | test

    @classmethod
    def load(cls) -> Data:
        case_id, X, columns = features.load()
        rows = manifest().loc[case_id]
        return cls(case_id, X, columns, rows, split_of(rows.theta.to_numpy()))

    @property
    def family(self) -> pd.Series:
        return self.rows.family


def save_predictions(path: Path, case_id, split, posterior, classes, **extra) -> None:
    np.savez(path, case_id=np.asarray(case_id), split=np.asarray(split),
             posterior=np.asarray(posterior, np.float32), classes=np.array(classes), **extra)


def load_predictions(path: Path) -> pd.DataFrame:
    """case_id-indexed frame: split, one posterior column per class, `pred` (argmax class)."""
    z = np.load(path)
    classes = list(z["classes"])
    df = pd.DataFrame(z["posterior"], columns=classes, index=pd.Index(z["case_id"], name="case_id"))
    df.insert(0, "split", z["split"])
    df["pred"] = np.array(classes)[z["posterior"].argmax(1)]
    return df


# ----------------------------------------------------------------------------------------- prior

def prior_weight(family: str) -> float:
    """balanced_groups: 1/3 per group, split evenly over the group's families."""
    return 1.0 / (len(CLASSES) * len(GROUPS[REGIME[family]]))


def sample_weights(family: pd.Series) -> np.ndarray:
    """Per-row weights that turn the bank's equal family counts into the prior. Normalized to mean 1
    over the rows given, so they are usable as-is for fitting and for weighted averages."""
    counts = family.value_counts()
    w = family.map(lambda f: prior_weight(f) / counts[f]).to_numpy()
    return w * len(w) / w.sum()


# ---------------------------------------------------------------------------------------- models

def make_classifier(name: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_leaf_nodes=31,
                                              l2_regularization=1.0, early_stopping=False,
                                              random_state=0)
    raise SystemExit(f"unknown classifier `{name}`")


def make_regressor(name: str):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=False,
                                             random_state=0)
    raise SystemExit(f"unknown regressor `{name}`")


def fit(model, X, y, w):
    """sklearn pipelines take sample_weight under the final step's name."""
    if hasattr(model, "steps"):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def proba(model, X, n_classes: int) -> np.ndarray:
    """predict_proba with one column per label 0..n_classes-1, including labels absent from the
    model's training data (a fold can miss a rare class)."""
    out = np.zeros((len(X), n_classes))
    out[:, model.classes_] = model.predict_proba(X)
    return out


def crossfit(name: str, X, y, w, theta, folds: int, seed: int = 0) -> np.ndarray:
    """Out-of-fold class probabilities, folds grouped by theta so replicates never straddle a fold."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(theta)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % folds))
    fold = np.array([fold_of[t] for t in theta])
    n_classes = int(y.max()) + 1
    out = np.zeros((len(y), n_classes))
    for k in range(folds):
        held = fold == k
        out[held] = proba(fit(make_classifier(name), X[~held], y[~held], w[~held]), X[held], n_classes)
    return out


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=float))
