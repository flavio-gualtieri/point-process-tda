"""Learners: what turns a model entry of the config into fitted predictions.

Every learner is a class with the same four methods over POSITIONAL indices into the row table
(core.rows), so train.py never needs to know which one it is running:

    Learner(cfg, name, rows)                          load / build the model's input for every row
    .classify(y, w, train, val, predict) -> (len(predict), K) class probabilities
    .crossfit(y, w, train, groups, folds) -> (len(train), K) out-of-fold probabilities, or None
    .regress(targets, train, val, predict) -> (len(predict), T) predictions of log(target)
    .artifacts() -> {filename: object} to save next to the predictions (optional)

y is an int label per row (-1 = not a training example), w a per-row weight (the prior), `val` the
rows a learner may early-stop on. To add a learner: write the class, register it in LEARNERS, and
use `learner: <name>` in a model entry. Options come from the entry's `params`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import inputs


# -------------------------------------------------------------------------------- table learners

class Table:
    """sklearn estimators on concatenated feature tables (oneshot/inputs.py)."""
    defaults: dict = {}

    def __init__(self, cfg: dict, name: str, rows: pd.DataFrame):
        spec = cfg["models"][name]
        self.params = {**self.defaults, **spec.get("params", {})}
        self.rows = rows
        self.X, self.columns = inputs.load(cfg, spec["inputs"], rows)
        self.models = {}

    def make_classifier(self):
        raise NotImplementedError

    def make_regressor(self):
        raise NotImplementedError

    def _fit(self, model, X, y, w):
        if hasattr(model, "steps"):                              # pipelines route weights by step name
            return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
        return model.fit(X, y, sample_weight=w)

    def _proba(self, model, X, k: int) -> np.ndarray:
        out = np.zeros((len(X), k))
        out[:, model.classes_] = model.predict_proba(X)          # a class absent from a fold stays 0
        return out

    def classify(self, y, w, train, val, predict):
        k = int(y.max()) + 1
        model = self._fit(self.make_classifier(), self.X[train], y[train], w[train])
        self.models["classifier"] = model
        return self._proba(model, self.X[predict], k)

    def crossfit(self, y, w, train, groups, folds, seed: int = 0):
        k = int(y.max()) + 1
        uniq = np.unique(groups)
        fold_of = pd.Series(np.random.default_rng(seed).permutation(len(uniq)) % folds, index=uniq)
        fold = fold_of.loc[groups].to_numpy()
        out = np.zeros((len(train), k))
        for f in range(folds):
            held = fold == f
            model = self._fit(self.make_classifier(), self.X[train[~held]], y[train[~held]], w[train[~held]])
            out[held] = self._proba(model, self.X[train[held]], k)
        return out

    def regress(self, targets, train, val, predict):
        Y = np.log(self.rows[targets].to_numpy(float))
        out = np.zeros((len(predict), len(targets)))
        for j, t in enumerate(targets):
            model = self._fit(self.make_regressor(), self.X[train], Y[train, j], None)
            self.models[f"regressor_{t}"] = model
            out[:, j] = model.predict(self.X[predict])
        return out

    def artifacts(self):
        return {"model.joblib": {"models": self.models, "columns": self.columns}}


class HGB(Table):
    """Gradient-boosted trees; NaN features (empty diagrams) handled natively."""
    defaults = {"max_iter": 400, "learning_rate": 0.1, "max_leaf_nodes": 31, "l2_regularization": 1.0,
                "early_stopping": False, "random_state": 0}

    def make_classifier(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**self.params)

    def make_regressor(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(**self.params)


class Linear(Table):
    """Standardised logistic / ridge regression, NaNs imputed by the train median."""
    defaults = {"C": 1.0, "alpha": 1.0, "max_iter": 2000}

    def _pipe(self, final):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), final)

    def make_classifier(self):
        from sklearn.linear_model import LogisticRegression
        return self._pipe(LogisticRegression(C=self.params["C"], max_iter=self.params["max_iter"]))

    def make_regressor(self):
        from sklearn.linear_model import Ridge
        return self._pipe(Ridge(alpha=self.params["alpha"]))


# ------------------------------------------------------------------------------------- networks

class NN:
    """cloudforger's PHNet via cascade/nn.py: curves, persistence images or PersLay, per the entry."""

    def __init__(self, cfg: dict, name: str, rows: pd.DataFrame):
        import nn                                                # cascade/nn.py (torch)
        self.nn = nn
        self.s = {**cfg["nn"], **{k: v for k, v in cfg["models"][name].items() if k != "learner"},
                  **cfg["models"][name].get("params", {})}
        # the network's input is built over every bank row; `pos` maps row-table positions into it
        self.dataset, self.n_tags, self.arm = nn.build(self.s, cfg["families"])
        self.pos = pd.Index(self.dataset.manifest["case_id"].to_numpy(str)).get_indexer(rows.index)
        if (self.pos < 0).any():
            raise SystemExit(f"{name}: {(self.pos < 0).sum()} rows have no network input")
        self.name, self.saved = name, {}

    def classify(self, y, w, train, val, predict):
        import torch
        import torch.nn as tnn
        k = int(y.max()) + 1
        # the loss takes class weights c_k, so class k totals c_k n_k: c_k = mean row weight in class k
        # gives it the prior's total (equal prior -> nn.weighted_ce's 1/count, up to scale)
        cw = np.bincount(y[train], weights=w[train], minlength=k) / np.maximum(np.bincount(y[train], minlength=k), 1)
        loss = tnn.CrossEntropyLoss(weight=torch.tensor(cw / cw.mean(), dtype=torch.float32).to(self.nn.DEVICE))
        yy = np.zeros(len(self.dataset.manifest), np.int64)
        yy[self.pos] = np.maximum(y, 0)
        model, info = self.nn.train(self.dataset, self.n_tags, self.s, yy, self.pos[train], self.pos[val], loss, k, self.name)
        self._keep(model, info)
        return self.nn.softmax(self.nn.predict(model, self.dataset, self.s, self.pos[predict]))

    def crossfit(self, *args, **kwargs):
        return None                                              # five GPU refits: not worth it

    def regress(self, targets, train, val, predict):
        import torch.nn as tnn
        from cloudforger.training import data as D
        train, val, predict = self.pos[train], self.pos[val], self.pos[predict]
        yz, norm = D.targets(self.dataset.manifest, targets, train)
        model, info = self.nn.train(self.dataset, self.n_tags, self.s, np.nan_to_num(yz), train, val,
                                    tnn.MSELoss(), len(targets), self.name)
        self._keep(model, info, target_norm=norm)
        return np.log(D.invert_targets(self.nn.predict(model, self.dataset, self.s, predict), norm))

    def _keep(self, model, info, **extra):
        self.saved = {"state_dict": model.state_dict(), "spec": self.s, **extra,
                      "fit": {k: info[k] for k in ("best_epoch", "epochs_run", "best_val_loss")}}

    def artifacts(self):
        return {"model.pt": self.saved}


LEARNERS = {"hgb": HGB, "linear": Linear, "nn": NN}
GPU = {"nn"}                                                     # learners that need a GPU job


def make(cfg: dict, name: str, rows: pd.DataFrame):
    return LEARNERS[cfg["models"][name]["learner"]](cfg, name, rows)


def needs_gpu(cfg: dict, name: str) -> bool:
    return cfg["models"][name]["learner"] in GPU
