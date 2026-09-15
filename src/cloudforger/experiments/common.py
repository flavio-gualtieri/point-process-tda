# src/cloudforger/experiments/common.py
"""Shared machinery for both Experiment (single dataset_path) and
MultiSourceExperiment (several dataset files joined by seed -- fusion,
pi_multik, pi_multik_fusion) subclasses, so every method -- regardless of
how many source files it reads -- writes the identical results.pt /
results.json / model.pt schema and shares the same device-selection /
label-handling / zscore utilities instead of each hand-rolling its own copy."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cloudforger.provenance import append_ledger_entry, provenance_stamp


def prepare_device(seed: int) -> str:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fit_zscore(values: np.ndarray) -> dict[str, float]:
    """Plain (non-log) fit/apply-frozen zscore -- for values that can
    legitimately be zero or negative (persistence entropy, PI pixels),
    where a log-based normalization would be undefined."""
    std = float(values.std())
    return {"mean": float(values.mean()), "std": std if std else 1.0}


def apply_zscore(values: np.ndarray, norm: dict[str, float]) -> np.ndarray:
    return (values - norm["mean"]) / norm["std"]


def select_labels(
    labels: np.ndarray,
    label_names: list[str],
    target_label_names: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    if not target_label_names:
        return labels, label_names

    idx = []
    for name in target_label_names:
        if name not in label_names:
            raise KeyError(f"Requested label {name!r}, available labels are {label_names}")
        idx.append(label_names.index(name))

    return labels[:, idx], list(target_label_names)


def _log_transform_by_name(
    labels: np.ndarray,
    label_names: list[str],
    log_label_names: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    log_set = set(log_label_names or label_names)

    transformed = labels.astype(float).copy()
    transforms = []

    for j, name in enumerate(label_names):
        if name in log_set:
            if np.any(transformed[:, j] <= 0):
                raise ValueError(f"Cannot log-transform non-positive label {name!r}")
            transformed[:, j] = np.log(transformed[:, j])
            transforms.append("log")
        else:
            transforms.append("identity")

    return transformed, transforms


def fit_label_norm(
    labels: np.ndarray,
    label_names: list[str],
    log_label_names: list[str] | None,
) -> dict:
    """Fit log+zscore label stats on `labels` -- pass ONLY the training rows
    here (see apply_label_norm to transform other splits with the result),
    the same fit-on-train/apply-frozen convention as fit_zscore /
    vihrs.fit_log_zscore. Fitting on train+val+test combined would leak test
    statistics into every normalized training label."""
    transformed, transforms = _log_transform_by_name(labels, label_names, log_label_names)

    mean = transformed.mean(axis=0)
    std = transformed.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    return {"mean": mean, "std": std, "transforms": transforms}


def apply_label_norm(
    labels: np.ndarray,
    label_names: list[str],
    label_norm: dict,
) -> np.ndarray:
    """Apply an already-fit label_norm (see fit_label_norm) to `labels` --
    the frozen-stats half of the fit/apply split, used for val/test rows and
    for the adversarial payload."""
    # label_norm["transforms"] is positional ("log"/"identity" per column,
    # written by fit_label_norm); _log_transform_by_name wants a name set,
    # so translate positionally rather than re-deriving it from log_set.
    log_names = [name for name, t in zip(label_names, label_norm["transforms"]) if t == "log"]
    transformed, _ = _log_transform_by_name(labels, label_names, log_names)
    return (transformed - label_norm["mean"]) / label_norm["std"]


def normalize_labels_by_name(
    labels: np.ndarray,
    label_names: list[str],
    log_label_names: list[str] | None,
) -> tuple[np.ndarray, dict]:
    """Fit AND apply in one call -- kept for callers that don't need a
    separate train-only fit (e.g. a one-off script normalizing a single
    array). Prefer fit_label_norm/apply_label_norm for a train/val/test
    split so val and test are transformed with frozen, train-only stats."""
    label_norm = fit_label_norm(labels, label_names, log_label_names)
    return apply_label_norm(labels, label_names, label_norm), label_norm


def save_results(
    output_dir: Path,
    *,
    model: nn.Module | None,
    best_state: dict | None,
    history: dict,
    cfg: dict,
    test_loss: float,
    label_names: list[str],
    label_norm: dict,
    test_loss_per_target: dict[str, float] | None = None,
    adversarial_loss: float | None = None,
    adversarial_loss_per_target: dict[str, float] | None = None,
    adversarial_path: Path | None = None,
    extra_meta: dict | None = None,
    eval_sets: dict[str, dict[str, Any]] | None = None,
) -> None:
    """The one place every method writes results.pt / (optionally) model.pt /
    results.json, regardless of whether it's a single-source Experiment, a
    multi-source one, or a classical baseline with no trained model at all
    (model=None) -- this is what lets evaluate.py have exactly one code path
    no matter which method produced the numbers.

    eval_sets (DV3 runs): {set -> summary} from record_eval_set, one entry
    per evaluation product (A/B/C). The per-pattern predictions themselves
    are already on disk next to results.pt (predictions_<set>.npz); the
    summaries here are only the whole-set headline numbers, so results.json
    stays readable without the regime analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Provenance: exact commit + dirty-tree flag + when + which run_tag
    # archive this belongs to (cfg["run_tag"]/cfg["results_root"] are set by
    # scripts/train.py, not part of the method's own hyperparameters) -- see
    # cloudforger.provenance for why this needs to be recoverable from
    # results.json alone, without loading the torch file.
    stamp = provenance_stamp(run_tag=cfg.get("run_tag"))

    torch.save(
        {
            "model_state": best_state,
            "history": history,
            "config": cfg,
            "test_loss": test_loss,
            "test_loss_per_target": test_loss_per_target,
            "label_names": list(label_names),
            "label_norm": label_norm,
            "label_log_mean": label_norm["mean"],
            "label_log_std": label_norm["std"],
            "label_transforms": label_norm.get("transforms", ["log"] * len(label_names)),
            "adversarial_loss": adversarial_loss,
            "adversarial_loss_per_target": adversarial_loss_per_target,
            "adversarial_path": adversarial_path,
            "eval_sets": eval_sets,
            **stamp,
        },
        output_dir / "results.pt",
    )
    if model is not None:
        torch.save(model.cpu(), output_dir / "model.pt")

    json_payload: dict[str, Any] = {
        "task": cfg.get("task", "params"),
        "method": cfg["method"],
        "test_loss": test_loss,
        "test_loss_per_target": test_loss_per_target,
        "seed": cfg["seed"],
        **stamp,
        "config": cfg,
        **(extra_meta or {}),
    }
    if eval_sets:
        json_payload["eval_sets"] = eval_sets
    if adversarial_loss is not None:
        json_payload["adversarial_loss"] = adversarial_loss
        json_payload["adversarial_loss_per_target"] = adversarial_loss_per_target
        json_payload["adversarial_path"] = str(adversarial_path)

    with open(output_dir / "results.json", "w") as f:
        json.dump(json_payload, f, indent=2, default=str)

    append_ledger_entry(
        output_dir=output_dir,
        results_root=cfg.get("results_root"),
        method=cfg["method"],
        seed=cfg["seed"],
        test_loss=test_loss,
        adversarial_loss=adversarial_loss,
        stamp=stamp,
    )


# ---------------------------------------------------------------------------
# DV3 evaluation-set helpers (cloudforger.evaluation). Shared by every method
# that trains on DV3, so the prediction bundles and the whole-set summaries
# in results.json come out identical whichever method wrote them.
# ---------------------------------------------------------------------------

def dv3_row_meta(clouds_path: Path | str, seeds: Any) -> dict[str, np.ndarray]:
    """{"case_id", "family", "split"} for the clouds identified by `seeds`,
    in that order, read from the DV3 clouds.pkl they came from (joined on the
    record's `seed`, which is unique within one DV3 clouds.pkl -- the
    per-(set, family) index, or the global id in a merged classification
    bundle). Raises if the file does not carry DV3 identity fields, so a
    legacy dataset can never be mistaken for a DV3 one."""
    from cloudforger.core.io import load_pickle  # local: keep this module's import surface small

    payload = load_pickle(clouds_path)
    clouds = payload["clouds"] if isinstance(payload, dict) and "clouds" in payload else payload
    by_seed: dict[int, Any] = {}
    for c in clouds:
        rec = c if isinstance(c, dict) else vars(c)
        by_seed[int(rec["seed"])] = rec
    first = next(iter(by_seed.values()))
    if "case_id" not in first:
        raise KeyError(
            f"{clouds_path} has no DV3 case_id field -- data.source: dv3 needs clouds written by "
            "scripts/generation/dv3.py (or the merged classification bundle built from them)."
        )
    rows = [by_seed[int(s)] for s in np.asarray(seeds).tolist()]
    return {
        "case_id": np.array([str(r["case_id"]) for r in rows], dtype=object),
        "family": np.array([str(r.get("process", "")) for r in rows], dtype=object),
        "split": np.array([str(r.get("split") or "") for r in rows], dtype=object),
    }


def invert_label_norm(std: np.ndarray, label_norm: dict) -> np.ndarray:
    """Standardized predictions back to natural units, for both label_norm
    shapes in this repo: vihrs.fit_log_zscore's {"mean", "std"} (every column
    logged) and fit_label_norm's, which adds per-column "transforms"."""
    std = np.asarray(std, dtype=np.float64)
    z = std * np.asarray(label_norm["std"]) + np.asarray(label_norm["mean"])
    transforms = label_norm.get("transforms") or ["log"] * z.shape[1]
    out = z.copy()
    for j, t in enumerate(transforms):
        if t == "log":
            out[:, j] = np.exp(z[:, j])
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def record_eval_set(
    output_dir: Path | str,
    set_name: str,
    *,
    task: str,
    outputs: np.ndarray,
    truth: np.ndarray,
    case_id: np.ndarray,
    family: np.ndarray,
    names: list[str],
    label_norm: dict | None = None,
    truth_raw: np.ndarray | None = None,
    method: str = "",
    seed: int | None = None,
) -> dict[str, Any]:
    """Write predictions_<set_name>.npz for one DV3 evaluation set and return
    its whole-set summary for results.json.

    task="params": outputs are standardized predictions and truth the
    standardized targets (same train-fit label_norm), so the summary loss is
    exactly the project's normalised loss L. task="classify": outputs are
    logits, truth class indices; the bundle stores the softmax posterior."""
    from cloudforger.evaluation.predictions import save_classify_predictions, save_params_predictions

    outputs = np.asarray(outputs, dtype=np.float64)
    if task == "classify":
        proba = _softmax(outputs)
        truth = np.asarray(truth, dtype=np.int64).reshape(-1)
        path = save_classify_predictions(
            output_dir, set_name, case_id=case_id, family=family, class_names=names,
            proba=proba, truth=truth, method=method, seed=seed,
        )
        pred = proba.argmax(axis=1)
        p_true = np.clip(proba[np.arange(len(truth)), truth], 1e-12, 1.0)
        per_class = {
            name: float((pred[truth == c] == c).mean()) for c, name in enumerate(names) if (truth == c).any()
        }
        return {
            "n": int(len(truth)), "loss": float(-np.log(p_true).mean()),
            "accuracy": float((pred == truth).mean()), "accuracy_per_class": per_class,
            "predictions": path.name,
        }

    truth = np.asarray(truth, dtype=np.float64)
    pred_raw = invert_label_norm(outputs, label_norm) if label_norm is not None else None
    path = save_params_predictions(
        output_dir, set_name, case_id=case_id, family=family, label_names=names,
        pred_std=outputs, truth_std=truth, pred_raw=pred_raw, truth_raw=truth_raw,
        method=method, seed=seed,
    )
    ok = np.isfinite(outputs).all(axis=1)
    mse_t = ((outputs[ok] - truth[ok]) ** 2).mean(axis=0) if ok.any() else np.full(len(names), np.nan)
    return {
        "n": int(len(truth)), "n_ok": int(ok.sum()), "loss": float(np.mean(mse_t)),
        "loss_per_target": dict(zip(names, mse_t.tolist())), "predictions": path.name,
    }


def print_eval_set(tag: str, set_name: str, summary: dict[str, Any]) -> None:
    if "accuracy" in summary:
        print(f"[{tag}] eval set {set_name}: n={summary['n']} | cross-entropy {summary['loss']:.4f} | "
              f"accuracy {summary['accuracy']:.4f}")
    else:
        print(f"[{tag}] eval set {set_name}: n={summary['n']} | loss {summary['loss']:.4f}")


def targets_for_task(
    targets: np.ndarray, train_idx: np.ndarray, n_classes: int, is_classify: bool, tag: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """(label_norm, training targets) for either task, as pi_multik.py does
    inline: class indices verbatim (CrossEntropyLoss) with a label_norm stub,
    or log + z-score fit on train_idx rows only (MSELoss)."""
    from cloudforger.baselines import vihrs  # local: keep this module's import surface small

    if is_classify:
        classes = sorted(int(c) for c in np.unique(targets))
        if classes != list(range(n_classes)):
            raise ValueError(f"[{tag}] expected contiguous class labels 0..{n_classes - 1}, got {classes}.")
        label_norm = {"kind": "classification", "classes": classes, "mean": None, "std": None,
                      "transforms": ["class_index"] * n_classes}
        return label_norm, targets.astype(np.int64)
    label_norm = vihrs.fit_log_zscore(targets[train_idx])
    return label_norm, vihrs.apply_log_zscore(targets, label_norm).astype(np.float32)


def dv3_eval_sets(
    model: nn.Module,
    device: str,
    output_dir: Path,
    eval_paths: dict[str, dict[str, Any]] | None,
    *,
    load_split: Any,
    build_inputs: Any,
    is_classify: bool,
    label_norm: dict,
    label_names: list[str],
    batch_size: int,
    method: str,
    seed: int,
    tag: str,
) -> dict[str, dict[str, Any]]:
    """Score `model` on every DV3 evaluation product and write its
    predictions_<set>.npz -- the method-agnostic tail of pi_multik.py's DV3
    block. load_split(set_name, set_paths) returns the set's split dict
    (targets, seeds, ...); build_inputs(split) returns its (main tensor, extra)
    through the FROZEN train-fit transforms. {set -> summary} for results.json."""
    from torch.utils.data import DataLoader, TensorDataset

    from cloudforger.baselines import vihrs  # local: keep this module's import surface small
    from cloudforger.training.train import predict_outputs

    out: dict[str, dict[str, Any]] = {}
    for set_name, set_paths in (eval_paths or {}).items():
        split = load_split(set_name, set_paths)
        if split is None:
            raise FileNotFoundError(f"[{tag} seed={seed}] eval set {set_name}: diagrams missing -- compute them "
                                    f"(scripts/processing/dv3_diagrams.py) or drop {set_name!r} from data.eval_sets.")
        truth = (split["targets"].astype(np.int64) if is_classify
                 else vihrs.apply_log_zscore(split["targets"], label_norm).astype(np.float32))
        main, extra = build_inputs(split)
        loader = DataLoader(TensorDataset(torch.from_numpy(main), torch.from_numpy(extra), torch.from_numpy(truth)),
                            batch_size=batch_size, shuffle=False)
        meta = dv3_row_meta(set_paths["clouds"], split["seeds"])
        out[set_name] = record_eval_set(
            output_dir, set_name, task="classify" if is_classify else "params",
            outputs=predict_outputs(model, loader, device), truth=truth,
            case_id=meta["case_id"], family=meta["family"], names=list(label_names),
            label_norm=None if is_classify else label_norm,
            truth_raw=None if is_classify else split["targets"], method=method, seed=seed,
        )
        print_eval_set(f"{tag} seed={seed}", set_name, out[set_name])
    return out


def fit_best_val(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module, device: str, cfg: dict[str, Any], is_classify: bool, tag: str,
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor], float]:
    """Best-val-loss training with early stopping -- the loop every
    MultiSourceExperiment here runs; accuracy is tracked under classify.
    Returns (history, best_state, best_val_loss); load best_state before evaluating."""
    from cloudforger.training.train import evaluate, train_one_epoch  # local: keep this module's import surface small

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    if is_classify:
        history["train_acc"], history["val_acc"] = [], []
    best_val_loss, best_state = float("inf"), None
    n_epochs = cfg["n_epochs"]
    patience = cfg.get("early_stopping_patience")
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if is_classify:
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epoch == 1 or epoch % 25 == 0 or epoch == n_epochs:
            msg = f"[{tag}] epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}"
            if is_classify:
                msg += f" | train_acc {train_acc:.4f} | val_acc {val_acc:.4f}"
            print(msg)
        if patience is not None and epochs_no_improve >= patience:
            print(f"[{tag}] early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break
    return history, best_state, best_val_loss


class MultiSourceExperiment(ABC):
    """Sibling to Experiment for methods that need several dataset files
    joined by seed (fusion: vihrs + images; pi_multik/betti_multik: images
    (diagrams) at several DTM k; pi_multik_fusion: vihrs + pi_multik) --
    Experiment.run()'s single dataset_path: Path contract has no hook for
    that join. Same registry (register under the same name via
    nn.experiments.base.register), same output schema (via save_results
    above); different run() shape.

    dataset_paths values are Path or list[Path] (list for pi_multik's/
    betti_multik's per-k diagram files), keyed from a small fixed
    vocabulary the concrete subclass documents (e.g. "clouds", "images")."""

    file_keys: tuple[str, ...]  # declares which dataset_paths keys this method needs, for CLI/config wiring
    supports_dv3: bool = False   # True once run() honours cfg["split"] == "dv3" and eval_paths

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @property
    @abstractmethod
    def subdir(self) -> str:
        """Result subdirectory and default log tag."""

    @property
    def tag(self) -> str:
        return self.subdir

    @abstractmethod
    def run(
        self,
        dataset_paths: dict[str, Any],
        output_dir: Path,
        adversarial_paths: dict[str, Any] | None = None,
        eval_paths: dict[str, dict[str, Any]] | None = None,
    ) -> dict:
        """Train, evaluate, and save results for one seed (self.cfg["seed"]).
        adversarial_paths uses the same keys as dataset_paths, or is None to
        skip adversarial evaluation. Mirrors Experiment.run()'s
        (dataset_path, output_dir, adversarial_path) shape, pluralized.

        eval_paths (DV3 runs, self.cfg["split"] == "dv3"): {set name ->
        dataset_paths-shaped dict} for each evaluation product. A subclass
        that supports DV3 sets supports_dv3 = True and writes one
        predictions_<set>.npz per entry via record_eval_set;
        scripts/train.py refuses DV3 configs for any other subclass rather
        than let it run with no regime output."""
