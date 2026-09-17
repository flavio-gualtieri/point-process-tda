# src/cloudforger/config.py
"""RunConfig: the single config schema driving every pipeline stage
(train/evaluate; simulation and featurization have their own configs). Loaded from a YAML file, optionally
overridden with repeatable --set dotted.path=value flags -- generalizes
scripts/runners/params/run_params.py's YAML-defaults-plus-CLI-overrides
convention rather than introducing a new dependency (Hydra/OmegaConf)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProcessConfig:
    name: str  # registry key: poisson, thomas, nested_thomas, matern, strauss, lgcp
    seed: int = 0
    design: dict[str, Any] = field(default_factory=dict)
    adversarial: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)  # fixed (non-swept) constructor kwargs, e.g. nested_thomas's parent process


@dataclass
class FiltrationConfig:
    name: str  # rips, dtm, alpha -- see configs/featurization/config.yaml
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodConfig:
    name: str  # registry key: pi, fusion, pi_multik, pi_multik_fusion, pi_multik_scaleconv, betti_multik, vihrs, mincontrast, palm, ...
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Where a run's clouds come from, and how they are split and evaluated.

    source: legacy (default) keeps the pre-DV3 behaviour exactly: one
    data/<process.name>/clouds.pkl, a random train_val_test_indices(n, seed)
    cut, and one scalar test loss. source: dv3 reads the DV3 products
    instead (cloudforger.evaluation.dv3): trains on data/dv3/train/<group>/
    with the split assigned at generation, and evaluates on each of
    `eval_sets` separately, writing one per-pattern predictions_<set>.npz
    each for scripts/evaluate_regimes.py.

    group is the per-set subdirectory: a family (thomas, nested, matern2,
    lgcp) for parameter estimation, or the merged multi-family bundle
    written by scripts/processing/dv3_classification_bundle.py (default
    name "_classify") for classification."""

    source: str = "legacy"                     # legacy | dv3
    root: str | None = None                    # None -> evaluation.dv3.DEFAULT_DV3_ROOT
    group: str | None = None                   # None -> process.name
    train_set: str = "train"
    eval_sets: list[str] = field(default_factory=lambda: ["A", "B", "C"])
    split_reshuffle_seed: int | None = None    # None -> the pre-registered generation split

    @property
    def is_dv3(self) -> bool:
        return self.source == "dv3"


@dataclass
class RunConfig:
    process: ProcessConfig
    filtration: list[FiltrationConfig] = field(default_factory=list)  # empty for raw_pc/vihrs/mincontrast/palm
    method: MethodConfig | None = None
    seeds: list[int] = field(default_factory=lambda: [0])
    target_label_names: list[str] | None = None  # None -> use every label the process/method exposes
    log_label_names: list[str] | None = None
    use_covariates: bool = False
    use_adversarial: bool = True
    data_root: str | None = None  # None -> paths.DEFAULT_DATA_ROOT
    results_root: str | None = None  # None -> paths.DEFAULT_RESULTS_ROOT
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        d = dict(d)

        process_raw = d.pop("process")
        process = ProcessConfig(**process_raw)

        filtration_raw = d.pop("filtration", [])
        if isinstance(filtration_raw, dict):
            filtration_raw = [filtration_raw]
        filtration = [f if isinstance(f, FiltrationConfig) else FiltrationConfig(**f) for f in filtration_raw]

        method_raw = d.pop("method", None)
        method = None
        if method_raw is not None:
            method = method_raw if isinstance(method_raw, MethodConfig) else MethodConfig(**method_raw)

        data_raw = d.pop("data", None) or {}
        data = data_raw if isinstance(data_raw, DataConfig) else DataConfig(**data_raw)
        if data.source not in ("legacy", "dv3"):
            raise ValueError(f"data.source must be 'legacy' or 'dv3', got {data.source!r}")

        return cls(process=process, filtration=filtration, method=method, data=data, **d)


def _set_by_dotted_path(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = target
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply --set path.to.field=value overrides in place. Value is parsed
    with yaml.safe_load so scalars/lists/bools/null come out as the right
    Python type (e.g. --set method.params.n_epochs=100, --set seeds=[0,1,2])."""
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        _set_by_dotted_path(raw, key, yaml.safe_load(value))
    return raw


def load_config(path: Path | str, overrides: list[str] | None = None) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    raw = apply_overrides(raw, overrides)
    return RunConfig.from_dict(raw)
