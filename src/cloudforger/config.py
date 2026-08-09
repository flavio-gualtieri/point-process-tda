# src/cloudforger/config.py
"""RunConfig: the single config schema driving every pipeline stage
(generate/featurize/train/evaluate). Loaded from a YAML file, optionally
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
    name: str  # registry key: thomas, nested_thomas, matern, poisson, inhom_thomas
    seed: int = 0
    design: dict[str, Any] = field(default_factory=dict)
    adversarial: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)  # fixed (non-swept) constructor kwargs, e.g. nested_thomas's parent process


@dataclass
class FiltrationConfig:
    name: str  # registry key: rips, dtm
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BifiltrationConfig:
    name: str  # registry key: dtm_bifiltration
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureConfig:
    name: str  # registry key: betti_curve, persistence_image, persistence_entropy
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodConfig:
    name: str  # registry key: betti_cnn, pi, fusion, pi_multik, pi_multik_fusion, pi_multik_scaleconv, vihrs, mincontrast, palm, ...
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    process: ProcessConfig
    filtration: list[FiltrationConfig] = field(default_factory=list)  # empty for raw_pc/vihrs/mincontrast/palm
    bifiltration: list[BifiltrationConfig] = field(default_factory=list)  # empty for raw_pc/vihrs/mincontrast/palm
    features: list[FeatureConfig] = field(default_factory=list)
    method: MethodConfig | None = None
    seeds: list[int] = field(default_factory=lambda: [0])
    target_label_names: list[str] | None = None  # None -> use every label the process/method exposes
    log_label_names: list[str] | None = None
    use_covariates: bool = False
    use_adversarial: bool = True
    data_root: str | None = None  # None -> paths.DEFAULT_DATA_ROOT
    results_root: str | None = None  # None -> paths.DEFAULT_RESULTS_ROOT

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        d = dict(d)

        process_raw = d.pop("process")
        process = ProcessConfig(**process_raw)

        filtration_raw = d.pop("filtration", [])
        if isinstance(filtration_raw, dict):
            filtration_raw = [filtration_raw]
        filtration = [f if isinstance(f, FiltrationConfig) else FiltrationConfig(**f) for f in filtration_raw]

        features_raw = d.pop("features", [])
        features = [
            f if isinstance(f, FeatureConfig)
            else FeatureConfig(**f) if isinstance(f, dict)
            else FeatureConfig(name=f)
            for f in features_raw
        ]

        method_raw = d.pop("method", None)
        method = None
        if method_raw is not None:
            method = method_raw if isinstance(method_raw, MethodConfig) else MethodConfig(**method_raw)

        return cls(process=process, filtration=filtration, features=features, method=method, **d)


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
