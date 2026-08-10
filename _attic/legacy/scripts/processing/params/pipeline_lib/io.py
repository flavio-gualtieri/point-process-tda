# scripts/processing/params/pipeline_lib/io.py
"""Generic path/pickle/YAML helpers shared by every pipeline stage."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from pipeline_lib import PROJECT_ROOT


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_int_list(value: int | Sequence[int], name: str) -> list[int]:
    if isinstance(value, int):
        return [value]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = list(value)
        if all(isinstance(v, int) for v in result):
            return result

    raise TypeError(f"CONFIG[{name!r}] must be an int or a list of ints.")


def require_format(value: str, name: str) -> str:
    if value not in {"dict", "object"}:
        raise ValueError(f"{name} must be 'dict' or 'object'.")
    return value


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_items(path: Path, key: str) -> list[Any]:
    data = load_pickle(path)
    if isinstance(data, dict) and key in data:
        return data[key]
    return data


def load_yaml_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return config


def build_labels(
    items: list[Any],
    params_fn: Callable[[Any], dict[str, Any]],
    item_name: str,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    if not items:
        raise ValueError(f"Cannot build labels from an empty {item_name} list.")

    params = [params_fn(item) for item in items]
    label_names = list(params[0].keys())

    for p in params:
        if list(p.keys()) != label_names:
            raise ValueError(
                f"{item_name.capitalize()} have inconsistent parameter keys; "
                "cannot stack labels."
            )

    labels = np.array([[p[k] for k in label_names] for p in params], dtype=float)
    return labels, label_names, params
