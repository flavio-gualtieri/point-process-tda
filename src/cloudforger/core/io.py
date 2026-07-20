# src/cloudforger/core/io.py
"""Generic pickle/YAML/label-stacking helpers shared across the pipeline.

Promoted from scripts/processing/params/pipeline_lib/io.py -- keeps the
path-agnostic helpers (pickling, YAML, label stacking) that are genuinely
reusable; the old orchestrator-specific config-validation helpers
(normalize_int_list/require_format/resolve_path) stay behind, since
RunConfig (see cloudforger.config) replaces that validation approach.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(path: Path, payload: Any) -> None:
    path = Path(path)
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
    """Stack each item's params dict into a (labels, label_names, params)
    triple, requiring every item to share the same parameter keys (in the
    same order) so the resulting label matrix's columns are well-defined."""
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


def intersect_seeds(seed_arrays: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """N-way intersection of seed arrays, e.g. joining several DTM-k feature
    caches that each survived the empty-diagram filter with a different
    seed subset (see pi_multik). Different arrays can store the same common
    seed set in DIFFERENT relative orders, so a same-set boolean mask is not
    enough to align them element-wise -- this returns actual per-array INDEX
    arrays instead: idx_per_array[j][i] is the row in seed_arrays[j] holding
    common_seeds[i], so seed_arrays[j][idx_per_array[j][i]] names the same
    cloud for every j, at every i. common_seeds is ordered by
    seed_arrays[0]'s own relative order (not globally sorted)."""
    if not seed_arrays:
        raise ValueError("seed_arrays must be non-empty")

    first = np.asarray(seed_arrays[0])
    common = set(int(s) for s in first.tolist())
    for arr in seed_arrays[1:]:
        common &= set(int(s) for s in np.asarray(arr).tolist())

    common_seeds = np.array([int(s) for s in first.tolist() if int(s) in common], dtype=np.int64)

    idx_per_array = []
    for arr in seed_arrays:
        arr = np.asarray(arr)
        pos = {int(s): i for i, s in enumerate(arr)}
        idx_per_array.append(np.array([pos[int(s)] for s in common_seeds], dtype=np.int64))
    return idx_per_array, common_seeds


def align_seeds(a_seeds: np.ndarray, b_seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2-way join of two seed arrays where one population may be a strict
    subset of the other (e.g. vihrs's full population vs. DTM's filtered
    population in fusion) and the two arrays' relative orders need not
    match. Returns (a_idx, b_idx) such that a_seeds[a_idx[i]] ==
    b_seeds[b_idx[i]] for every i, in a_seeds's own relative order
    restricted to the intersection."""
    a_seeds = np.asarray(a_seeds)
    b_pos = {int(s): i for i, s in enumerate(np.asarray(b_seeds))}
    a_idx, b_idx = [], []
    for i, s in enumerate(a_seeds):
        s = int(s)
        if s in b_pos:
            a_idx.append(i)
            b_idx.append(b_pos[s])
    return np.asarray(a_idx, dtype=np.int64), np.asarray(b_idx, dtype=np.int64)
