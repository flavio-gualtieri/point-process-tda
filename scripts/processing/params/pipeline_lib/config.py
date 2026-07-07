# scripts/processing/params/pipeline_lib/config.py
"""Default pipeline configuration and the YAML config loader.

Run parameters (process, dimensions, stage settings, ...) live in a YAML file
such as configs/params/pipeline.yaml rather than in this module. The
DEFAULT_CONFIG below only supplies fallback values for keys a YAML config
chooses to omit, so partial configs stay valid.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from pipeline_lib.io import load_yaml_config, resolve_path

DEFAULT_STAGE_ORDER = (
    "generate_clouds",
    "compute_pairwise",
    "compute_diagrams",
    "vectorize_diagrams",
    "compute_betti",
)

DEFAULT_SPLITS: dict[str, dict[str, str]] = {
    "train_test": {
        "clouds": "clouds.pkl",
        "diagrams": "diagrams.pkl",
        "images": "images.pkl",
        "betti": "betti.pkl",
        "features": "features.pkl",
    },
    "adversarial": {
        "clouds": "adversarial_clouds.pkl",
        "diagrams": "adversarial_diagrams.pkl",
        "images": "adversarial_images.pkl",
        "betti": "adversarial_betti.pkl",
        "features": "adversarial_features.pkl",
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "process": "",
    "dimensions": [2],
    "data_root": "data/params",
    "skip_missing": True,
    "overwrite": True,
    "stages": list(DEFAULT_STAGE_ORDER),
    "splits": DEFAULT_SPLITS,
    "formats": {
        "clouds": "dict",
        "diagrams": "dict",
        "betti": "dict",
        "features": "dict",
    },
    "cloud_generation": {
        "config_path": "configs/params/thomas_cloudgen.yaml",
    },
    "filtration": {
        "maxdim": 1,
        "thresh": None,
    },
    "vectorization": {
        "homology_dims": [0, 1],
        "resolution": 64,
        "sigma": 0.05,
        "calibration_split": "train_test",
    },
    "betti": {
        "homology_dims": (0, 1),
        "grid_size": 128,
        "grid_range": (0.0, 1.0),
        "drop_infinite": True,
        "normalize": False,
    },
    "pair_distance": {
        "n_samples": 5_000,
        "grid_size": 64,
    },
}

DEFAULT_CONFIG_PATH = "configs/params/pipeline.yaml"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_pipeline_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a pipeline run config from YAML, filling gaps with DEFAULT_CONFIG."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    yaml_path = resolve_path(path)
    if yaml_path.exists():
        _deep_update(config, load_yaml_config(yaml_path))
    return config
