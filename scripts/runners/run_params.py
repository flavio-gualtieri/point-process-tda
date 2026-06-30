#!/usr/bin/env python3
"""Launch the params runner from a YAML config.

Reads a config such as configs/params/thomas.yaml and runs each listed method
(e.g. raw_pc, pairwise, pi_0, betti_0) through ``runner_params.run``, resolving
each run's dataset directory and output directory from the config's path
templates. Persistence tokens carry the homology dim (``pi_0`` = persistence
image of H0, ``betti_1`` = Betti curve of H1).

Usage
-----
    python scripts/runners/run_params.py
    python scripts/runners/run_params.py configs/params/thomas.yaml
    python scripts/runners/run_params.py --methods pi_0 pi_1 betti_0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]          # scripts/runners -> scripts -> root
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))       # import the sibling runner

from runner_params import run as run_method  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "params" / "thomas.yaml"

# Keys passed straight through to the runner cfg (paths/methods handled separately).
PASS_THROUGH = [
    "task", "process", "batch_size", "n_epochs", "lr",
    "n_points", "embedding_dim", "hidden_dims", "seed",
]


def _anchor(path_str: str) -> Path:
    """Resolve a (possibly relative) config path against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _fill(template: str, **kw) -> str:
    try:
        return template.format(**kw)
    except KeyError as exc:
        raise KeyError(
            f"Path template {template!r} references {exc} which wasn't provided; "
            f"available placeholders: {sorted(kw)}."
        ) from None


def build_cfg(config: dict, method: str) -> dict:
    """Assemble the per-method cfg the runner expects from the YAML config."""
    process = config["process"]
    fmt = dict(method=method, process=process)
    dataset_dir = _anchor(_fill(config["dataset_path_template"], **fmt))
    output_dir = _anchor(_fill(config["output_dir_template"], **fmt))

    cfg = {k: config[k] for k in PASS_THROUGH if k in config}
    cfg["method"] = method
    cfg["dataset_dir"] = str(dataset_dir)
    cfg["output_dir"] = str(output_dir)
    return cfg


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config", nargs="?", default=str(DEFAULT_CONFIG),
        help="Path to the YAML config (default: configs/params/thomas.yaml).",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Override the config's methods list (e.g. --methods pi_0 betti_1).",
    )
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    methods = args.methods if args.methods is not None else config["methods"]
    print(f"Config:  {args.config}")
    print(f"Process: {config['process']} | task: {config.get('task')}")
    print(f"Methods: {methods}")

    summary = {}
    for method in methods:
        cfg = build_cfg(config, method)
        print(f"\n######### {method}  ->  {cfg['output_dir']} #########")
        summary[method] = run_method(cfg)

    print("\nAll requested methods finished.")
    return summary


if __name__ == "__main__":
    main()