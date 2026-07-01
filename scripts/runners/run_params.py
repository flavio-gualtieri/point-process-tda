# scripts/runners/run_params.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cloudforger.nn.experiments import build_experiment

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "params" / "thomas.yaml"

# file_key (declared by each Experiment) -> pickle filename.
METHOD_FILES = {
    "raw_pc": "clouds.pkl",
    "pi": "images.pkl",
    "pairwise": "features.pkl",
    "betti": "betti.pkl",
}

# Concrete methods expanded by the "all" meta-method.
ALL_METHODS = ["raw_pc", "pi_0", "pi_1", "pairwise", "betti_0", "betti_1"]

# Config keys passed straight through into the runner cfg.
PASS_THROUGH = [
    "task", "process", "batch_size", "n_epochs", "lr",
    "n_points", "embedding_dim", "hidden_dims", "seed",
]


# ---------------------------------------------------------------------------
# Path / layout resolution
# ---------------------------------------------------------------------------

def _dimension_dirname(cfg: dict) -> str:
    dim = cfg.get("dimension", cfg.get("dim", 2))
    if isinstance(dim, str) and dim.endswith("d"):
        return dim
    return f"{int(dim)}d"


def _data_dir(cfg: dict) -> Path:
    base = Path(cfg["data_dir"]) if cfg.get("data_dir") else PROJECT_ROOT / "data"
    return base / "params" / _dimension_dirname(cfg) / cfg["process"]


def _results_dir(cfg: dict) -> Path:
    base = Path(cfg["results_dir"]) if cfg.get("results_dir") else PROJECT_ROOT / "results"
    return base / "params" / _dimension_dirname(cfg) / cfg["process"]


def _dataset_path(cfg: dict, file_key: str) -> Path:
    if cfg.get("dataset_dir"):
        return Path(cfg["dataset_dir"]) / METHOD_FILES[file_key]
    return _data_dir(cfg) / METHOD_FILES[file_key]


def _output_dir(cfg: dict, subdir: str) -> Path:
    if cfg.get("output_dir"):
        return Path(cfg["output_dir"])
    return _results_dir(cfg) / subdir


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def run(cfg: dict):
    method = cfg["method"]

    if method == "all":
        results = {}
        for m in ALL_METHODS:
            print(f"\n========== method: {m} ==========")
            results[m] = run({**cfg, "method": m})
        print("\nAll methods done.")
        return results

    if method in ("pi", "betti"):
        results = {}
        for dim in (0, 1):
            dim_method = f"{method}_{dim}"
            dim_cfg = {**cfg, "method": dim_method}
            if cfg.get("output_dir"):
                dim_cfg["output_dir"] = str(Path(cfg["output_dir"]) / dim_method)
            print(f"\n========== method: {dim_method} ==========")
            results[dim] = run(dim_cfg)
        return results

    experiment = build_experiment(cfg)
    dataset_path = _dataset_path(cfg, experiment.file_key)
    output_dir = _output_dir(cfg, experiment.subdir)
    return experiment.run(dataset_path, output_dir)


# ---------------------------------------------------------------------------
# Config / CLI layer
# ---------------------------------------------------------------------------

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


def _as_dim_list(dims) -> list[int]:
    if isinstance(dims, (list, tuple)):
        return [int(d) for d in dims]
    return [int(dims)]


def _resolve_dims(config: dict, override) -> list[int]:
    if override is not None:
        return _as_dim_list(override)
    if config.get("dims") is not None:
        return _as_dim_list(config["dims"])
    if config.get("dim") is not None or config.get("dimension") is not None:
        return _as_dim_list(config.get("dim", config.get("dimension")))
    raise KeyError("Config must specify 'dims' (an int or a list of ints).")


def build_cfg(config: dict, method: str, dim: int) -> dict:
    """Assemble the per-(method, dim) cfg the runner expects from the config."""
    # Templates may reference any config value plus these convenience aliases.
    fmt = {**config, "method": method, "experiment": method,
           "dim": dim, "dimension": dim}

    cfg = {k: config[k] for k in PASS_THROUGH if k in config}
    cfg["method"] = method
    cfg["dim"] = dim

    # Optional filesystem roots (anchored if relative).
    for key in ("data_dir", "results_dir"):
        if config.get(key):
            cfg[key] = str(_anchor(str(config[key])))

    # Optional explicit templates; otherwise the runner's convention applies.
    if config.get("dataset_path_template"):
        cfg["dataset_dir"] = str(_anchor(_fill(config["dataset_path_template"], **fmt)))
    if config.get("output_dir_template"):
        cfg["output_dir"] = str(_anchor(_fill(config["output_dir_template"], **fmt)))

    return cfg


def main(argv: list[str] | None = None):
    import yaml  # lazy: only the CLI path needs PyYAML

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
    parser.add_argument(
        "--dims", nargs="+", type=int, default=None,
        help="Override the config's dims (e.g. --dims 2 3).",
    )
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    methods = args.methods if args.methods is not None else config["methods"]
    dims = _resolve_dims(config, args.dims)

    print(f"Config:  {args.config}")
    print(f"Process: {config['process']} | task: {config.get('task')}")
    print(f"Methods: {methods}")
    print(f"Dims:    {dims}")

    summary: dict = {}
    for method in methods:
        for dim in dims:
            cfg = build_cfg(config, method, dim)
            target = cfg.get("output_dir", "<convention>")
            print(f"\n######### {method} | dim={dim}  ->  {target} #########")
            summary.setdefault(method, {})[dim] = run(cfg)

    print("\nAll requested (method, dim) runs finished.")
    return summary


if __name__ == "__main__":
    main()