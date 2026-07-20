# scripts/processing/params/pipeline.py
"""CLI entry point for the params dataset pipeline.

Stage logic lives in pipeline_lib/ (one module per stage plus shared
config/io/records helpers); this file only handles argument parsing, SLURM
array dispatch, and loading the YAML run config before handing off to
pipeline_lib.orchestrator.ParamsPipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline_lib.config import DEFAULT_CONFIG_PATH, DEFAULT_STAGE_ORDER, load_pipeline_config
from pipeline_lib.io import normalize_int_list
from pipeline_lib.orchestrator import ParamsPipeline


# ---------------------------------------------------------------------------
# Execution mode
# ---------------------------------------------------------------------------
# Controls how the dimension list is dispatched:
#   "auto"  -> run as a SLURM array (one dimension per task) iff the env var
#              SLURM_ARRAY_TASK_ID is present; otherwise run the full list
#              serially. This is rsync-safe: the same file behaves correctly
#              on the cluster and locally with no edits.
#   "hpc"   -> require SLURM_ARRAY_TASK_ID and run exactly one dimension.
#   "local" -> always run the full dimension list serially, ignoring SLURM.
RUN_MODE = "local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the params processing pipeline.")
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML run config (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        help="Dimensions to run, e.g. --dimensions 2 3 5. Overrides the config file.",
    )
    parser.add_argument("--process", help="Process name, e.g. thomas. Overrides the config file.")
    parser.add_argument(
        "--cloud-config",
        help="Path to a cloud_generation YAML. Overrides cloud_generation.config_path.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=DEFAULT_STAGE_ORDER,
        help="Subset of stages to run, in the order provided.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip stage outputs that already exist.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on missing split inputs instead of skipping them.",
    )
    return parser.parse_args()


def resolve_execution(config: dict[str, Any]) -> None:
    """Restrict this process to a single dimension when running as a SLURM
    array task, while preserving that dimension's original index so the
    per-dimension seed matches a local serial run.

    The mapping is positional: SLURM_ARRAY_TASK_ID == i selects
    config["dimensions"][i]. Keep --array=0-(N-1) in sync with the length of
    the dimension list, and do not reorder the list between runs.
    """
    if RUN_MODE == "local":
        return

    task_id_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_env is None:
        if RUN_MODE == "hpc":
            raise RuntimeError(
                "RUN_MODE='hpc' but SLURM_ARRAY_TASK_ID is not set. Launch via "
                "`sbatch --array=...`, or use RUN_MODE='auto'/'local' to run locally."
            )
        return  # auto + not in an array -> full serial run

    task_id = int(task_id_env)
    full_dims = normalize_int_list(config.get("dimensions", [2]), "dimensions")
    if not 0 <= task_id < len(full_dims):
        raise IndexError(
            f"SLURM_ARRAY_TASK_ID={task_id} is out of range for "
            f"{len(full_dims)} dimension(s) {full_dims}. Use --array=0-{len(full_dims) - 1}."
        )

    config["dimensions"] = [full_dims[task_id]]
    config["dimension_indices"] = [task_id]
    print(
        f"[SLURM] array task {task_id} -> dimension {full_dims[task_id]} "
        f"(seed index {task_id} of {len(full_dims)})"
    )


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)

    if args.dimensions is not None:
        config["dimensions"] = args.dimensions
    if args.process is not None:
        config["process"] = args.process
    if args.cloud_config is not None:
        config.setdefault("cloud_generation", {})["config_path"] = args.cloud_config
    if args.stages is not None:
        config["stages"] = args.stages
    if args.no_overwrite:
        config["overwrite"] = False
    if args.strict:
        config["skip_missing"] = False

    resolve_execution(config)

    ParamsPipeline(config).run()


if __name__ == "__main__":
    main()
