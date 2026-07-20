# scripts/processing/params/new_clouds.py

from __future__ import annotations
from typing import Any

from pipeline import resolve_execution
from pipeline_lib.orchestrator import ParamsPipeline
from pipeline_lib.config import load_pipeline_config


CONFIG_PATH = "/Users/qp252676/Desktop/point-process-tda/configs/params/processing/new_features.yaml"

def main() -> None:
    config = load_pipeline_config(CONFIG_PATH)

    resolve_execution(config)

    pl = ParamsPipeline(config)

    cloud_design = pl.prepare_cloud_design()

    stage_runners: dict[str, Any] = {
        "generate_clouds": lambda dim_index, dim: pl.generate_clouds_for_dim(dim_index, dim, cloud_design),
    }

    for dim_index, dim in zip(pl.dim_indices, pl.dimensions):
        print(f"\n[dim={dim}] Base directory: {pl.base_dir(dim)}")

        for stage_name in pl.stages:
            print(f"\n  Stage: {stage_name}")
            stage_runners[stage_name](dim_index, dim)

    print("\nDone.")

if __name__ == "__main__":
    main()