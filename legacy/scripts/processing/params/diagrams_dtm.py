# scripts/processing/params/diagrams_dtm.py

from __future__ import annotations

import copy
from typing import Any

from pipeline import resolve_execution
from pipeline_lib.orchestrator import ParamsPipeline
from pipeline_lib.config import load_pipeline_config


CONFIG_PATH = "/Users/qp252676/Desktop/point-process-tda/configs/params/processing/new_features.yaml"

# DTM's k is the density-estimation neighbourhood size -- treat it the way
# run_vihrs.py treats r in an L-function (see DTMFiltration's own
# docstring): sweep a few values rather than committing to one a priori.
K_VALUES = [5, 10, 20]


def run_for_k(base_config: dict[str, Any], k: int) -> None:
    config = copy.deepcopy(base_config)
    config["filtration"]["k"] = k

    # Distinct, k-tagged filenames per sweep value so (a) different k runs
    # don't overwrite each other and (b) DTM output never collides with the
    # existing Rips diagrams.pkl/adversarial_diagrams.pkl.
    for split_name, split_cfg in config["splits"].items():
        prefix = "adversarial_" if split_name == "adversarial" else ""
        split_cfg["diagrams"] = f"{prefix}diagrams_dtm_k{k}.pkl"

    resolve_execution(config)
    pl = ParamsPipeline(config)

    stage_runners: dict[str, Any] = {
        "compute_diagrams": lambda dim_index, dim: pl.compute_diagrams_for_dim(dim),
    }

    for dim_index, dim in zip(pl.dim_indices, pl.dimensions):
        print(f"\n[k={k}] [dim={dim}] Base directory: {pl.base_dir(dim)}")
        for stage_name in pl.stages:
            print(f"\n  Stage: {stage_name}")
            stage_runners[stage_name](dim_index, dim)


def main() -> None:
    base_config = load_pipeline_config(CONFIG_PATH)
    for k in K_VALUES:
        print(f"\n{'#' * 90}\n### DTM diagrams | k={k}\n{'#' * 90}")
        run_for_k(base_config, k)

    print(f"\nAll k values done: {K_VALUES}")
    print(
        "Each k wrote data/params/2d/thomas/diagrams_dtm_k<k>.pkl and "
        "adversarial_diagrams_dtm_k<k>.pkl -- next step is vectorize_diagrams/"
        "compute_betti against these (with similarly k-tagged betti/images "
        "output filenames, for the same collision reason)."
    )


if __name__ == "__main__":
    main()
