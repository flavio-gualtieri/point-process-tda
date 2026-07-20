# scripts/processing/params/pipeline_lib/pairwise.py
"""Stage: compute_pairwise — pairwise/correlation features from point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cloudforger.core.features import CorrelationFeatures

from pipeline_lib.io import build_labels, dump_pickle, load_items
from pipeline_lib.records import cloud_params, cloud_seed, features_to_record, to_pointcloud


def compute_pairwise_features(
    clouds_path: Path,
    out_path: Path,
    statistics: list[Any],
    process: str,
    feature_format: str,
) -> None:
    clouds = load_items(clouds_path, "clouds")
    labels, label_names, params = build_labels(clouds, cloud_params, "clouds")
    seeds = [cloud_seed(c) for c in clouds]
    stat_params = {s.name: s.params for s in statistics}
    order = sorted(s.name for s in statistics)

    computed: list[CorrelationFeatures] = []
    rows: list[np.ndarray] = []

    print(f"  Computing pairwise features for {len(clouds)} clouds {[s.name for s in statistics]} from {clouds_path} ...")
    for i, cloud in enumerate(clouds):
        pc = to_pointcloud(cloud)
        vectors = {s.name: np.asarray(s.compute(pc)) for s in statistics}
        rows.append(np.concatenate([vectors[name] for name in order]))
        computed.append(
            CorrelationFeatures(
                features=vectors,
                generator_name=pc.generator_name,
                generator_params=pc.generator_params,
                seed=pc.seed,
                statistic_params=stat_params,
            )
        )
        print(f"\r    {i + 1}/{len(clouds)}", end="", flush=True)
    print()

    feature_matrix = np.stack(rows)
    features = [features_to_record(cf) for cf in computed] if feature_format == "dict" else computed

    dump_pickle(
        out_path,
        {
            "features": features,
            "feature_matrix": feature_matrix,
            "feature_order": order,
            "labels": labels,
            "label_names": label_names,
            "params": params,
            "seeds": seeds,
            "process": process,
            "statistic_params": stat_params,
        },
    )
    print(f"  Saved pairwise features for {len(computed)} clouds, feature_matrix {feature_matrix.shape} → {out_path}")
