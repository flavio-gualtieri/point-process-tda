# scripts/processing/params/pipeline_lib/orchestrator.py
"""ParamsPipeline: ties the per-stage modules together into one run."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from cloudforger.stats.pair_dist import PairDistanceCDF

from pipeline_lib import clouds, diagrams, images, betti as betti_stage, pairwise, records
from pipeline_lib.config import DEFAULT_SPLITS, DEFAULT_STAGE_ORDER
from pipeline_lib.io import load_yaml_config, normalize_int_list, require_format, resolve_path


class ParamsPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config)
        self.process = str(self.config.get("process", ""))
        if not self.process:
            _cloud_cfg_path = resolve_path(
                self.config.get("cloud_generation", {}).get(
                    "config_path", "configs/params/thomas_cloudgen.yaml"
                )
            )
            if _cloud_cfg_path.exists():
                self.process = str(load_yaml_config(_cloud_cfg_path).get("process", ""))
                self.config["process"] = self.process
        self.dimensions = normalize_int_list(self.config.get("dimensions", [2]), "dimensions")
        # dimension_indices lets a single-dimension run (e.g. one SLURM array
        # task) keep its ORIGINAL position in the full list, so that the seed
        # `base_seed + dim_index * dimension_seed_stride` matches a serial run.
        dim_indices = self.config.get("dimension_indices")
        if dim_indices is None:
            self.dim_indices = list(range(len(self.dimensions)))
        else:
            self.dim_indices = normalize_int_list(dim_indices, "dimension_indices")
            if len(self.dim_indices) != len(self.dimensions):
                raise ValueError(
                    "CONFIG['dimension_indices'] must have the same length as "
                    "CONFIG['dimensions']."
                )
        self.data_root = resolve_path(self.config.get("data_root", "data/params"))
        self.splits = self.config.get("splits", DEFAULT_SPLITS)
        self.skip_missing = bool(self.config.get("skip_missing", True))
        self.overwrite = bool(self.config.get("overwrite", True))
        self.stages = tuple(self.config.get("stages", DEFAULT_STAGE_ORDER))
        self._validate()

    def _validate(self) -> None:
        unknown = set(self.stages) - set(DEFAULT_STAGE_ORDER)
        if unknown:
            raise ValueError(f"Unknown stage(s): {sorted(unknown)}")

        formats = self.config.get("formats", {})
        for key in ("clouds", "diagrams", "betti", "features"):
            require_format(formats.get(key, "dict"), f"formats.{key}")

        for split_name, split_cfg in self.splits.items():
            missing = {"clouds", "diagrams", "images", "betti", "features"} - set(split_cfg)
            if missing:
                raise ValueError(f"Split {split_name!r} is missing file role(s): {sorted(missing)}")

    def base_dir(self, dim: int) -> Path:
        return self.data_root / f"{dim}d" / self.process

    def split_path(self, dim: int, split_name: str, role: str) -> Path:
        return self.base_dir(dim) / self.splits[split_name][role]

    def should_skip_missing(self, path: Path, message: str) -> bool:
        if path.exists():
            return False
        if self.skip_missing:
            print(f"  Skipping: {message}: {path}")
            return True
        raise FileNotFoundError(f"{message}: {path}")

    def should_skip_existing(self, path: Path, message: str) -> bool:
        if self.overwrite or not path.exists():
            return False
        print(f"  Skipping: {message}; output exists: {path}")
        return True

    def prepare_cloud_design(self) -> clouds.CloudDesignBundle:
        bundle, process = clouds.prepare_cloud_design(self.config, self.dimensions, self.process)
        self.process = self.config["process"] = process
        return bundle

    def generate_clouds_for_dim(self, dim_index: int, dim: int, design: clouds.CloudDesignBundle) -> None:
        clouds.generate_clouds_for_dim(
            dim_index,
            dim,
            design,
            self.process,
            train_test_output=self.split_path(dim, "train_test", "clouds"),
            adversarial_output=self.split_path(dim, "adversarial", "clouds"),
            manifest_output=self.base_dir(dim) / "cloud_generation_manifest.yaml",
            overwrite=self.overwrite,
        )

    def compute_diagrams_for_dim(self, dim: int) -> None:
        filtration = diagrams.build_filtration(self.config)
        diagram_format = require_format(self.config.get("formats", {}).get("diagrams", "dict"), "formats.diagrams")

        for split_name in self.splits:
            clouds_path = self.split_path(dim, split_name, "clouds")
            out_path = self.split_path(dim, split_name, "diagrams")
            if self.should_skip_missing(clouds_path, f"[dim={dim}] Missing {split_name} clouds"):
                continue
            if self.should_skip_existing(out_path, f"[dim={dim}] {split_name} diagrams"):
                continue
            print(f"  Split: {split_name}")
            diagrams.compute_diagrams_from_clouds(clouds_path, out_path, filtration, self.process, diagram_format)

    def vectorize_diagrams_for_dim(self, dim: int) -> None:
        vec_cfg = self.config["vectorization"]
        homology_dims = normalize_int_list(vec_cfg.get("homology_dims", [0, 1]), "vectorization.homology_dims")
        resolution = int(vec_cfg.get("resolution", 64))
        sigma = float(vec_cfg.get("sigma", 0.05))
        calibration_split = str(vec_cfg.get("calibration_split", "train_test"))

        calibration_path = self.split_path(dim, calibration_split, "diagrams")
        if self.should_skip_missing(calibration_path, f"[dim={dim}] Missing calibration diagrams"):
            return

        calibration_diagrams, _ = records.load_diagrams(calibration_path)
        imager = images.build_imagers(calibration_diagrams, homology_dims, resolution, sigma)

        for split_name in self.splits:
            diagrams_path = self.split_path(dim, split_name, "diagrams")
            out_path = self.split_path(dim, split_name, "images")
            if self.should_skip_missing(diagrams_path, f"[dim={dim}] Missing {split_name} diagrams"):
                continue
            if self.should_skip_existing(out_path, f"[dim={dim}] {split_name} images"):
                continue
            images.vectorize_diagram_split(diagrams_path, out_path, imager, homology_dims, resolution, dim, split_name)

    def compute_betti_for_dim(self, dim: int) -> None:
        betti = betti_stage.build_betti(self.config)
        curve_format = require_format(self.config.get("formats", {}).get("betti", "dict"), "formats.betti")

        for split_name in self.splits:
            diagrams_path = self.split_path(dim, split_name, "diagrams")
            out_path = self.split_path(dim, split_name, "betti")
            if self.should_skip_missing(diagrams_path, f"[dim={dim}] Missing {split_name} diagrams"):
                continue
            if self.should_skip_existing(out_path, f"[dim={dim}] {split_name} Betti curves"):
                continue
            betti_stage.compute_betti_from_diagrams(diagrams_path, out_path, betti, self.process, curve_format)

    def compute_pairwise_for_dim(self, dim: int) -> None:
        feature_format = require_format(self.config.get("formats", {}).get("features", "dict"), "formats.features")
        pair_cfg = self.config["pair_distance"]
        statistics = [
            PairDistanceCDF(
                n_samples=int(pair_cfg.get("n_samples", 5_000)),
                grid_size=int(pair_cfg.get("grid_size", 64)),
            )
        ]

        for split_name in self.splits:
            clouds_path = self.split_path(dim, split_name, "clouds")
            out_path = self.split_path(dim, split_name, "features")
            if self.should_skip_missing(clouds_path, f"[dim={dim}] Missing {split_name} clouds"):
                continue
            if self.should_skip_existing(out_path, f"[dim={dim}] {split_name} pairwise features"):
                continue
            pairwise.compute_pairwise_features(clouds_path, out_path, statistics, self.process, feature_format)

    def run(self) -> None:
        cloud_design = self.prepare_cloud_design() if "generate_clouds" in self.stages else None

        # Dispatch through a {stage_name: callable} map and iterate self.stages
        # in the order the caller gave (previously this ran a hardcoded
        # if-chain in a fixed order regardless of --stages order).
        stage_runners: dict[str, Any] = {
            "generate_clouds": lambda dim_index, dim: self.generate_clouds_for_dim(dim_index, dim, cloud_design),
            "compute_diagrams": lambda dim_index, dim: self.compute_diagrams_for_dim(dim),
            "vectorize_diagrams": lambda dim_index, dim: self.vectorize_diagrams_for_dim(dim),
            "compute_betti": lambda dim_index, dim: self.compute_betti_for_dim(dim),
            "compute_pairwise": lambda dim_index, dim: self.compute_pairwise_for_dim(dim),
        }

        for dim_index, dim in zip(self.dim_indices, self.dimensions):
            print(f"\n[dim={dim}] Base directory: {self.base_dir(dim)}")

            for stage_name in self.stages:
                print(f"\n  Stage: {stage_name}")
                stage_runners[stage_name](dim_index, dim)

        print("\nDone.")
