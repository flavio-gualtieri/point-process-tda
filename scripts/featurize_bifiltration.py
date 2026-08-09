# scripts/featurize_bifiltration.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import FeatureConfig, RunConfig, load_config
from cloudforger.core.io import dump_pickle, load_pickle
from cloudforger.core.records import diagram_to_record, load_diagrams, to_pointcloud
from cloudforger.features import REGISTRY as FEATURE_REGISTRY
from cloudforger.filtration import REGISTRY as BIFILTRATION_REGISTRY
from cloudforger.filtration.bifiltration import Bifiltration
from cloudforger.filtration.base import Filtration
from cloudforger.paths import DEFAULT_DATA_ROOT, DataPaths
from cloudforger.vectorizers.calibrated import build_calibrated_imager


def run_bifiltration(cfg: RunConfig, bifilt_cfg: FeatureConfig, data_paths: DataPaths, force: bool) -> None:
    bifiltration = BIFILTRATION_REGISTRY[bifilt_cfg.name](**bifilt_cfg.params)
    tag = bifiltration.path_tag()

    clouds_path = data_paths.clouds()
    adv_clouds_path = data_paths.clouds(adversarial=True)

    ...


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="RunConfig YAML path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="path.to.field=value")
    parser.add_argument("--force", action="store_true", help="recompute even if outputs already exist")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    data_paths = DataPaths(cfg.process.name, root=cfg.data_root or DEFAULT_DATA_ROOT)

    run_bifiltration(cfg, bifilt_cfg=cfg.bifiltration, data_paths=data_paths, force=args.force)