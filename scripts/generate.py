#!/usr/bin/env python3
# scripts/generate.py
"""Generate point clouds for one RunConfig's process, saving under
data/<process>/{clouds.pkl, adversarial_clouds.pkl, cloud_generation_manifest.yaml}.

Usage:
    python scripts/generate.py configs/runs/thomas_dtm_k5_betti_cnn.yaml
    python scripts/generate.py configs/runs/foo.yaml --set process.seed=1 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import load_config
from cloudforger.core.design import CloudDesign
from cloudforger.core.io import dump_pickle
from cloudforger.core.records import cloud_to_record
from cloudforger.paths import DEFAULT_DATA_ROOT, DataPaths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="RunConfig YAML path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="path.to.field=value")
    parser.add_argument("--force", action="store_true", help="regenerate even if clouds.pkl already exists")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    data_paths = DataPaths(cfg.process.name, root=cfg.data_root or DEFAULT_DATA_ROOT)

    out_clouds = data_paths.clouds()
    if out_clouds.exists() and not args.force:
        print(f"{out_clouds} already exists; skipping (pass --force to regenerate).")
        return

    design = CloudDesign.build(
        cfg.process.name,
        seed=cfg.process.seed,
        design=cfg.process.design,
        adversarial=cfg.process.adversarial,
    )
    print(
        f"Generating {cfg.process.name}: {len(design.train_test_design)} train/test + "
        f"{len(design.adversarial_design)} adversarial clouds ..."
    )
    train_clouds, adversarial_clouds = design.generate()

    dump_pickle(out_clouds, [cloud_to_record(c) for c in train_clouds])
    dump_pickle(data_paths.clouds(adversarial=True), [cloud_to_record(c) for c in adversarial_clouds])

    manifest = design.manifest(train_clouds, adversarial_clouds)
    manifest_path = data_paths.manifest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    print(f"Saved {len(train_clouds)} train/test clouds -> {out_clouds}")
    print(f"Saved {len(adversarial_clouds)} adversarial clouds -> {data_paths.clouds(adversarial=True)}")
    print(f"Saved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
