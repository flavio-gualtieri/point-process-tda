#!/usr/bin/env python3
# scripts/build_classification_data.py
"""Assemble the point-process *classification* dataset by merging the
per-process clouds already sitting in data/<process>/ into a single labelled
payload under data/classification/.

Every source process contributes one class; the class label is the index of
the process name in a fixed, sorted `label_names` list. lgcp and lgcp_strauss
are excluded (Cox / Cox-Gibbs hybrids -- kept out of the discrete
"which cluster process generated this?" task).

Output schema (one pickle, matching the legacy classify pipeline that lived
at _attic/legacy/scripts/processing/classify/generate_clouds.py, cross-checked
against the current loaders cloudforger.training.data.PointCloudDataset /
cloudforger.experiments.base.Experiment.extract_labels):

    {
        "clouds":         list[PointCloud],      # cloud.points / cloud.dimension
        "labels":         np.ndarray[int64] (N,),# contiguous class index
        "dimensions":     np.ndarray[int64] (N,),# per-cloud ambient dim
        "dimension":      int,                   # single-dim payload
        "label_names":    list[str],             # class index -> process name
        "process_params": dict,                  # provenance, not read by training
        "config":         dict,                  # provenance
    }

Usage:
    python scripts/build_classification_data.py
    python scripts/build_classification_data.py --force
    python scripts/build_classification_data.py --data-root data --out-dir data/classification
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.core.io import dump_pickle, load_pickle
from cloudforger.core.records import to_pointcloud
from cloudforger.provenance import provenance_stamp

# Every process directory under data/ EXCEPT these two becomes a class.
EXCLUDE = {"lgcp", "lgcp_strauss"}

# The two payloads each source process directory carries, merged independently.
SPLITS = {
    "clouds.pkl": "clouds.pkl",
    "adversarial_clouds.pkl": "adversarial_clouds.pkl",
}


def discover_processes(data_root: Path) -> list[str]:
    names = sorted(
        p.name
        for p in data_root.iterdir()
        if p.is_dir() and (p / "clouds.pkl").exists() and p.name not in EXCLUDE
    )
    if not names:
        raise SystemExit(f"No usable process directories found under {data_root}/")
    return names


def merge_split(
    data_root: Path,
    processes: list[str],
    filename: str,
) -> dict:
    clouds: list = []
    labels: list[int] = []
    dimensions: list[int] = []
    process_params: dict[str, dict] = {}

    for class_idx, name in enumerate(processes):
        path = data_root / name / filename
        records = load_pickle(path)
        if isinstance(records, dict) and "clouds" in records:
            records = records["clouds"]

        for rec in records:
            cloud = to_pointcloud(rec)
            clouds.append(cloud)
            labels.append(class_idx)
            dimensions.append(cloud.dimension)

        param_names = sorted({k for rec in records for k in (rec["params"] or {})})
        process_params[name] = {
            "class_index": class_idx,
            "n_clouds": len(records),
            "param_names": param_names,
            "source": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
        }
        print(f"  [{class_idx}] {name:16s} +{len(records):5d} clouds  ({filename})")

    dims = sorted(set(dimensions))
    if len(dims) != 1:
        raise SystemExit(f"Mixed ambient dimensions {dims}; the classify payload is single-dimension.")

    return {
        "clouds": clouds,
        "labels": np.asarray(labels, dtype=np.int64),
        "dimensions": np.asarray(dimensions, dtype=np.int64),
        "dimension": int(dims[0]),
        "label_names": list(processes),
        "process_params": process_params,
        "config": {
            "task": "classify",
            "source": filename,
            "excluded_processes": sorted(EXCLUDE),
            "n_classes": len(processes),
            "n_clouds": len(clouds),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provenance": provenance_stamp(),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "classification")
    parser.add_argument("--force", action="store_true", help="overwrite existing classification payloads")
    args = parser.parse_args(argv)

    data_root: Path = args.data_root
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    processes = discover_processes(data_root)
    print(f"Classes ({len(processes)}): " + ", ".join(f"{i}={n}" for i, n in enumerate(processes)))
    print(f"Excluded: {', '.join(sorted(EXCLUDE))}\n")

    manifest: dict = {
        "task": "classify",
        "label_names": processes,
        "excluded_processes": sorted(EXCLUDE),
        "splits": {},
    }

    for src_name, out_name in SPLITS.items():
        missing = [p for p in processes if not (data_root / p / src_name).exists()]
        if missing:
            print(f"Skipping {src_name}: absent for {missing}")
            continue

        print(f"Merging {src_name} ->")
        payload = merge_split(data_root, processes, src_name)
        out_path = out_dir / out_name
        if out_path.exists() and not args.force:
            raise SystemExit(f"{out_path} exists; pass --force to overwrite.")
        dump_pickle(out_path, payload)

        classes, counts = np.unique(payload["labels"], return_counts=True)
        manifest["splits"][out_name] = {
            "n_clouds": int(len(payload["clouds"])),
            "dimension": payload["dimension"],
            "class_counts": {processes[int(c)]: int(n) for c, n in zip(classes, counts)},
        }
        print(f"  saved {len(payload['clouds'])} clouds -> {out_path}\n")

    manifest_path = out_dir / "classification_manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"saved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
