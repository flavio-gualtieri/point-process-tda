#!/usr/bin/env python3
# scripts/build_classification_data.py
"""Assemble the point-process *classification* dataset by merging the
per-process artefacts already sitting in data/<process>/ into a single
labelled tree under data/classification/.

Every source process contributes one class; the class label is the index of
its *canonical* class name in a fixed, sorted `label_names` list. lgcp and
lgcp_strauss are excluded (Cox / Cox-Gibbs hybrids -- kept out of the discrete
"which cluster process generated this?" task).

MERGED LABELS. Some source processes are not meant to be told apart: `MERGE`
maps such a process dir onto another class name, so e.g. aniso_thomas clouds
carry the thomas label and a model calling an aniso_thomas cloud "thomas"
scores as correct. Seeds still get a *per-source-dir* offset (see below), so
the pi_multik seed-join stays unambiguous even when two dirs share a label.

Merged artefacts (each split -> train_test + adversarial):
  * data/classification/clouds.pkl                -- from data/<p>/clouds.pkl
  * data/classification/<tag>/diagrams.pkl        -- from data/<p>/<tag>/diagrams.pkl
    for every filtration tag common to all classes (dtm_k5, dtm_k10, dtm_k15, rips)

SEED OFFSET. Each source design independently reuses cloud seeds 0..6999
(train) / 100000..100999 (adversarial). pi_multik joins diagrams<->clouds
*by seed* (cloudforger.core.io.intersect_seeds / n_points_by_seed), so a
naive merge would collapse every source dir onto one seed axis. We therefore
remap
    seed  ->  source_dir_index * SEED_OFFSET + seed
identically in clouds and in every diagram bundle (SEED_OFFSET = 1_000_000,
the same idiom legacy generate_clouds.py used for its per-dimension offset).
The offset keys on the source dir, not the class, so MERGE'd dirs sharing a
label still get disjoint seed ranges. The original is always recoverable as
`seed % SEED_OFFSET`.

Output schema mirrors the legacy classify pipeline
(_attic/legacy/scripts/processing/classify/{generate_clouds,compute_diagrams}.py),
cross-checked against the current loaders (cloudforger.training.data,
cloudforger.core.records, cloudforger.experiments.base):

  clouds.pkl : {
      clouds:         list[PointCloud],       # cloud.points / cloud.dimension
      labels:         np.ndarray[int64] (N,), # contiguous class index
      dimensions:     np.ndarray[int64] (N,),
      dimension:      int,
      label_names:    list[str],              # class index -> process name
      process_params: dict, config: dict,     # provenance, not read by training
  }
  <tag>/diagrams.pkl : {
      diagrams:       list[dict],             # per-cloud {diagrams:{0:.,1:.}, params, seed, process, filtration, filtration_params}
      labels:         np.ndarray[int64] (N,),
      dimensions:     np.ndarray[int64] (N,),
      dimension:      int,
      label_names:    list[str],
      seeds:          list[int],              # offset seeds, aligned with clouds.pkl
      filtration:     str, filtration_params: dict,
      process_params: dict, config: dict,
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

# Every process directory under data/ EXCEPT these becomes a class. "classification"
# is this script's own out-dir (a prior run leaves data/classification/clouds.pkl,
# which discover_processes would otherwise mistake for a source process).
EXCLUDE = {"lgcp", "lgcp_strauss", "classification"}

# Source process dir -> canonical class name. A dir listed here contributes its
# clouds/diagrams under another class's label (aniso_thomas is not told apart
# from thomas). The RHS need not itself be a source dir, but here it is.
MERGE = {"aniso_thomas": "thomas"}

# clouds.pkl <-> adversarial_clouds.pkl, diagrams.pkl <-> adversarial_diagrams.pkl
SPLITS = ("train_test", "adversarial")
CLOUD_FILE = {"train_test": "clouds.pkl", "adversarial": "adversarial_clouds.pkl"}
DIAGRAM_FILE = {"train_test": "diagrams.pkl", "adversarial": "adversarial_diagrams.pkl"}

# seed -> class_index * SEED_OFFSET + seed, so the merged seed axis is unique
# across classes (max source seed is 100_999, well under 1e6). Recover the
# original with `seed % SEED_OFFSET`.
SEED_OFFSET = 1_000_000


def discover_processes(data_root: Path) -> list[str]:
    names = sorted(
        p.name
        for p in data_root.iterdir()
        if p.is_dir() and (p / "clouds.pkl").exists() and p.name not in EXCLUDE
    )
    if not names:
        raise SystemExit(f"No usable process directories found under {data_root}/")
    return names


def canonical_classes(processes: list[str]) -> list[str]:
    """Sorted, de-duplicated class names after applying MERGE."""
    return sorted({MERGE.get(p, p) for p in processes})


def common_filtration_tags(data_root: Path, processes: list[str]) -> list[str]:
    """Filtration-tag subdirs holding diagrams.pkl for *every* class."""
    per_process = []
    for name in processes:
        per_process.append({
            d.name for d in (data_root / name).iterdir()
            if d.is_dir() and (d / "diagrams.pkl").exists()
        })
    shared = set.intersection(*per_process) if per_process else set()
    dropped = sorted(set.union(*per_process) - shared) if per_process else []
    if dropped:
        print(f"  (skipping filtration tags not present for all classes: {dropped})")
    return sorted(shared)


def _records(payload):
    return payload["diagrams"] if isinstance(payload, dict) and "diagrams" in payload else payload


def merge_clouds(data_root: Path, processes: list[str], class_names: list[str], split: str) -> dict:
    clouds, labels, dimensions = [], [], []
    process_params: dict[str, dict] = {}
    filename = CLOUD_FILE[split]

    for src_idx, name in enumerate(processes):
        class_idx = class_names.index(MERGE.get(name, name))
        path = data_root / name / filename
        records = _records(load_pickle(path))
        for rec in records:
            cloud = to_pointcloud(rec)
            cloud.seed = src_idx * SEED_OFFSET + int(cloud.seed)
            clouds.append(cloud)
            labels.append(class_idx)
            dimensions.append(cloud.dimension)
        process_params[name] = _class_meta(name, class_idx, records, path)
        print(f"  [{class_idx}] {name:16s} +{len(records):5d} clouds")

    return _assemble(
        {"clouds": clouds}, labels, dimensions, class_names, process_params,
        source=filename, extra_config={},
    )


def merge_diagrams(data_root: Path, processes: list[str], class_names: list[str], tag: str, split: str) -> dict:
    diagrams, labels, dimensions, seeds = [], [], [], []
    process_params: dict[str, dict] = {}
    filename = DIAGRAM_FILE[split]
    filtration = filtration_params = None

    for src_idx, name in enumerate(processes):
        class_idx = class_names.index(MERGE.get(name, name))
        path = data_root / name / tag / filename
        payload = load_pickle(path)
        records = _records(payload)
        for rec in records:
            rec = dict(rec)
            rec["seed"] = src_idx * SEED_OFFSET + int(rec["seed"])
            diagrams.append(rec)
            labels.append(class_idx)
            dimensions.append(2)
            seeds.append(rec["seed"])
        filtration = records[0].get("filtration", tag)
        filtration_params = records[0].get("filtration_params", {})
        process_params[name] = _class_meta(name, class_idx, records, path)
        print(f"  [{class_idx}] {name:16s} +{len(records):5d} diagrams")

    bundle = _assemble(
        {"diagrams": diagrams}, labels, dimensions, class_names, process_params,
        source=f"{tag}/{filename}", extra_config={"filtration_tag": tag},
    )
    bundle["seeds"] = seeds
    bundle["filtration"] = filtration
    bundle["filtration_params"] = filtration_params
    return bundle


def _class_meta(name: str, class_idx: int, records: list, path: Path) -> dict:
    seeds = [int(r["seed"]) for r in records]
    return {
        "class": MERGE.get(name, name),
        "class_index": class_idx,
        "n": len(records),
        "param_names": sorted({k for r in records for k in (r["params"] or {})}),
        "orig_seed_range": [min(seeds), max(seeds)],
        "source": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
    }


def _assemble(base, labels, dimensions, class_names, process_params, *, source, extra_config) -> dict:
    dims = sorted(set(dimensions))
    if len(dims) != 1:
        raise SystemExit(f"Mixed ambient dimensions {dims}; the classify payload is single-dimension.")
    return {
        **base,
        "labels": np.asarray(labels, dtype=np.int64),
        "dimensions": np.asarray(dimensions, dtype=np.int64),
        "dimension": int(dims[0]),
        "label_names": list(class_names),
        "process_params": process_params,
        "config": {
            "task": "classify",
            "source": source,
            "excluded_processes": sorted(EXCLUDE),
            "merged_labels": dict(MERGE),
            "n_classes": len(class_names),
            "n": len(labels),
            "seed_offset": SEED_OFFSET,
            "orig_seed_expr": "seed % seed_offset",
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provenance": provenance_stamp(),
            **extra_config,
        },
    }


def _write(payload: dict, out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite.")
    dump_pickle(out_path, payload)
    classes, counts = np.unique(payload["labels"], return_counts=True)
    n = len(payload.get("clouds", payload.get("diagrams", [])))
    print(f"  saved {n} -> {out_path.relative_to(ROOT)}  "
          f"(classes {dict(zip(classes.tolist(), counts.tolist()))})\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "classification")
    parser.add_argument("--force", action="store_true", help="overwrite existing classification artefacts")
    args = parser.parse_args(argv)

    data_root: Path = args.data_root
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    processes = discover_processes(data_root)
    class_names = canonical_classes(processes)
    tags = common_filtration_tags(data_root, processes)
    print(f"Classes ({len(class_names)}): " + ", ".join(f"{i}={n}" for i, n in enumerate(class_names)))
    if MERGE:
        print("Merged labels: " + ", ".join(f"{k} -> {v}" for k, v in sorted(MERGE.items())))
    print(f"Excluded: {', '.join(sorted(EXCLUDE))}")
    print(f"Filtration tags: {', '.join(tags)}")
    print(f"Seed offset: source_dir_index * {SEED_OFFSET} + seed\n")

    manifest: dict = {
        "task": "classify",
        "label_names": class_names,
        "merged_labels": dict(MERGE),
        "excluded_processes": sorted(EXCLUDE),
        "seed_offset": SEED_OFFSET,
        "filtration_tags": tags,
        "artefacts": {},
    }

    def record(rel: str, payload: dict) -> None:
        classes, counts = np.unique(payload["labels"], return_counts=True)
        manifest["artefacts"][rel] = {
            "n": int(len(payload["labels"])),
            "dimension": payload["dimension"],
            "class_counts": {class_names[int(c)]: int(n) for c, n in zip(classes, counts)},
        }

    for split in SPLITS:
        missing = [p for p in processes if not (data_root / p / CLOUD_FILE[split]).exists()]
        if missing:
            print(f"Skipping clouds/{split}: absent for {missing}")
        else:
            print(f"Merging {CLOUD_FILE[split]} ->")
            payload = merge_clouds(data_root, processes, class_names, split)
            _write(payload, out_dir / CLOUD_FILE[split], args.force)
            record(CLOUD_FILE[split], payload)

        for tag in tags:
            missing = [p for p in processes if not (data_root / p / tag / DIAGRAM_FILE[split]).exists()]
            if missing:
                print(f"Skipping {tag}/{DIAGRAM_FILE[split]}: absent for {missing}")
                continue
            print(f"Merging {tag}/{DIAGRAM_FILE[split]} ->")
            payload = merge_diagrams(data_root, processes, class_names, tag, split)
            _write(payload, out_dir / tag / DIAGRAM_FILE[split], args.force)
            record(f"{tag}/{DIAGRAM_FILE[split]}", payload)

    manifest_path = out_dir / "classification_manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"saved manifest -> {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
