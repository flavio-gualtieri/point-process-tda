#!/usr/bin/env python3
# scripts/processing/dv3_classification_bundle.py
"""Merge DV3's per-family clouds (and, where computed, diagrams) into one
multi-family bundle per set, for the model-classification task:

    data/dv3/<set>/_classify/clouds.pkl
    data/dv3/<set>/_classify/<tag>/diagrams.pkl        for every tag present in ALL families

The classification code paths (pi_multik task=classify, vihrs task=classify)
read exactly this layout through DataPaths("_classify", root=data/dv3/<set>),
so no training code needs to know DV3 is per-family on disk.

Class index = position in cloudforger.evaluation.dv3.FAMILIES
(poisson, thomas, nested, matern2, lgcp) in EVERY set, including B, which
has no CSR cells: the bundle's label_names is always the full list, so a
class index means the same family everywhere and a model trained on `train`
scores B/C with the same output layout.

Identity. DV3's per-family `seed` is the within-(set, family) index, so it
collides across families; the merged record's seed is
    class_index * SEED_OFFSET + index
(globally unique inside one merged set), and every record keeps its DV3
case_id, split, process and params, so predictions join back to the
manifests and regimes exactly. Each record also carries `label` (its class
index), which vihrs reads when no diagram bundle is consulted.

Usage:
    python scripts/processing/dv3_classification_bundle.py                 # every set, clouds + all diagram tags
    python scripts/processing/dv3_classification_bundle.py --sets train A  # a subset
    python scripts/processing/dv3_classification_bundle.py --clouds-only   # skip diagrams (e.g. before PH is computed)
    python scripts/processing/dv3_classification_bundle.py --force         # rebuild even if up to date

Re-run after scripts/processing/dv3_diagrams.py merges new tags; bundles
already up to date (same source files, same sizes/mtimes) are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.core.io import dump_pickle, load_pickle  # noqa: E402
from cloudforger.evaluation.dv3 import ALL_SETS, DEFAULT_DV3_ROOT, FAMILIES, data_paths  # noqa: E402

GROUP = "_classify"
# B/C indices encode (cell, level, rep) and reach ~2e6 (generation/seeding.py
# index_B/index_C), so the offset must clear that comfortably.
SEED_OFFSET = 100_000_000
TAGS = ("rips", "dtm_k5", "dtm_k10", "dtm_k15")
STAMP = ".sources.json"


def _stamp(paths: list[Path]) -> dict[str, list[float]]:
    return {str(p): [p.stat().st_size, p.stat().st_mtime] for p in paths}


def _up_to_date(out: Path, sources: list[Path]) -> bool:
    stamp = out.with_name(out.name + STAMP)
    return out.exists() and stamp.exists() and json.loads(stamp.read_text()) == _stamp(sources)


def _write(out: Path, payload, sources: list[Path]) -> None:
    dump_pickle(out, payload)
    out.with_name(out.name + STAMP).write_text(json.dumps(_stamp(sources)))


def merge_clouds(set_: str, root: Path, force: bool) -> list[str]:
    fams = [f for f in FAMILIES if data_paths(set_, f, root).clouds().exists()]
    if not fams:
        print(f"[{set_}] no family clouds under {root / set_} -- skipping")
        return []
    sources = [data_paths(set_, f, root).clouds() for f in fams]
    out = data_paths(set_, GROUP, root).clouds()
    if not force and _up_to_date(out, sources):
        print(f"[{set_}] clouds up to date ({', '.join(fams)})")
        return fams
    merged = []
    for fam, src in zip(fams, sources):
        label = FAMILIES.index(fam)
        for rec in load_pickle(src):
            rec = dict(rec)
            if not 0 <= int(rec["seed"]) < SEED_OFFSET:
                raise ValueError(f"{src}: index {rec['seed']} does not fit under SEED_OFFSET={SEED_OFFSET}")
            rec["seed"] = label * SEED_OFFSET + int(rec["seed"])
            rec["label"] = label
            merged.append(rec)
    _write(out, merged, sources)
    counts = {f: sum(r["process"] == f for r in merged) for f in fams}
    print(f"[{set_}] clouds -> {out}  {counts}")
    return fams


def merge_diagrams(set_: str, fams: list[str], root: Path, force: bool) -> None:
    for tag in TAGS:
        sources = [data_paths(set_, f, root).process_dir / tag / "diagrams.pkl" for f in fams]
        missing = [f for f, p in zip(fams, sources) if not p.exists()]
        if missing:
            print(f"[{set_}] {tag}: not merged (no diagrams yet for {', '.join(missing)})")
            continue
        out = data_paths(set_, GROUP, root).process_dir / tag / "diagrams.pkl"
        if not force and _up_to_date(out, sources):
            print(f"[{set_}] {tag}: up to date")
            continue
        diagrams, seeds, labels, params = [], [], [], []
        for fam, src in zip(fams, sources):
            b = load_pickle(src)
            label = FAMILIES.index(fam)
            if len(b["diagrams"]) != len(b["seeds"]):
                raise ValueError(f"{src}: {len(b['diagrams'])} diagrams for {len(b['seeds'])} seeds")
            diagrams += list(b["diagrams"])
            seeds += [label * SEED_OFFSET + int(s) for s in b["seeds"]]
            labels += [label] * len(b["seeds"])
            params += list(b.get("params") or [{}] * len(b["seeds"]))
        bundle = {
            "diagrams": diagrams,
            "labels": np.asarray(labels, dtype=np.int64),
            "label_names": list(FAMILIES),
            "seeds": seeds,
            "params": params,
            "process": GROUP,
            "config": {"seed_offset": SEED_OFFSET, "families": fams, "dv": 3},
        }
        _write(out, bundle, sources)
        print(f"[{set_}] {tag}: {len(diagrams)} diagrams -> {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_DV3_ROOT)
    ap.add_argument("--sets", nargs="+", default=list(ALL_SETS), choices=list(ALL_SETS))
    ap.add_argument("--clouds-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    for set_ in args.sets:
        fams = merge_clouds(set_, args.root, args.force)
        if fams and not args.clouds_only:
            merge_diagrams(set_, fams, args.root, args.force)


if __name__ == "__main__":
    main()
