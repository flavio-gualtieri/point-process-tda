#!/usr/bin/env python3
# scripts/archive_run.py
"""Move a method's current (untagged) results aside into a named, permanent
archive slot -- results/<process>/<filtration_tag>/<method>/_runs/<tag>/ --
so retraining after a code change doesn't silently overwrite the results you
want to compare against. Replaces the ad hoc "cp -r results/<process>
results/<process>/old" pattern (see results/nested_thomas/old/) with a
per-method, git-metadata-stamped snapshot that scripts/evaluate.py can load
directly via its method@run_tag syntax.

Usage:
    python scripts/archive_run.py configs/runs/nested_thomas/nested_thomas_pi_multik_towers.yaml \\
        --methods pi_multik_towers --tag pre_coordconv_refactor

    python scripts/archive_run.py configs/runs/foo.yaml --methods a b \\
        --tag baseline --note "before removing dropout"

    python scripts/archive_run.py configs/runs/foo.yaml --methods a --tag baseline --copy  # keep the original too

After archiving, retrain normally (results.pt no longer exists at the
default path, so train.py won't skip it) and compare old vs new with:
    python scripts/evaluate.py configs/runs/foo.yaml --methods a a@baseline
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.config import load_config
from cloudforger.data_generation.filtration import REGISTRY as FILTRATION_REGISTRY
from cloudforger.paths import DEFAULT_RESULTS_ROOT, RUN_ARCHIVE_DIR_NAME, ResultsPaths
from cloudforger.provenance import provenance_stamp

# Methods whose results always live under the "raw" filtration tag -- mirrors
# scripts/train.py's FILTRATION_INDEPENDENT_FILE_KEYS + scripts/evaluate.py's
# RAW_TAG_METHODS.
RAW_TAG_METHODS = {"raw_pc", "pairwise", "vihrs", "vihrs_checkpointed", "vihrs_500", "mincontrast", "palm"}


def archive_method(
    results_paths: ResultsPaths,
    filtrations: list,
    method: str,
    tag: str,
    *,
    copy: bool,
    force: bool,
    note: str | None,
) -> None:
    tag_filtrations = [] if method in RAW_TAG_METHODS else filtrations
    src = results_paths.method_dir(tag_filtrations, method)
    if not src.exists():
        print(f"  ! no current results at {src} -- skipping {method!r}")
        return

    # Archive only the live children (seed_* dirs) -- src may already contain
    # a RUN_ARCHIVE_DIR_NAME child from a previous --tag, and dst lives
    # *inside* src (src/_runs/<tag>/...), so moving src as a whole would mean
    # moving a directory into its own descendant.
    live_children = [p for p in src.iterdir() if p.name != RUN_ARCHIVE_DIR_NAME]
    if not live_children:
        print(f"  ! no current (untagged) results under {src} -- skipping {method!r}")
        return

    dst = results_paths.method_dir(tag_filtrations, method, run_tag=tag)
    if dst.exists():
        if not force:
            raise SystemExit(f"{dst} already exists -- pass --force to overwrite, or pick a different --tag.")
        shutil.rmtree(dst)

    dst.mkdir(parents=True)
    for child in live_children:
        target = dst / child.name
        if copy:
            shutil.copytree(child, target) if child.is_dir() else shutil.copy2(child, target)
        else:
            shutil.move(str(child), str(target))

    meta = {
        **provenance_stamp(run_tag=tag),
        "archived_from": str(src),
        "copied": copy,
        "note": note,
    }
    with open(dst / "_archive_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    verb = "Copied" if copy else "Moved"
    print(f"  {verb} {src} -> {dst}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="RunConfig YAML path (process/filtration source)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="path.to.field=value")
    parser.add_argument("--methods", nargs="+", required=True, help="method subdirectories to archive")
    parser.add_argument("--tag", required=True, help="name for this archive, e.g. 'pre_coordconv_refactor'")
    parser.add_argument("--note", default=None, help="free-text note describing this archived variant")
    parser.add_argument("--copy", action="store_true", help="copy instead of move (keeps the current results too)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing archive under this --tag")
    args = parser.parse_args(argv)

    if "@" in args.tag or "/" in args.tag:
        raise SystemExit(f"--tag {args.tag!r} must not contain '@' or '/' (it's used as a directory name).")

    cfg = load_config(args.config, overrides=args.overrides)
    filtrations = [FILTRATION_REGISTRY.build(f.name, **f.params) for f in cfg.filtration]
    results_paths = ResultsPaths(cfg.process.name, root=cfg.results_root or DEFAULT_RESULTS_ROOT)

    for method in args.methods:
        archive_method(results_paths, filtrations, method, args.tag, copy=args.copy, force=args.force, note=args.note)

    compare_methods = " ".join(f"{m} {m}@{args.tag}" for m in args.methods)
    print(f"\nDone. Compare old vs new with:\n  python scripts/evaluate.py {args.config} --methods {compare_methods}")


if __name__ == "__main__":
    main()
