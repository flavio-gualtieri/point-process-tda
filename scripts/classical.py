#!/usr/bin/env python3
"""Summary-function curves for the simulated patterns (configs/classical/config.yaml).

    python scripts/classical.py                                  # every family x grid (resumable)
    python scripts/classical.py --family thomas --tag fixed       # one family on one grid

Writes data/classical/<family>/<tag>/curves.npz. Fixed-length curves, so there is no shard or merge
step: a family is about a minute and the whole sweep about six. Diagrams get sharded because Rips is
CPU-hours; these do not.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.classical.curves import Config, families, run, tag   # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", help="default: every family under data/simulation")
    p.add_argument("--tag", help="default: every grid in the config (fixed, sqrtn_u2)")
    args = p.parse_args(argv)

    cfg = Config.load()
    fams = [args.family] if args.family else families()
    tags = [args.tag] if args.tag else [tag(s) for s in cfg.grids]

    for family in fams:
        for tag_ in tags:
            t0 = time.time()
            path = run(family, tag_, cfg)
            print(f"{family:8s} {tag_:12s} {time.time() - t0:6.1f}s  -> {path}", flush=True)


if __name__ == "__main__":
    main()
