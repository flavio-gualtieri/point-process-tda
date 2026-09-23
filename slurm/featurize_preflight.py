"""Refuse to featurize on top of diagrams left over from an earlier bank.

run_shard() skips a (family, tag) whose diagrams.npz already exists, and merge() returns that file
untouched, so diagrams merged from a previous bank survive a re-run in silence -- the job succeeds
and every downstream step reads the old patterns. This checks each merged file's case_id against
data/bank/<family>/manifest.csv and exits 1 listing what to delete.

    python slurm/featurize_preflight.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from cloudforger.featurization.filtrations import tag
from cloudforger.featurization.sweep import DATA, Config, families
from cloudforger.simulation.bank import DATA as BANK


def main() -> int:
    cfg = Config.load()
    tags = [tag(s) for s in cfg.filtrations]
    stale = []
    for family in families():
        case_id = pd.read_csv(BANK / family / "manifest.csv").case_id.to_numpy(str)
        for tag_ in tags:
            path = DATA / family / tag_ / "diagrams.npz"
            if not path.exists():
                continue
            have = np.load(path)["case_id"]  # npz members load lazily; this reads one array
            if len(have) != len(case_id) or (have != case_id).any():
                stale.append((path, len(have), len(case_id)))
                print(f"stale: {path}  {len(have)} diagrams, bank has {len(case_id)}", file=sys.stderr)
    if stale:
        print(f"\n{len(stale)} merged file(s) predate the current bank. Delete them, and delete\n"
              f"{DATA / 'shards'} as well -- a shard of the old bank has the same name and the same\n"
              "500 patterns as the shard that replaces it, so it cannot be told apart and would be\n"
              "merged in. Then re-submit.", file=sys.stderr)
        return 1
    print(f"preflight ok: {len(families())} families x {len(tags)} filtrations against {BANK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
