"""CSR null tables for the studentised L departure.

    python scripts/departure.py simulate --jobs 10          # every grid n (resumable), then validation patterns
    python scripts/departure.py simulate --n 137 --jobs 10  # one grid n (e.g. one SLURM array task)
    python scripts/departure.py fit                         # -> configs/departure/tables.npz
    python scripts/departure.py validate                    # -> configs/departure/report.json
"""

from __future__ import annotations

import argparse
import json
import time

from cloudforger.departure import config, fit, simulate, validate
from cloudforger.departure.tables import Tables


def cmd_simulate(cfg, args):
    grid = [args.n] if args.n else sorted(cfg.grid(), reverse=True)
    for n in grid:
        if n not in cfg.grid():
            raise SystemExit(f"n={n} is not on the grid")
        t0 = time.time()
        simulate.simulate_n(cfg, int(n), args.jobs)
        print(f"n={n:5d}  {time.time() - t0:7.1f}s", flush=True)
    if not args.n:
        t0 = time.time()
        simulate.simulate_validation(cfg, args.jobs)
        print(f"validation  {time.time() - t0:7.1f}s", flush=True)


def cmd_fit(cfg, args):
    report = {"fit": fit.fit(cfg)}
    config.REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def cmd_validate(cfg, args):
    report = json.loads(config.REPORT.read_text())
    report["validation"] = validate.size(cfg, Tables())
    config.REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["validation"], indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--n", type=int)
    s.add_argument("--jobs", type=int, default=1)
    s.set_defaults(func=cmd_simulate)
    sub.add_parser("fit").set_defaults(func=cmd_fit)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    args = p.parse_args()
    args.func(config.load(), args)


if __name__ == "__main__":
    main()
