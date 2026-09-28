"""Every generated figure and table of the paper, from the results paper/paper.yaml names.

    python paper/scripts/make.py             # -> paper/figs/*.pdf, paper/tables/*.tex
    python paper/scripts/make.py f4 t6       # just these
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # the sibling scripts, run as a folder

import f1_examples  # noqa: E402
import f3_confusion  # noqa: E402
import f4_regime  # noqa: E402
import f5_main  # noqa: E402
import tables  # noqa: E402

DISPLAYS = {"f1": f1_examples.main, "f3": f3_confusion.main, "f4": f4_regime.main, "f5": f5_main.main,
            "t3": tables.t3_cost, "t4": tables.t4_classification, "t5": tables.t5_boundary,
            "t6": tables.t6_poisson_gap, "t8": tables.t8_power, "t9": tables.t9_estimation, "t10": tables.t10_ph}


def main(names: list[str]) -> None:
    for name in names or DISPLAYS:
        DISPLAYS[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
