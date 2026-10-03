"""What every paper script shares: which results it reads (paper/paper.yaml), loaders for a run's
stored outputs, the regime coordinate u, and the figure style.

Nothing here trains or scores anything: every display is built from files a pipeline run wrote.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cloudforger.paths import ROOT
from cloudforger.pipeline.core import rows as bank_rows
from cloudforger.pipeline.regime import coordinate

PAPER = ROOT / "paper"
FIGS, TABLES = PAPER / "figs", PAPER / "tables"

FAMILIES = ["poisson", "thomas", "nested", "lgcp", "matern2", "ring", "matern1", "cell"]
NAMES = {"poisson": "Poisson", "thomas": "Thomas", "nested": "Nested", "lgcp": "LGCP", "matern2": "Matérn II",
         "ring": "Ring", "matern1": "Matérn I", "cell": "Cell"}
TEX_NAMES = {**NAMES, "matern2": r"Mat\'ern II", "matern1": r"Mat\'ern I"}


# ------------------------------------------------------------------------------------ the inputs

@lru_cache(maxsize=1)
def cfg() -> dict:
    return yaml.safe_load((PAPER / "paper.yaml").read_text())


def run() -> Path:
    """The pipeline run the paper reports (paper.yaml `run`)."""
    return ROOT / cfg()["run"]


def run_config() -> dict:
    """The run's own config, as it was run (written by compare.py)."""
    return yaml.safe_load((run() / "compare" / "config.yaml").read_text())


def evaluation(name: str) -> Path:
    """An end-to-end evaluation set of the run (paper.yaml `evaluation`)."""
    return run() / cfg()["evaluation"][name]


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def compare_report() -> dict:
    return read_json(run() / "compare" / "report.json")


def cutoffs() -> dict:
    return read_json(run() / "compare" / "cutoffs.json")


@lru_cache(maxsize=1)
def rows() -> pd.DataFrame:
    """Every bank row of every family, indexed by case_id, with `split`."""
    return bank_rows({"families": FAMILIES})


def classifier(model: str) -> pd.DataFrame:
    """case_id-indexed: split + one posterior column per family."""
    z = np.load(run() / "classify" / model / "predictions.npz")
    df = pd.DataFrame(z["posterior"], columns=[str(c) for c in z["classes"]], index=pd.Index(z["case_id"], name="case_id"))
    df.insert(0, "split", z["split"])
    return df


def estimator(family: str, model: str) -> pd.DataFrame:
    z = np.load(run() / "estimate" / family / model / "predictions.npz")
    return pd.DataFrame(z["theta_hat"], columns=[str(c) for c in z["targets"]], index=pd.Index(z["case_id"], name="case_id"))


def u_of(r: pd.DataFrame, family: str, cuts: dict) -> np.ndarray:
    """The regime coordinate u = log x + a log nbar of rows of one family (compare's frozen cutoffs)."""
    c = cuts[family]
    return np.log(coordinate(r, family, c["coordinate"])) + c["nbar_exponent"] * np.log(r.nbar.to_numpy(float))


# ------------------------------------------------------------------------------------- the style

# The reference data-viz palette (light mode, print): categorical slots in fixed order -- only the
# first three are used together, the all-pairs-safe set -- a single-hue blue ramp for magnitude, and
# recessive ink for everything that is not data.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "lines.linewidth": 1.2, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "figure.dpi": 150,
    })


def log_ticks(ax, x: bool = True, y: bool = True) -> None:
    """Readable log axes: n-bar ticks at 100, 200, 500 in plain digits; powers of ten elsewhere; no
    minor-tick labels."""
    from matplotlib.ticker import (FixedLocator, FuncFormatter, LogFormatterSciNotation, LogLocator, NullFormatter,
                                   ScalarFormatter)
    if x:
        ax.xaxis.set_major_locator(FixedLocator([100, 200, 500]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    if y:
        lo, hi = ax.get_ylim()
        if hi / lo < 100:                                       # under two decades: 1-3-10 steps, plain digits
            ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 3.0)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=True))
        ax.yaxis.set_minor_formatter(NullFormatter())


def blues():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blues", BLUE_RAMP)


def save(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / f"{name}.pdf"
    fig.savefig(path)
    print(f"-> {path.relative_to(ROOT)}")
    return path


# ------------------------------------------------------------------------------------ the tables

def fmt_ci(s: dict, d: int = 3) -> str:
    """{est, lo, hi} -> 0.535 {\\scriptsize [0.530, 0.540]}."""
    return f"{s['est']:.{d}f} {{\\scriptsize [{s['lo']:.{d}f}, {s['hi']:.{d}f}]}}"


def tex(name: str) -> str:
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def write_table(name: str, header: list[str], body: list[list[str]], align: str | None = None,
                rules_after: tuple[int, ...] = ()) -> Path:
    """A booktabs tabular, \\input by main.tex inside its own table environment and caption."""
    TABLES.mkdir(parents=True, exist_ok=True)
    align = align or "@{}l" + "r" * (len(header) - 1) + "@{}"
    lines = ["% generated by paper/scripts -- do not edit; rerun python paper/scripts/make.py",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for i, row in enumerate(body):
        lines.append(" & ".join(row) + " \\\\")
        if i in rules_after:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path = TABLES / f"{name}.tex"
    path.write_text("\n".join(lines))
    print(f"-> {path.relative_to(ROOT)}")
    return path
