"""What every paper script shares: which results it reads (paper/paper.yaml), loaders for a run's
stored outputs, the regime coordinate u, and the figure style.

PAPER_CONFIG names another paper config (default paper/paper.yaml), e.g. paper/paper_v2.yaml for the
v2 bank: its `data` is the bank the run was trained on, and its `outputs` keep its figures, tables and
summary apart from the default's. The families are the run's own (its compare/config.yaml), in order.

Nothing here trains or scores anything: every display is built from files a pipeline run wrote.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_HERE = Path(__file__).resolve().parents[2]
CONFIG = Path(os.environ.get("PAPER_CONFIG", _HERE / "paper" / "paper.yaml")).resolve()
_CFG = yaml.safe_load(CONFIG.read_text())
if _CFG.get("data"):                                            # the bank, before cloudforger.paths reads it
    _data = (_HERE / _CFG["data"]).resolve()
    if os.environ.get("CLOUDFORGER_DATA") and Path(os.environ["CLOUDFORGER_DATA"]).resolve() != _data:
        raise SystemExit(f"{CONFIG.name} reads the bank in {_data}, but CLOUDFORGER_DATA={os.environ['CLOUDFORGER_DATA']}")
    os.environ["CLOUDFORGER_DATA"] = str(_data)

from cloudforger.paths import ROOT  # noqa: E402
from cloudforger.pipeline.core import rows as bank_rows  # noqa: E402
from cloudforger.pipeline.regime import coordinate  # noqa: E402

PAPER = ROOT / "paper"
_OUT = _CFG.get("outputs", {})
FIGS, TABLES = ROOT / _OUT.get("figs", "paper/figs"), ROOT / _OUT.get("tables", "paper/tables")
RESULTS = ROOT / _OUT.get("results", "results")                 # summary.csv, story.md

FAMILIES = yaml.safe_load((ROOT / _CFG["run"] / "compare" / "config.yaml").read_text())["families"]
NAMES = {"poisson": "Poisson", "thomas": "Thomas", "nested": "Nested", "lgcp": "LGCP", "matern2": "Matérn II",
         "ring": "Ring", "matern1": "Matérn I", "cell": "Cell", "strauss": "Strauss"}
TEX_NAMES = {**NAMES, "matern2": r"Mat\'ern II", "matern1": r"Mat\'ern I"}
PLOT_NAMES = {**NAMES, "matern2": r"Mat$\acute{\rm e}$rn II", "matern1": r"Mat$\acute{\rm e}$rn I"}   # cmr10 has no é


# ------------------------------------------------------------------------------------ the inputs

def cfg() -> dict:
    return _CFG


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


def x_of(r: pd.DataFrame, family: str, cuts: dict) -> np.ndarray:
    """u rescaled so every family's boundaries coincide: 0 at u*(0.5), 1 at u*(0.9), positive on the
    structured side."""
    b = cuts[family]["u_boundary"]
    return (u_of(r, family, cuts) - b["0.5"]) / (b["0.9"] - b["0.5"])


CSR_TEST = "CSR test"


@lru_cache(maxsize=None)
def detected(model: str) -> pd.Series:
    """case_id -> detected, on every test pattern, as scripts/regime.py --size counts it: the CSR test
    (exact-size L test on the stored L(r) - r, rejects at 5%), or a classifier as a test of the cutoffs'
    size -- P(poisson | x) below its size quantile over the val Poisson patterns."""
    r = rows()
    test = r[r.split == "test"]
    if model == CSR_TEST:
        from cloudforger.departure.tables import Tables
        from cloudforger.paths import CURVES
        tables, out = Tables(), pd.Series(False, index=test.index)
        for f in test.family.unique():
            ids = test.index[test.family == f]
            z = np.load(CURVES / f / "fixed" / "curves.npz")
            at = pd.Index(z["case_id"].astype(str)).get_indexer(ids)
            out[ids] = tables.statistic(z["L"][at].astype(np.float64), test.n[ids].to_numpy(float)) > 1
        return out
    size = next(c["size"] for c in cutoffs().values())
    post = classifier(model)
    val = post[(post.split == "val") & (r.family.reindex(post.index) == "poisson").to_numpy()]
    return post.poisson.reindex(test.index) < np.quantile(val.poisson.to_numpy(), size)


# ------------------------------------------------------------------------------------- the style

# Palette. Categorical slots in fixed order, saturated enough to read at print size: ParamNet is always
# SERIES[0]; the classical second-order tools (the L test, minimum contrast on K) share SERIES[1]. A
# single-hue blue ramp for magnitude, and recessive ink for everything that is not data.
SERIES = ["#3355d6", "#ef5b4c", "#12a38f", "#f2a93b"]
BLUE_RAMP = ["#dfe6fb", "#c8d4f8", "#afc1f5", "#94acf0", "#7b97ea", "#6383e3", "#4b6fdc", "#3355d6",
             "#2b49bd", "#243ea3", "#1d3388", "#16286e", "#101e55"]
INK, INK2, MUTED, GRID, AXIS = "#14161f", "#4a4d57", "#9a9ca5", "#e6e7ec", "#b9bbc4"

# The AISTATS page (aistats2027.sty), in inches. Every figure is drawn at the width it is printed at --
# main.tex includes it at \textwidth (figure*) or \columnwidth (figure) -- so the font sizes below are
# the printed sizes (body text is 10pt). The font is Computer Modern, the body text's.
TEXT_WIDTH, COLUMN_WIDTH = 6.75, 3.25


def style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,   # cmr10 has no unicode minus
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "lines.linewidth": 1.4, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "figure.dpi": 150,
    })


def legend_below(ax, labels: list[str], colours: list[str], ncol: int | None = None, y: float = -0.2):
    """A legend of plain colour squares under the axes."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="none") for c in colours]
    return ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(labels),
                     handlelength=0.8, handleheight=0.8, handletextpad=0.4, columnspacing=0.9, borderaxespad=0)


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
                rules_after: tuple[int, ...] = (), groups: list[tuple[str, int]] = ()) -> Path:
    """A booktabs tabular, \\input by main.tex inside its own table environment and caption. `groups`
    (label, span) adds a row of column-group labels above the header, each ruled by a \\cmidrule; an empty
    label spans its columns unlabelled."""
    TABLES.mkdir(parents=True, exist_ok=True)
    align = align or "@{}l" + "r" * (len(header) - 1) + "@{}"
    lines = ["% generated by paper/scripts -- do not edit; rerun python paper/scripts/make.py",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    if groups:
        cells, rules, col = [], [], 1
        for label, span in groups:
            cells.append(f"\\multicolumn{{{span}}}{{c}}{{{label}}}" if label else " & ".join([""] * span))
            if label:
                rules.append(f"\\cmidrule(lr){{{col}-{col + span - 1}}}")
            col += span
        lines += [" & ".join(cells) + " \\\\", " ".join(rules)]
    lines += [" & ".join(header) + " \\\\", "\\midrule"]
    for i, row in enumerate(body):
        lines.append(" & ".join(row) + " \\\\")
        if i in rules_after:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path = TABLES / f"{name}.tex"
    path.write_text("\n".join(lines))
    print(f"-> {path.relative_to(ROOT)}")
    return path
