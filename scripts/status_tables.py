#!/usr/bin/env python
r"""Generate the status document's LaTeX tables from results/testset_summary.json.

    python scripts/status_tables.py

Writes docs/status/tables/*.tex, which docs/status/status.tex \input's. The
tables are generated rather than hand-written so the document cannot drift
from the results: re-run scripts/rescore_results.py, re-run this, recompile.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "testset_summary.json"
OUT = ROOT / "docs" / "status" / "tables"

STRATA = ("below", "weak", "moderate", "strong")
FAMILIES = [("thomas", "Thomas"), ("nested", "Nested"),
            ("matern2", "Mat\\'ern II"), ("lgcp", "LGCP")]

# Display label for each run directory, and the order they appear in.
SUBSETS = [
    ("vihrs", "$L$"), ("vihrs_checkpointed", "$L$"),
    ("vihrs_f", "$F$"), ("vihrs_g", "$G$"), ("vihrs_j", "$J$"),
    ("vihrs_lf", "$L{+}F$"), ("vihrs_lg", "$L{+}G$"), ("vihrs_lj", "$L{+}J$"),
    ("vihrs_fg", "$F{+}G$"), ("vihrs_fj", "$F{+}J$"), ("vihrs_gj", "$G{+}J$"),
    ("vihrs_lfg", "$L{+}F{+}G$"), ("vihrs_lfj", "$L{+}F{+}J$"),
    ("vihrs_lgj", "$L{+}G{+}J$"), ("vihrs_fgj", "$F{+}G{+}J$"),
    ("vihrs_lfgj", "$L{+}F{+}G{+}J$"),
]
PH = [
    ("pi_multik_h0_nocc_norm", "PH image $H_0$"),
    ("pi_multik_h1_nocc_norm", "PH image $H_1$"),
    ("pi_multik_h01_nocc_norm", "PH image $H_0{+}H_1$"),
]
# The r_max = 0.08 variants, paired with the r_max = 0.25 run they compete with.
GRID_PAIRS = {"vihrs": "vihrs_lfgj_r080", "vihrs_checkpointed": "vihrs_lfgj_r080",
              "vihrs_f": "vihrs_f_r080", "vihrs_g": "vihrs_g_r080",
              "vihrs_lfgj": "vihrs_lfgj_r080"}


def load() -> dict:
    if not SUMMARY.exists():
        sys.exit(f"no {SUMMARY} -- run scripts/rescore_results.py first")
    doc = json.loads(SUMMARY.read_text())
    return {Path(r["run"]).relative_to(ROOT / "results").as_posix(): r for r in doc["runs"]}


def pick(runs: dict, task: str, name: str, lower_is_better: bool) -> dict | None:
    """The run for `name` under `task`, preferring whichever radius grid won on
    the test set among the pair -- the same rule the previous document used,
    applied to the same pairs."""
    tag = "raw" if name.startswith("vihrs") else "dtm_k5"
    key = f"dv3_{task}/{tag}/{name}"
    best = runs.get(key)
    alt = runs.get(f"dv3_{task}/raw/{GRID_PAIRS[name]}") if name in GRID_PAIRS else None
    if best and alt:
        better = alt["overall"]["mean"] < best["overall"]["mean"]
        if better == lower_is_better:
            return alt
    return best


def fmt(v: float, sd: float, bold: bool, places: int = 4) -> str:
    s = f"{v:.{places}f} $\\pm$ {sd:.{places}f}"
    return f"\\B{{{s}}}" if bold else s


def params_table(runs: dict) -> str:
    """Family x model x stratum, normalised loss."""
    lines = []
    for fam, label in FAMILIES:
        rows = []
        for name, disp in SUBSETS:
            if name == "vihrs":
                continue  # params runs use vihrs_checkpointed for L
            r = pick(runs, fam, name, lower_is_better=True)
            if r:
                rows.append((disp, r))
        for name, disp in PH:
            r = runs.get(f"dv3_{fam}/dtm_k5/{name}")
            if r:
                rows.append((disp, r))
        best = min(r["overall"]["mean"] for _, r in rows)
        for i, (disp, r) in enumerate(rows):
            head = label if i == 0 else ""
            b = abs(r["overall"]["mean"] - best) < 1e-12
            cells = [fmt(r["overall"]["mean"], r["overall"]["sd"], b)]
            for s in STRATA:
                g = r["strata"][s]
                cells.append(f"\\B{{{g['mean']:.3f}}}" if b else f"{g['mean']:.3f}")
            lines.append(f"{head} & {disp} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    body = "\n".join(lines[:-1])
    head = ("Family & Model & Overall & \\texttt{below} & \\texttt{weak} & "
            "\\texttt{moderate} & \\texttt{strong}\\\\")
    return "\n".join([
        r"\begin{longtable}{@{}ll c cccc@{}}",
        r"\caption{Parameter estimation: normalised loss (lower is better), mean $\pm$ SD "
        r"over 10 seeds, with the breakdown by distance from Poisson. All 15 non-empty "
        r"subsets of $\{L,F,G,J\}$ are shown. \textbf{Bold} marks the single best model "
        r"per family.}",
        r"\label{tab:params}\\",
        r"\toprule", head, r"\midrule", r"\endfirsthead",
        r"\multicolumn{7}{l}{\small\slshape (Table~\ref{tab:params} continued)}\\",
        r"\toprule", head, r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{7}{r}{\small\slshape continued on next page}\\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
        body,
        r"\end{longtable}",
    ])


def classify_table(runs: dict) -> str:
    rows = []
    for name, disp in SUBSETS:
        if name == "vihrs_checkpointed":
            continue
        r = pick(runs, "classify", name, lower_is_better=False)
        if r:
            rows.append((disp, r))
    for name, disp in PH:
        r = runs.get(f"dv3_classify/dtm_k5/{name}")
        if r:
            rows.append((disp, r))
    best = max(r["overall"]["mean"] for _, r in rows)
    out = []
    for disp, r in rows:
        b = abs(r["overall"]["mean"] - best) < 1e-12
        cells = [fmt(r["overall"]["mean"], r["overall"]["sd"], b, places=3)]
        cells += [(f"\\B{{{r['strata'][s]['mean']:.3f}}}" if b
                   else f"{r['strata'][s]['mean']:.3f}") for s in STRATA]
        cells += [f"{r['recall'][f]['mean']:.3f}"
                  for f in ("poisson", "thomas", "nested", "matern2", "lgcp")]
        out.append(f"{disp} & " + " & ".join(cells) + "\\\\")
    return "\n".join([
        r"\begin{table}[ht]", r"\centering\footnotesize",
        r"\begin{tabular}{@{}l c cccc ccccc@{}}", r"\toprule",
        r" & & \multicolumn{4}{c}{By distance from Poisson} & "
        r"\multicolumn{5}{c}{Recall per family}\\",
        r"\cmidrule(lr){3-6}\cmidrule(l){7-11}",
        r"Model & Overall & \texttt{below} & \texttt{weak} & \texttt{mod.} & "
        r"\texttt{strong} & Poisson & Thomas & Nested & Mat\'ern & LGCP\\",
        r"\midrule",
        "\n".join(out),
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Classification accuracy, mean $\pm$ SD over 10 seeds, on the stratified "
        r"test set. All 15 non-empty subsets of $\{L,F,G,J\}$; $L$, $L{+}F{+}G{+}J$, $G$ "
        r"and $F$ report whichever $F/G/J$ radius grid won. \textbf{Bold}: best overall "
        r"model.}",
        r"\label{tab:classify}", r"\end{table}",
    ])


def headline(runs: dict) -> str:
    """The few numbers the prose quotes, as macros, so the text cannot drift."""
    macros = []
    cb = max((r for k, r in runs.items() if k.startswith("dv3_classify/raw/")),
             key=lambda r: r["overall"]["mean"])
    ph = runs["dv3_classify/dtm_k5/pi_multik_h01_nocc_norm"]
    lonly = runs["dv3_classify/raw/vihrs"]
    macros.append(f"\\newcommand{{\\bestclassical}}{{{cb['overall']['mean']:.3f}}}")
    label = dict(SUBSETS).get(Path(cb["run"]).name, Path(cb["run"]).name)
    macros.append(f"\\newcommand{{\\bestclassicalname}}{{{label}}}")
    macros.append(f"\\newcommand{{\\bestph}}{{{ph['overall']['mean']:.3f}}}")
    macros.append(f"\\newcommand{{\\lonly}}{{{lonly['overall']['mean']:.3f}}}")
    for s in STRATA:
        macros.append(f"\\newcommand{{\\ph{s}}}{{{ph['strata'][s]['mean']:.3f}}}")
        macros.append(f"\\newcommand{{\\lonlyx{s}}}{{{lonly['strata'][s]['mean']:.3f}}}")
        macros.append(f"\\newcommand{{\\bestx{s}}}{{{cb['strata'][s]['mean']:.3f}}}")
    return "\n".join(macros)


def main() -> int:
    runs = load()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in [("params", params_table(runs)),
                       ("classify", classify_table(runs)),
                       ("headline", headline(runs))]:
        path = OUT / f"{name}.tex"
        path.write_text(body + "\n")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
