#!/usr/bin/env python
r"""Generate status_concise.tex's tables from the scored results.

    python scripts/status_concise_tables.py

Writes docs/status/tables/concise_*.tex, which docs/status/status_concise.tex
\input's. Reads results/testset_summary.json for loss and accuracy, and
recomputes detection power from the prediction bundles (it is not in the
summary). Run scripts/rescore_results.py first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cloudforger.evaluation import testset as T  # noqa: E402
from cloudforger.evaluation.testset_scoring import detection_run  # noqa: E402
from status_tables import GRID_PAIRS, SUBSETS, pick  # noqa: E402

SUMMARY = ROOT / "results" / "testset_summary.json"
OUT = ROOT / "docs" / "status" / "tables"
STRATA = T.STRATUM_NAMES
FAMILIES = [("thomas", "Thomas"), ("nested", "Nested"),
            ("matern2", "Mat\\'ern II"), ("lgcp", "LGCP")]
PH = [("pi_multik_h0_nocc_norm", "PH $H_0$"),
      ("pi_multik_h1_nocc_norm", "PH $H_1$"),
      ("pi_multik_h01_nocc_norm", "PH $H_0{+}H_1$")]
PH_BEST = "pi_multik_h01_nocc_norm"
LABEL = dict(SUBSETS)
STRATUM_HEAD = " & ".join(f"\\texttt{{{s}}}" for s in STRATA)


def load() -> dict:
    if not SUMMARY.exists():
        sys.exit(f"no {SUMMARY} -- run scripts/rescore_results.py first")
    doc = json.loads(SUMMARY.read_text())
    return {Path(r["run"]).as_posix().split("results/", 1)[1]: r for r in doc["runs"]}


def classical(runs: dict, task: str, lower: bool) -> dict[str, dict]:
    """Every classical subset for `task`, grid-selected, keyed by display label."""
    out = {}
    for name, disp in SUBSETS:
        if task == "classify" and name == "vihrs_checkpointed":
            continue
        if task != "classify" and name == "vihrs":
            continue
        r = pick(runs, task, name, lower_is_better=lower)
        if r:
            out[disp] = r
    return out


def best(cands: dict[str, dict], lower: bool) -> tuple[str, dict]:
    key = (min if lower else max)(cands, key=lambda k: cands[k]["overall"]["mean"])
    return key, cands[key]


def b(text: str, on: bool) -> str:
    return f"\\B{{{text}}}" if on else text


# -- parameter estimation ----------------------------------------------------

def params_overall(runs: dict) -> tuple[str, dict]:
    per_fam = {f: classical(runs, f, lower=True) for f, _ in FAMILIES}
    winners = {f: best(c, lower=True)[0] for f, c in per_fam.items()}
    # Rows: L, then every subset that wins at least one family, in SUBSETS order.
    shown = ["$L$"] + [d for _, d in SUBSETS if d in set(winners.values()) and d != "$L$"]
    shown = list(dict.fromkeys(shown))
    lines = []
    for disp in shown:
        cells = []
        for f, _ in FAMILIES:
            r = per_fam[f].get(disp)
            if r is None:
                cells.append("--")
                continue
            o = r["overall"]
            cells.append(b(f"{o['mean']:.4f} $\\pm$ {o['sd']:.4f}", winners[f] == disp))
        lines.append(f"{disp} & " + " & ".join(cells) + "\\\\")
    lines.append("\\midrule")
    for name, disp in PH:
        cells = []
        for f, _ in FAMILIES:
            o = runs[f"dv3_{f}/dtm_k5/{name}"]["overall"]
            cells.append(f"{o['mean']:.4f} $\\pm$ {o['sd']:.4f}")
        lines.append(f"{disp} & " + " & ".join(cells) + "\\\\")
    body = "\n".join(lines)
    table = "\n".join([
        r"\begin{table}[H]", r"\centering\small", r"\begin{tabular}{@{}l cccc@{}}", r"\toprule",
        r"Model & Thomas & Nested & Mat\'ern II & LGCP\\", r"\midrule", body,
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Parameter estimation, normalised loss on the test set (mean $\pm$ SD over "
        r"10 seeds). Shown: $L$ alone and every subset of $\{L,F,G,J\}$ that is best on at "
        r"least one family. Bold: best per family.}",
        r"\label{tab:params}", r"\end{table}",
    ])
    return table, per_fam


def params_delta(runs: dict, per_fam: dict) -> str:
    lines = []
    for f, flabel in FAMILIES:
        wname, wrun = best(per_fam[f], lower=True)
        rows = [("$L$", per_fam[f]["$L$"])]
        if wname != "$L$":
            rows.append((f"{wname} (best)", wrun))
        else:
            rows[0] = ("$L$ (best)", wrun)
        rows += [(disp, runs[f"dv3_{f}/dtm_k5/{name}"]) for name, disp in PH]
        for i, (disp, r) in enumerate(rows):
            head = flabel if i == 0 else ""
            cells = [f"{r['strata'][s]['mean']:.3f}" for s in STRATA]
            lines.append(f"{head} & {disp} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    return "\n".join([
        r"\begin{table}[H]", r"\centering\small", r"\begin{tabular}{@{}ll cccc@{}}", r"\toprule",
        r" & & \multicolumn{4}{c}{Distance from Poisson}\\", r"\cmidrule(l){3-6}",
        f"Family & Model & {STRATUM_HEAD}\\\\", r"\midrule",
        "\n".join(lines[:-1]),
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Parameter estimation by band (normalised loss, mean over 10 seeds; "
        r"2{,}500 patterns per band per family).}",
        r"\label{tab:params-delta}", r"\end{table}",
    ])


def params_gap(runs: dict, per_fam: dict) -> str:
    """Two ratios per family: against the best classical model, and against L
    alone -- the second is what says whether PH beats the published baseline."""
    lines = []
    for f, flabel in FAMILIES:
        ph = runs[f"dv3_{f}/dtm_k5/{PH_BEST}"]
        _, wrun = best(per_fam[f], lower=True)
        for i, (ref_lab, ref) in enumerate([("best", wrun), ("$L$", per_fam[f]["$L$"])]):
            ratios = [ph["strata"][s]["mean"] / ref["strata"][s]["mean"] for s in STRATA]
            top = max(ratios)
            cells = [b(f"{x:.2f}", x == top and i == 0) for x in ratios]
            head = flabel if i == 0 else ""
            lines.append(f"{head} & vs.\\ {ref_lab} & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule")
    return "\n".join([
        r"\begin{table}[H]", r"\centering\small", r"\begin{tabular}{@{}ll cccc@{}}", r"\toprule",
        r" & & \multicolumn{4}{c}{Distance from Poisson}\\", r"\cmidrule(l){3-6}",
        f"Family & & {STRATUM_HEAD}\\\\", r"\midrule",
        "\n".join(lines[:-1]),
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Gap between PH and the classical models: loss of PH $H_0{+}H_1$ divided by "
        r"the loss of the family's best classical model, and of $L$ alone (1 = parity, "
        r"$>1$ = PH worse). Bold: largest gap against the best model, per family.}",
        r"\label{tab:params-gap}", r"\end{table}",
    ])


# -- classification ----------------------------------------------------------

def classify_rows(runs: dict) -> list[tuple[str, dict]]:
    cands = classical(runs, "classify", lower=False)
    ranked = sorted(cands, key=lambda k: -cands[k]["overall"]["mean"])
    shown = ["$L$"] + [k for k in ranked if k != "$L$"][:3]
    order = [d for _, d in SUBSETS if d in shown]
    return [(d, cands[d]) for d in dict.fromkeys(order)]


def classify_overall(runs: dict) -> str:
    rows = classify_rows(runs)
    ph = [(disp, runs[f"dv3_classify/dtm_k5/{name}"]) for name, disp in PH]
    every = rows + ph
    fams = ["poisson", "thomas", "nested", "matern2", "lgcp"]
    top_acc = max(r["overall"]["mean"] for _, r in every)
    top_rec = {f: max(r["recall"][f]["mean"] for _, r in every) for f in fams}
    lines = []
    for i, (disp, r) in enumerate(every):
        if i == len(rows):
            lines.append("\\midrule")
        o = r["overall"]
        cells = [b(f"{o['mean']:.3f} $\\pm$ {o['sd']:.3f}", o["mean"] == top_acc)]
        cells += [b(f"{r['recall'][f]['mean']:.3f}", r["recall"][f]["mean"] == top_rec[f])
                  for f in fams]
        lines.append(f"{disp} & " + " & ".join(cells) + "\\\\")
    return "\n".join([
        r"\begin{table}[H]", r"\centering\small", r"\begin{tabular}{@{}l c ccccc@{}}", r"\toprule",
        r" & & \multicolumn{5}{c}{Recall per family}\\", r"\cmidrule(l){3-7}",
        r"Model & Accuracy & Poisson & Thomas & Nested & Mat\'ern II & LGCP\\", r"\midrule",
        "\n".join(lines),
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Classification on the test set (mean over 10 seeds; SD shown for accuracy). "
        r"Shown: $L$ alone and the three most accurate subsets of $\{L,F,G,J\}$. Bold: best "
        r"per column among the models shown.}",
        r"\label{tab:classify}", r"\end{table}",
    ])


def classify_delta(runs: dict, ts: T.TestSet) -> tuple[str, dict]:
    rows = classify_rows(runs) + [(d, runs[f"dv3_classify/dtm_k5/{n}"]) for n, d in PH]
    det = {}
    for disp, r in rows:
        det[disp] = detection_run(ROOT / "results" / Path(r["run"]).as_posix().split("results/", 1)[1], ts)
    lines = []
    n_classical = len(classify_rows(runs))
    for i, (disp, r) in enumerate(rows):
        if i == n_classical:
            lines.append("\\midrule")
        acc = [f"{r['strata'][s]['mean']:.3f}" for s in STRATA]
        pwr = [f"{det[disp]['strata'][s]['mean']:.3f}" for s in STRATA]
        lines.append(f"{disp} & " + " & ".join(acc) + " & & " + " & ".join(pwr) + "\\\\")
    sizes = [d["size"] for d in det.values()]
    short = " & ".join(f"\\texttt{{{a}}}" for a in ("below", "weak", "mod.", "strong"))
    table = "\n".join([
        r"\begin{table}[H]", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l cccc c cccc@{}}", r"\toprule",
        r" & \multicolumn{4}{c}{Accuracy} & & \multicolumn{4}{c}{Detection power}\\",
        r"\cmidrule(lr){2-5}\cmidrule(l){7-10}",
        f"Model & {short} & & {short}\\\\", r"\midrule",
        "\n".join(lines),
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Left: accuracy by band (chance 0.20 in every band). Right: the classifiers "
        r"used as tests of ``Poisson vs.\ not Poisson'' (score $1-P(\text{Poisson})$), threshold "
        r"set at 5\% false alarms per $\nbar$ octave on half of the test set's Poisson patterns, "
        r"power measured on the structured patterns; the false-alarm rate on the other half is "
        f"{min(sizes):.3f}--{max(sizes):.3f} for every model.}}",
        r"\label{tab:classify-delta}", r"\end{table}",
    ])
    return table, det


def classify_gap(runs: dict, det: dict) -> str:
    rows = dict(classify_rows(runs))
    cands = classical(runs, "classify", lower=False)
    wname, wrun = best(cands, lower=False)
    ph_disp = dict((d, n) for n, d in PH)["PH $H_0{+}H_1$"]
    ph = runs[f"dv3_classify/dtm_k5/{ph_disp}"]
    ph_det = det["PH $H_0{+}H_1$"]
    lonly = rows["$L$"]

    def row(label: str, a: list[float], ref: list[float], bold_worst: bool) -> str:
        # Round before signing, so a gap that rounds to zero prints as 0.0, not -0.0.
        gaps = [round(100 * (x - y), 1) + 0.0 for x, y in zip(a, ref)]
        worst = min(gaps)
        cells = [b(("0.0" if g == 0 else f"{g:+.1f}").replace("-", "$-$"),
                   bold_worst and g == worst and g < 0) for g in gaps]
        return f"{label} & " + " & ".join(cells) + "\\\\"

    acc_ph = [ph["strata"][s]["mean"] for s in STRATA]
    det_ph = [ph_det["strata"][s]["mean"] for s in STRATA]
    lines = [
        f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{vs.\\ the best classical model ({wname})}}}}\\\\",
        row("\\quad Accuracy", acc_ph, [wrun["strata"][s]["mean"] for s in STRATA], True),
        row("\\quad Detection power", det_ph, [det[wname]["strata"][s]["mean"] for s in STRATA], True),
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textit{vs.\ $L$ alone (the published baseline)}}\\",
        row("\\quad Accuracy", acc_ph, [lonly["strata"][s]["mean"] for s in STRATA], False),
        row("\\quad Detection power", det_ph, [det["$L$"]["strata"][s]["mean"] for s in STRATA], False),
    ]
    return "\n".join([
        r"\begin{table}[H]", r"\centering\small", r"\begin{tabular}{@{}l cccc@{}}", r"\toprule",
        f" & {STRATUM_HEAD}\\\\", r"\midrule",
        "\n".join(lines),
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Gap between PH $H_0{+}H_1$ and the classical models, in percentage points "
        r"(negative = PH worse, positive = PH better). Bold: largest deficit against the best "
        r"model, per row.}",
        r"\label{tab:classify-gap}", r"\end{table}",
    ])


def macros(runs: dict, per_fam: dict, det: dict) -> str:
    """Numbers the prose quotes, so the text cannot drift from the tables."""
    m = []
    def add(name: str, value: str) -> None:
        m.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    cands = classical(runs, "classify", lower=False)
    wname, wrun = best(cands, lower=False)
    ph = runs[f"dv3_classify/dtm_k5/{PH_BEST}"]
    ph0 = runs["dv3_classify/dtm_k5/pi_multik_h0_nocc_norm"]
    ph1 = runs["dv3_classify/dtm_k5/pi_multik_h1_nocc_norm"]
    lonly = cands["$L$"]
    add("cBestName", wname)
    add("cBestAcc", f"{wrun['overall']['mean']:.3f}")
    add("cLAcc", f"{lonly['overall']['mean']:.3f}")
    add("cPhAcc", f"{ph['overall']['mean']:.3f}")
    add("cPhZeroAcc", f"{ph0['overall']['mean']:.3f}")
    add("cPhOneAcc", f"{ph1['overall']['mean']:.3f}")
    add("cGapBest", f"{100 * (wrun['overall']['mean'] - ph['overall']['mean']):.1f}")
    add("cGapL", f"{100 * (lonly['overall']['mean'] - ph['overall']['mean']):.1f}")
    for s in STRATA:
        tag = s.capitalize()
        add(f"cPh{tag}", f"{ph['strata'][s]['mean']:.3f}")
        add(f"cL{tag}", f"{lonly['strata'][s]['mean']:.3f}")
        add(f"cBest{tag}", f"{wrun['strata'][s]['mean']:.3f}")
        add(f"cVsL{tag}", f"{100 * (ph['strata'][s]['mean'] - lonly['strata'][s]['mean']):+.1f}"
            .replace("-", "$-$"))
    for fam in ("thomas", "matern2"):
        tag = {"thomas": "Thomas", "matern2": "Matern"}[fam]
        add(f"cRec{tag}Ph", f"{ph['recall'][fam]['mean']:.3f}")
        add(f"cRec{tag}Best", f"{wrun['recall'][fam]['mean']:.3f}")
    # Parameter estimation: the range of PH / best-classical ratios.
    ratios = []
    for f, _ in FAMILIES:
        _, wr = best(per_fam[f], lower=True)
        ratios.append((runs[f"dv3_{f}/dtm_k5/{PH_BEST}"]["overall"]["mean"] / wr["overall"]["mean"], f))
    lo, hi = min(ratios), max(ratios)
    name = {"thomas": "Thomas", "nested": "nested", "matern2": "Mat\\'ern~II", "lgcp": "LGCP"}
    add("pRatioLo", f"{lo[0]:.1f}")
    add("pRatioLoFam", name[lo[1]])
    add("pRatioHi", f"{hi[0]:.1f}")
    add("pRatioHiFam", name[hi[1]])
    return "\n".join(m)


def main() -> int:
    runs = load()
    ts = T.load()
    OUT.mkdir(parents=True, exist_ok=True)
    p_overall, per_fam = params_overall(runs)
    c_delta, det = classify_delta(runs, ts)
    outputs = {
        "concise_params": p_overall,
        "concise_params_delta": params_delta(runs, per_fam),
        "concise_params_gap": params_gap(runs, per_fam),
        "concise_classify": classify_overall(runs),
        "concise_classify_delta": c_delta,
        "concise_classify_gap": classify_gap(runs, det),
        "concise_macros": macros(runs, per_fam, det),
    }
    for name, body in outputs.items():
        path = OUT / f"{name}.tex"
        path.write_text(body + "\n")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
