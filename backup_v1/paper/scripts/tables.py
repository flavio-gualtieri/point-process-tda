"""The paper's data tables, as booktabs tabulars in paper/tables/ (main.tex \\input's them inside its
own table environments, which keep the captions). T1 and T2 are written by hand from the code; T7
waits on the choice of headline pipeline.

    T3  training cost per model          <run>/{classify,estimate}/*/report.json (seconds)
    T4  classification                   <run>/compare/report.json; <run>/mincontrast/report.json
    T5  fitted regime boundary           <run>/compare/cutoffs.json
    T6  model vs Poisson, by stratum     evaluation main: clouds.csv + report.json (paper.yaml headline_variant)
    T8  score power                      paper.yaml power: report.json
    T9  estimation error                 <run>/compare/report.json; <run>/mincontrast/report.json
    T10 what PH adds                     <run>/compare/report.json; evaluation ph_ablation, ph_cell

    python paper/scripts/tables.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

REG = ["all", "regime 0.5", "regime 0.9"]


def t3_cost() -> None:
    rc = C.run_config()
    body = []
    for m in rc["classify"]:
        spec = rc["models"][m]
        clf = C.read_json(C.run() / "classify" / m / "report.json")["seconds"]
        est = [C.run() / "estimate" / f / m / "report.json" for f in C.FAMILIES if f != "poisson"]
        est = sum(C.read_json(p)["seconds"] for p in est if p.exists()) if m in rc["estimate"] else None
        inputs = spec.get("inputs") or [spec.get("curves") and "curves", spec.get("filtration")]
        body.append([C.tex(m), spec["learner"], ", ".join(i for i in inputs if i).replace("_", "\\_"),
                     f"{clf:,}", "---" if est is None else f"{est:,}"])
    C.write_table("t3_cost", ["Model", "Learner", "Input", "Classifier (s)", "Estimators (s)"], body, "@{}lllrr@{}")


def t4_classification() -> None:
    rep, fams = C.compare_report(), C.FAMILIES
    body = []
    for m, v in sorted(rep["classifiers"].items(), key=lambda kv: -kv[1]["accuracy all"]["est"]):
        body.append([C.tex(m), *[C.fmt_ci(v[f"accuracy {s}"]) for s in REG], f"{v['nll']:.3f}",
                     *[f"{v['recall'][f]:.2f}" for f in fams]])
    mc = C.run() / "mincontrast" / "report.json"
    if mc.exists():
        v = C.read_json(mc)["classifiers"]["mincontrast (tuned)"]
        body.append(["min.\\ contrast", f"{v['accuracy']:.3f}", "---", "---", "---", *[f"{v['recall'][f]:.2f}" for f in fams]])
    C.write_table("t4_classification", ["Model", "Acc.\\ all", "Acc.\\ $\\tau{=}0.5$", "Acc.\\ $\\tau{=}0.9$", "NLL",
                                        *[C.TEX_NAMES[f] for f in fams]], body, rules_after=(len(body) - 2,) if mc.exists() else ())


def t5_boundary() -> None:
    body = []
    for f, c in C.cutoffs().items():
        if "constant" in c:
            continue
        rule = lambda tau: "---" if c["u_boundary"].get(tau) is None else f"{np.exp(c['u_boundary'][tau]):.3g}"
        op = "\\ge" if c["direction"] == "increasing" else "\\le"
        body.append([C.TEX_NAMES[f], c["coordinate"].replace("_", "\\_"), f"{c['nbar_exponent']:+.2f}", f"${op}$",
                     rule("0.5"), rule("0.9"), f"{c['pool_fraction']['0.5']:.2f}", f"{c['pool_fraction']['0.9']:.2f}"])
    C.write_table("t5_boundary", ["Family", "$x$", "$a$", "detected if $x\\,\\bar n^{a}$", "$c_{0.5}$", "$c_{0.9}$",
                                  "in regime $0.5$", "in regime $0.9$"], body)


def t6_poisson_gap() -> None:
    v = C.cfg()["headline_variant"]
    d = pd.read_csv(C.evaluation("main") / "clouds.csv")
    d = d[d.variant == v]
    failed = d.failed.fillna(False).astype(bool) if "failed" in d else np.zeros(len(d), bool)   # older runs: no column
    fit = d.kernel_fit.where(~failed, d.kernel_csr)                       # an unrealisable fit is scored as CSR
    d = d.assign(gap=d.kernel_csr - fit)
    rep = C.read_json(C.evaluation("main") / "report.json")["variants"][v]
    se = lambda s: s.std(ddof=1) / np.sqrt(len(s))
    pm = lambda s: f"{s.mean():+.1e} $\\pm$ {se(s):.0e}"
    order = {"below 0.5": 0, "0.5-0.9": 1, "above 0.9": 2}              # cell's k bins sort after, by name
    groups = d[d.family != "poisson"].groupby(["family", "stratum"])
    keys = sorted(groups.groups, key=lambda k: (C.FAMILIES.index(k[0]), order.get(k[1], 3), k[1]))
    body = []
    for f, s in keys:
        g = groups.get_group((f, s))
        skill = rep["by_family_stratum"][f"{f} | {s}"]["kernel"]["skill"]
        body.append([C.TEX_NAMES[f], s, str(len(g)), pm(g.kernel_gain), pm(g.gap), "---" if skill is None else f"{skill:.2f}"])
    st = d[(d.family != "poisson") & (d.ended == "poisson")]
    body.append(["structured, called Poisson", "", str(len(st)), pm(st.kernel_gain), pm(st.gap), ""])
    C.write_table("t6_poisson_gap", ["Family", "Stratum", "$n$", "Oracle gain", "$\\Delta_{\\rm CSR}$", "Skill"], body,
                  "@{}llrrrr@{}", rules_after=(len(body) - 2,))


def t8_power() -> None:
    rep = C.read_json(C.ROOT / C.cfg()["power"] / "report.json")
    cols = [("kernel/1.5/point/1.0", "kernel"), ("dss/no_ph", "DSS"), ("energy_w", "W$_1$ energy")]
    rank = lambda key: (C.FAMILIES.index(key.split(" ")[0]), key)
    cells = sorted(rep[cols[0][0]], key=rank)
    body = []
    for key in cells:
        fam, _, bin_ = key.partition(" ")
        label = "$" + bin_.replace("inf", "\\infty") + "$" if bin_ else "---"
        row = [C.TEX_NAMES.get(fam, fam), label, str(rep[cols[0][0]][key]["n"])]
        row += [f"{rep[v][key]['d_prime']:+.2f}" if key in rep.get(v, {}) else "---" for v, _ in cols]
        row.append(f"{rep[cols[0][0]][key]['z']:+.1f}")
        body.append(row)
    C.write_table("t8_power", ["Family", "$\\tilde\\delta$ bin", "$n$", *[f"$d'$ {name}" for _, name in cols], "$z$ kernel"],
                  body, "@{}llrrrrr@{}")


def t9_estimation() -> None:
    rep = C.compare_report()
    mc = C.run() / "mincontrast" / "report.json"
    mc = C.read_json(mc)["estimators"] if mc.exists() else {}
    body = []
    for f, d in rep["estimators"].items():
        for m, v in sorted(d.items(), key=lambda kv: kv[1]["all"]["est"]):
            body.append([C.TEX_NAMES[f], C.tex(m) + (" $\\star$" if rep["best"].get(f) == m else ""), f"{v['val']:.3f}",
                         *[C.fmt_ci(v[s]) for s in REG]])
        if f in mc:
            body.append([C.TEX_NAMES[f], "min.\\ contrast (tuned)", "---", f"{mc[f]['tuned']:.3f}", "---", "---"])
    C.write_table("t9_estimation", ["Family", "Estimator ($\\star$ = best on val)", "Val", "All", "$\\tau{=}0.5$", "$\\tau{=}0.9$"],
                  body, "@{}llrrrr@{}")


def t10_ph() -> None:
    rep = C.compare_report()
    ph = "hgb_classical_ph"                                    # paired against the run's baseline, hgb_classical
    sk = {}
    for name in ("ph_ablation", "ph_cell"):
        p = C.evaluation(name) / "report.json"
        if p.exists():
            sk[name] = C.read_json(p)["variants"]
    fmt_skill = lambda c: "---" if c is None or c["kernel"]["skill"] is None else f"{c['kernel']['skill']:.2f}"
    body = []
    for f, d in rep["estimators"].items():
        est = C.fmt_ci(d[ph]["minus baseline"]["all"]) if ph in d and "minus baseline" in d[ph] else "---"
        e2e = sk.get("ph_ablation", {})
        skills = " / ".join(fmt_skill(e2e.get(v, {}).get("by_family", {}).get(f)) for v in ("classical", "ph")) if e2e else "---"
        body.append([C.TEX_NAMES[f], est, skills])
    acc = rep["classifiers"][ph]["accuracy minus baseline"]["all"]
    body.append(["classifier accuracy", C.fmt_ci(acc), ""])
    if "ph_cell" in sk:
        cell = " / ".join(fmt_skill(sk["ph_cell"].get(v, {}).get("overall")) for v in ("classical", "ph"))
        body.append(["cell, $k\\ge5$ (1000 clouds)", "", cell])
    C.write_table("t10_ph", ["Family", "Estimator error, $+$PH $-$ classical", "Skill classical / $+$PH"], body,
                  "@{}lrr@{}", rules_after=(len(body) - 3,))


def main() -> None:
    for t in (t3_cost, t4_classification, t5_boundary, t6_poisson_gap, t8_power, t9_estimation, t10_ph):
        t()


if __name__ == "__main__":
    main()
