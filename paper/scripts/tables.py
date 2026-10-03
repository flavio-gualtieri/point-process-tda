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

Results tables, named and labelled by the paper config (`names`, `fidelity`, `main_table`, `family_pair`,
`family_pair_true`, `coarse`); m_main is the main body's, the others are the appendix's:
    m_fidelity        skill given predicted vs true family, paired differences; by outcome      evaluation main
    m_families        per-family skill; difference to minimum contrast, also given true family  evaluation main
    m_main            identification beside fidelity, per pipeline (`main_table`); paired differences       evaluation main, classify/*
    m_classification  balanced accuracy, all and in the structured regime; coarse accuracy by mechanism    classify/*/predictions.npz

    python paper/scripts/tables.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as C
from cloudforger.baselines.mincontrast import FITTABLE

# structured families in the paper's order (Table 1); minimum contrast cannot fit the last two
FAMILY_ORDER = ["thomas", "nested", "ring", "lgcp", "matern2", "strauss", "cell"]
UNFITTABLE = [f for f in FAMILY_ORDER if f not in FITTABLE]

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
    if not C.cfg().get("power"):
        print("t8_power: no power check in this paper config -- skipped")
        return
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
        if name not in C.cfg()["evaluation"]:
            continue
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


# ------------------------------------------------------------------------------ the main-body tables

B_CLOUDS = 2000                                                  # bootstrap resamples over clouds


def name(key: str) -> str:
    return C.cfg().get("names", {}).get(key, key.replace("_", "\\_"))


def main_clouds() -> pd.DataFrame:
    """The main evaluation's per-cloud kernel scores. A fit the sampler cannot realise is scored as CSR
    (regret = gain); a cloud the score could not score at all (no gain) is dropped for every variant."""
    d = pd.read_csv(C.evaluation("main") / "clouds.csv")
    d = d[d.groupby("scored_case_id").kernel_gain.transform(lambda g: g.notna().all())]
    failed = d.failed.fillna(False).astype(bool) if "failed" in d else np.zeros(len(d), bool)
    return d.assign(reg=d.kernel_regret.where(~failed, d.kernel_gain))


def per_cloud(d: pd.DataFrame, variant: str) -> pd.DataFrame:
    return d[d.variant == variant].set_index("scored_case_id").sort_index()


def skill_ci(reg: np.ndarray, gain: np.ndarray, reg_b: np.ndarray | None = None, seed: int = 0):
    """Pooled skill 1 - sum(regret) / sum(gain) -- or, with reg_b, the paired difference skill(a) - skill(b) --
    and its 95% bootstrap interval over clouds."""
    est = (1 - reg.sum() / gain.sum()) if reg_b is None else (reg_b.sum() - reg.sum()) / gain.sum()
    idx = np.random.default_rng(seed).integers(0, len(gain), (B_CLOUDS, len(gain)))
    g = gain[idx].sum(1)
    boot = 1 - reg[idx].sum(1) / g if reg_b is None else (reg_b[idx].sum(1) - reg[idx].sum(1)) / g
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return est, lo, hi


def fmt_skill(est, lo, hi, sign: bool = False, stacked: bool = False) -> str:
    """0.90 [0.85, 0.95]; stacked, the interval goes on a line of its own under the estimate (narrow,
    single-column tables)."""
    f = "{:+.2f}" if sign else "{:.2f}"
    ci = f"{{\\scriptsize[${f.format(lo)}$, ${f.format(hi)}$]}}"
    if stacked:
        return f"\\begin{{tabular}}[t]{{@{{}}r@{{}}}}${f.format(est)}$\\\\[-2pt]{ci}\\end{{tabular}}"
    return f"${f.format(est)}$ {ci}"


def m_fidelity() -> None:
    """Top: skill of each estimator given the predicted and the true family (paper config `fidelity`), with
    paired differences down and across; the difference down also on the families minimum contrast can fit
    and on those it cannot. Bottom: the predicted-family fits split by what the classifier did
    with each structured cloud -- skill and count on the identified and the wrong-family clouds, and the
    count sent to Poisson."""
    d = main_clouds()
    grid = C.cfg()["fidelity"]
    ci = lambda a, b=None: skill_ci(a.reg.to_numpy(), a.kernel_gain.to_numpy(), None if b is None else b.reg.to_numpy())
    pc = {v: per_cloud(d, v) for row in grid for v in row}
    body = [[name(pred), fmt_skill(*ci(pc[pred])), fmt_skill(*ci(pc[true])),
             fmt_skill(*ci(pc[true], pc[pred].loc[pc[true].index]), sign=True)] for pred, true in grid]
    (a, _), (b, _) = grid[:2]
    fit = pc[a].family.isin(FITTABLE)
    for label, m in [(f"$\\Delta$ {name(a)} $-$ {name(b)}", slice(None)),
                     ("\\quad families with a $K$ fit", fit),
                     ("\\quad " + ", ".join(C.TEX_NAMES[f] for f in UNFITTABLE), pc[a].family.isin(UNFITTABLE))]:
        body.append([label, *[fmt_skill(*ci(pc[x][m], pc[y].loc[pc[x].index][m]), sign=True) for x, y in zip(*grid[:2])], ""])
    body.append(["\\emph{Predicted family, by outcome}", "Identified ($n$)", "Wrong family ($n$)", "To Poisson ($n$)"])
    for pred, _ in grid:
        s = pc[pred][pc[pred].family != "poisson"]
        right, fam = s.family_hat == s.family, s.ended == "family"
        cell = lambda g: f"{fmt_skill(*ci(g))} (${len(g)}$)"
        body.append([name(pred), cell(s[right]), cell(s[~right & fam]), f"${int((~fam).sum())}$"])
    C.write_table("m_fidelity", ["Pipeline", "Predicted family", "True family", "$\\Delta$ true $-$ predicted"],
                  body, "@{}lrrr@{}", rules_after=(len(grid) - 1, len(grid) + 2, len(grid) + 3))


def m_families() -> None:
    """Per-family skill of `family_pair`'s first pipeline, and the paired difference to the second given the
    predicted family and given the true one (`family_pair_true`); the second's skill is the first's minus the
    difference, left out to fit a single column. The families minimum contrast cannot fit (no closed-form K; it uses the
    training median when given them) come last, daggered."""
    d = main_clouds()
    pairs = [C.cfg()["family_pair"], C.cfg()["family_pair_true"]]
    pc = {v: per_cloud(d, v) for p in pairs for v in p}
    pc = {v: c.loc[pc[pairs[0][0]].index] for v, c in pc.items()}
    st = lambda est, lo, hi, sign=False: fmt_skill(est, lo, hi, sign=sign, stacked=True)
    body = []
    for f in FAMILY_ORDER:
        m = (pc[pairs[0][0]].family == f).to_numpy()
        r = {v: c.reg.to_numpy()[m] for v, c in pc.items()}
        g = pc[pairs[0][0]].kernel_gain.to_numpy()[m]
        (a, b), (ta, tb) = pairs
        body.append([C.TEX_NAMES[f] + ("$^\\dagger$" if f in UNFITTABLE else ""), st(*skill_ci(r[a], g)),
                     st(*skill_ci(r[a], g, r[b]), sign=True),
                     st(*skill_ci(r[ta], g, r[tb]), sign=True)])
    # single-column: the intervals and two-word headers are stacked, and the clouds per family are given in
    # main.tex's caption
    head = lambda h: h if " " not in h else "\\begin{tabular}[b]{@{}r@{}}" + h.replace(" ", "\\\\", 1) + "\\end{tabular}"
    C.write_table("m_families", ["Family", head(name(pairs[0][0])), head("$\\Delta$ predicted"), head("$\\Delta$ true family")],
                  body, "@{}lrrr@{}",
                  rules_after=(len(FAMILY_ORDER) - len(UNFITTABLE) - 1,))


def accuracies(model: str) -> list[float]:
    """Balanced accuracy (equal weight per family, test split) of one classifier, or of minimum-contrast
    selection: over the families on all patterns and in the structured regime at tau = 0.5 and 0.9, then over
    the mechanisms (paper config `coarse`) at tau = 0.9."""
    from cloudforger.pipeline.regime import in_regime
    coarse, cuts = C.cfg()["coarse"], C.cutoffs()
    p = C.classifier(model)
    p = p[p.split == "test"]
    r = C.rows().loc[p.index]
    truth = r.family.to_numpy()
    call = np.array(C.FAMILIES)[p[C.FAMILIES].to_numpy().argmax(1)]
    bal = lambda t, c, mask, labels: np.mean([np.mean(c[mask & (t == k)] == k) for k in labels])
    cells = [bal(truth, call, np.ones(len(r), bool) if tau is None else in_regime(cuts, r, tau), C.FAMILIES)
             for tau in (None, 0.5, 0.9)]
    gt, gc = np.vectorize(coarse.get)(truth), np.vectorize(coarse.get)(call)
    return cells + [bal(gt, gc, in_regime(cuts, r, 0.9), sorted(set(coarse.values())))]


def m_classification() -> None:
    """Balanced accuracy of every classifier the run compared and of minimum-contrast selection, all computed
    the same way from their predictions (`accuracies`); the coarse column is by mechanism at tau = 0.9."""
    skip = set(C.cfg().get("main_body_exclude") or ())          # ablations that belong to a section, not to Table 1
    models = [*sorted((m for m in C.compare_report()["classifiers"] if m not in skip),
                      key=lambda m: -C.compare_report()["classifiers"][m]["accuracy all"]["est"]),
              "mincontrast"]
    body = [[name(m), *[f"{x:.3f}" for x in accuracies(m)]] for m in models]
    C.write_table("m_classification", ["Classifier", "All", "$\\tau{=}0.5$", "$\\tau{=}0.9$", "Coarse"], body,     # coarse: at tau = 0.9 (caption)
                  "@{}lrrrr@{}", rules_after=(len(models) - 2,))


def m_main() -> None:
    """The main results table: per pipeline (paper config `main_table`: pipeline -> the classifier that routes
    it), identification beside fidelity. Identification is balanced accuracy over the families on all test
    patterns and at tau = 0.9, and over the mechanisms at tau = 0.9. Fidelity is skill given the predicted and
    the true family (`fidelity` pairs them), and on the non-Poisson clouds routed to a wrong family. Below, the
    paired difference of the first two pipelines, on every cloud and split by whether minimum contrast can fit
    the family."""
    cfg = C.cfg()
    if "main_table" not in cfg:
        print("m_main: no main_table in this paper config -- skipped")
        return
    routes, true_of = cfg["main_table"], dict(cfg["fidelity"])
    d = main_clouds()
    pc = {v: per_cloud(d, v) for p in routes for v in (p, true_of[p])}
    ci = lambda a, b=None: skill_ci(a.reg.to_numpy(), a.kernel_gain.to_numpy(), None if b is None else b.reg.to_numpy())
    body = []
    for p, clf in routes.items():
        acc = accuracies(clf)
        s = pc[p][pc[p].family != "poisson"]
        wrong = s[(s.family_hat != s.family) & (s.ended == "family")]
        body.append([name(p), *[f"{acc[i]:.3f}" for i in (0, 2, 3)], fmt_skill(*ci(pc[p])), fmt_skill(*ci(pc[true_of[p]])),
                     f"{fmt_skill(*ci(wrong))} (${len(wrong)}$)"])
    a, b = list(routes)[:2]
    fit = pc[a].family.isin(FITTABLE)
    for label, m in [("$\\Delta$, all families", slice(None)),            # Delta = first pipeline - second (caption)
                     ("$\\Delta$, closed-form $K$", fit),
                     ("$\\Delta$, " + ", ".join(C.TEX_NAMES[f] for f in UNFITTABLE), pc[a].family.isin(UNFITTABLE))]:
        diff = [fmt_skill(*ci(pc[x][m], pc[y].loc[pc[x].index][m]), sign=True) for x, y in ((a, b), (true_of[a], true_of[b]))]
        body.append([label, "", "", "", *diff, ""])
    C.write_table("m_main", ["", "All", "$\\tau{=}0.9$", "Mechanism", "Predicted family", "True family", "Wrong family ($n$)"],
                  body, "@{}lrrrrrr@{}", rules_after=(len(routes) - 1,),
                  groups=[("", 1), ("Identification (balanced accuracy)", 3), ("Fidelity (skill)", 3)])


def main() -> None:
    for t in (t3_cost, t4_classification, t5_boundary, t6_poisson_gap, t8_power, t9_estimation, t10_ph):
        t()


if __name__ == "__main__":
    main()
