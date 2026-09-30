"""The frozen results as one tidy table, results/summary.csv, and the LaTeX tables built from it.

One row per (method, family, metric, subset): `mean` is the point estimate on the test split;
`std` and `n_seeds` are over training seeds (the paper run has one, so `std` is empty); `lo`, `hi`
are compare's 95% bootstrap interval over test thetas and `se` the standard error over clouds of
an end-to-end mean. `source` names the file a row was read from, relative to the run.

    classifiers     accuracy (all, regime 0.5, regime 0.9), nll, recall and detected per family
    estimators      rmse_over_sd (val, all, regime 0.5, regime 0.9) per family; min. contrast beside them
    end to end      kernel / dss skill, regret and gain per pipeline, evaluation set and group of clouds
    cost            train_seconds per unit

A unit trained after the last compare has no interval yet: its row comes from its own report.json.
Nothing is scored here; like every paper script this only reads a finished run (paper.yaml).

    python paper/scripts/summary.py          # -> results/summary.csv, paper/tables/s{1,2}_*.tex
"""

from __future__ import annotations

import pandas as pd

import common as C

SUMMARY = C.ROOT / "results" / "summary.csv"
COLUMNS = ["method", "family", "metric", "mean", "std", "n_seeds", "lo", "hi", "se", "n", "subset", "source"]
REG = ["all", "regime 0.5", "regime 0.9"]
GROUPS = {"overall": "all", "structured": "structured"}                # report key -> subset
NESTED = {"by_family_stratum": "", "structured_by_end": "ended ", "identified": "identified "}


def row(method, family, metric, mean, source, subset="all", **more) -> dict:
    return {"method": method, "family": family, "metric": metric, "mean": mean, "n_seeds": 1, "subset": subset,
            "source": source, **more}


def ci(v: dict) -> dict:
    return {"mean": v["est"], "lo": v["lo"], "hi": v["hi"]}


def classifiers() -> list[dict]:
    rep, out = C.compare_report()["classifiers"], []
    for m, v in rep.items():
        src = "compare/report.json"
        out += [row(m, "all", "accuracy", None, src, s) | ci(v[f"accuracy {s}"]) for s in REG]
        out.append(row(m, "all", "nll", v["nll"], src))
        out += [row(m, f, k, x, src) for k in ("recall", "detected") for f, x in v[k].items()]
    for p in sorted((C.run() / "classify").glob("*/report.json")):
        v, src = C.read_json(p), f"classify/{p.parent.name}/report.json"
        if "task" not in v:                                            # oracle, mincontrast: not trained units
            continue
        out.append(row(v["model"], "all", "train_seconds", v["seconds"], src))
        if v["model"] not in rep:                                      # trained after the last compare
            out += [row(v["model"], "all", k, v[k], src) for k in ("accuracy", "nll")]
            out += [row(v["model"], f, k, x, src) for k in ("recall", "detected") for f, x in v[k].items()]
    mc = C.read_json(C.run() / "mincontrast" / "report.json")["classifiers"]
    for name in ("mincontrast (tuned)", "mincontrast (default)"):
        out.append(row(name, "all", "accuracy", mc[name]["accuracy"], "mincontrast/report.json"))
        out += [row(name, f, "recall", x, "mincontrast/report.json") for f, x in mc[name]["recall"].items()]
    return out


def estimators() -> list[dict]:
    rep, out = C.compare_report()["estimators"], []
    for f, d in rep.items():
        for m, v in d.items():
            out.append(row(m, f, "rmse_over_sd", v["val"], "compare/report.json", "val"))
            out += [row(m, f, "rmse_over_sd", None, "compare/report.json", s) | ci(v[s]) for s in REG]
    for p in sorted((C.run() / "estimate").glob("*/*/report.json")):
        v, src = C.read_json(p), f"estimate/{p.parent.parent.name}/{p.parent.name}/report.json"
        if "task" not in v:
            continue
        out.append(row(v["model"], v["family"], "train_seconds", v["seconds"], src))
        if v["model"] not in rep[v["family"]]:
            out.append(row(v["model"], v["family"], "rmse_over_sd", v["mean_rmse_over_sd"], src))
    for f, v in C.read_json(C.run() / "mincontrast" / "report.json")["estimators"].items():
        out += [row(f"mincontrast ({k})", f, "rmse_over_sd", v[k], "mincontrast/report.json") for k in ("tuned", "default")]
    return out


def end_to_end() -> list[dict]:
    out = []
    for name in C.cfg()["evaluation"]:
        path = C.evaluation(name) / "report.json"
        if not path.exists():
            continue
        src = str(path.relative_to(C.run()))
        for variant, v in C.read_json(path)["variants"].items():
            cells = [("all", GROUPS[k], v[k]) for k in GROUPS]
            cells += [(f, "all", c) for f, c in v["by_family"].items()]
            for key, prefix in NESTED.items():
                for k, c in v[key].items():
                    f, _, s = k.partition(" | ") if key == "by_family_stratum" else ("all", "", k)
                    if s != "all":                                     # poisson's one stratum is its by_family cell
                        cells.append((f, prefix + s, c))
            for f, subset, c in cells:
                for score in ("kernel", "dss"):
                    x = c[score]
                    out.append(row(f"pipeline:{variant}", f, f"{score}_skill", x["skill"], src, subset, n=c["n"]))
                    out += [row(f"pipeline:{variant}", f, f"{score}_{k}", x[k], src, subset, se=x[f"{k}_se"], n=c["n"])
                            for k in ("regret", "gain")]
    return out


def build() -> pd.DataFrame:
    df = pd.DataFrame(classifiers() + estimators() + end_to_end()).reindex(columns=COLUMNS)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY, index=False, float_format="%.6g")
    print(f"-> {SUMMARY.relative_to(C.ROOT)}  ({len(df)} rows)")
    return df


# ------------------------------------------------------------------------------------ the tables

def s1_pipelines(df: pd.DataFrame) -> None:
    """Kernel skill of every pipeline of the `mincontrast` evaluation: overall and per family."""
    d = df[(df.source == C.cfg()["evaluation"]["mincontrast"] + "/report.json") & (df.subset == "all")]
    skill = d[d.metric == "kernel_skill"].pivot(index="method", columns="family", values="mean")
    regret = d[(d.metric == "kernel_regret") & (d.family == "all")].set_index("method")
    fams = [f for f in C.FAMILIES if f != "poisson"]
    num = lambda x: "---" if pd.isna(x) else f"{x:.2f}"
    body = [[C.tex(m.removeprefix("pipeline:")), num(skill.loc[m, "all"]),
             f"{regret.loc[m, 'mean']:.1e} $\\pm$ {regret.loc[m, 'se']:.0e}", *[num(skill.loc[m, f]) for f in fams]]
            for m in skill.sort_values("all", ascending=False).index]
    C.write_table("s1_pipelines", ["Pipeline", "Skill", "Regret", *[C.TEX_NAMES[f] for f in fams]], body)


def s2_models(df: pd.DataFrame) -> None:
    """Every model: accuracy overall and in regime, and its estimation error per family."""
    acc = df[(df.metric == "accuracy") & (df.family == "all")].pivot(index="method", columns="subset", values="mean")
    err = df[(df.metric == "rmse_over_sd") & (df.subset == "all")].pivot(index="method", columns="family", values="mean")
    fams = [f for f in C.FAMILIES if f != "poisson"]
    num = lambda t, m, c: "---" if m not in t.index or c not in t or pd.isna(t.loc[m, c]) else f"{t.loc[m, c]:.3f}"
    body = [[C.tex(m) if "(" not in m else m, *[num(acc, m, s) for s in REG], *[num(err, m, f) for f in fams]]
            for m in acc.sort_values("all", ascending=False).index.union(err.index, sort=False)]
    C.write_table("s2_models", ["Model", "Acc.\\ all", "$\\tau{=}0.5$", "$\\tau{=}0.9$", *[C.TEX_NAMES[f] for f in fams]], body)


def main() -> None:
    df = build()
    s1_pipelines(df)
    s2_models(df)


if __name__ == "__main__":
    main()
