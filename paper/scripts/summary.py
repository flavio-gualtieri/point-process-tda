"""The frozen results as one tidy table, results/summary.csv, and the story's tables built from it.

One row per (method, family, metric, subset): `mean` is the point estimate on the test split;
`std` and `n_seeds` are over seeds of the kind `seeds` names -- `training` (the paper run has one,
so `std` is empty) or `simulation` (the headline clouds rescored with 5 simulation seeds: `mean` is
the mean of the per-seed values). `lo`, `hi` are a 95% bootstrap interval: over test thetas for
models (compare's resamples), over clouds for end-to-end numbers. `se` is the standard error over
clouds of an end-to-end mean. `source` names the file a row was read from: relative to the run, or
to the repo for results/frozen/.

    classifiers     accuracy (all, regime 0.5, regime 0.9), nll, ece, recall and detected per family;
                    the mean posterior of every classifier as `ensemble`
    estimators      rmse_over_sd (val, all, regime 0.5, regime 0.9) per family; min. contrast beside them
    contrasts       `A - B` rows: paired differences of accuracy and rmse_over_sd for the story's pairs
                    (paper.yaml story.grid), on compare's bootstrap resamples
    end to end      kernel / dss skill, regret and gain per pipeline, evaluation set and group of clouds;
                    kernel skill over 5 simulation seeds, and paired `A - B` skill differences
    score checks    the wrong-model ladder (regret of a perturbed truth) and the oracle's gain over CSR
                    per stratum, with its minimum detectable gain at 100 clouds
    out of dist.    `called <family>` shares of each classifier on families the run never trained on
    cost            train_seconds per unit

A paper config that lacks an input (paper.yaml `repeat`, `dose`, `ood`; a grid cell) skips what reads it:
without `repeat`, fidelity is read from the main evaluation set at its one simulation seed.

A unit trained after the last compare has no interval yet: its row comes from its own report.json.
Nothing is trained or simulated here; like every paper script this only reads finished outputs
(paper.yaml). The tables are views of summary.csv, each written as LaTeX (paper/tables/) and as
Markdown into results/story.md, the page to consult.

    python paper/scripts/summary.py          # -> results/summary.csv, results/story.md, paper/tables/{s*_*,m_topology}.tex
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

import common as C

sys.path.insert(0, str(C.ROOT / "scripts"))
from compare import Boot, accuracy, ci as boot_ci, rmse_sd  # noqa: E402

from cloudforger.baselines.mincontrast import FITTABLE  # noqa: E402
from cloudforger.pipeline.core import TARGETS, family_weights  # noqa: E402
from cloudforger.pipeline.regime import in_regime  # noqa: E402

RESULTS = C.RESULTS
SUMMARY, STORY = RESULTS / "summary.csv", RESULTS / "story.md"
COLUMNS = ["method", "family", "metric", "mean", "std", "n_seeds", "seeds", "lo", "hi", "se", "n", "subset", "source"]
REG = ["all", "regime 0.5", "regime 0.9"]
GROUPS = {"overall": "all", "structured": "structured"}                # report key -> subset
NESTED = {"by_family_stratum": "", "structured_by_end": "ended ", "identified": "identified "}
FAMS = [f for f in C.FAMILIES if f != "poisson"]
PAPER_ORDER = ["thomas", "nested", "ring", "lgcp", "matern2", "strauss", "cell"]   # Table 1's order
B_CLOUDS = 2000                                                        # bootstrap resamples over clouds


def row(method, family, metric, mean, source, subset="all", **more) -> dict:
    return {"method": method, "family": family, "metric": metric, "mean": mean, "n_seeds": 1, "seeds": "training",
            "subset": subset, "source": source, **more}


def ci(v: dict) -> dict:
    return {"mean": v["est"], "lo": v["lo"], "hi": v["hi"]}


def rel(path) -> str:
    return str(path.relative_to(C.ROOT))


# ------------------------------------------------------------------------------------ the models

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


def ece(P: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the top call, equal-width confidence bins."""
    conf, hit = P.max(1), P.argmax(1) == y
    at = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum((at == b).mean() * abs(hit[at == b].mean() - conf[at == b].mean()) for b in range(bins) if (at == b).any()))


def calibration(post: dict, y: np.ndarray) -> list[dict]:
    """ECE per classifier, and the accuracy of their mean posterior (the test split is balanced)."""
    out = [row(m, "all", "ece", ece(P, y), f"classify/{m}/predictions.npz") for m, P in post.items()]
    call = np.mean(list(post.values()), axis=0).argmax(1)
    acc = np.mean([(call[y == k] == k).mean() for k in range(len(C.FAMILIES))])
    out.append(row(f"ensemble ({len(post)} classifiers)", "all", "accuracy", float(acc), "classify/*/predictions.npz"))
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


def story_pairs() -> list[tuple[str, str]]:
    """What topology adds (down the grid) and what the learner adds (across it): the pairs the grid has."""
    g = C.cfg()["story"]["grid"]
    pairs = [(g["classical + PH"].get(k), g["classical"].get(k)) for k in ("network", "trees")] + \
            [(g[i].get("network"), g[i].get("trees")) for i in ("classical + PH", "classical")]
    return [(a, b) for a, b in pairs if a and b]


def contrasts() -> list[dict]:
    """Paired accuracy and error differences on compare's resamples (same bootstrap, same seed), and
    calibration. Reads every model's predictions once."""
    rc, r = C.run_config(), C.rows()
    test = r[r.split == "test"]
    tid, tfam = test.index, test.family.to_numpy()
    boot = Boot(test, rc["compare"]["bootstrap"], rc["compare"]["seed"])
    subsets = {"all": np.ones(len(test), bool), **{f"regime {t:g}": in_regime(C.cutoffs(), test, t) for t in (0.5, 0.9)}}
    prior = family_weights(rc)
    post = {m: C.classifier(m).reindex(tid)[C.FAMILIES].to_numpy() for m in rc["classify"]}
    y = np.array([C.FAMILIES.index(f) for f in tfam])
    out = calibration(post, y)
    calls = {m: np.array(C.FAMILIES)[P.argmax(1)] for m, P in post.items()}
    acc = lambda m, mask: accuracy(boot, tfam, calls[m] == tfam, prior, mask)
    e2 = {}
    for a, b in story_pairs():
        name = f"{a} - {b}"
        out += [row(name, "all", "accuracy", None, "classify/*/predictions.npz", s) | ci(boot_ci(acc(a, k) - acc(b, k)))
                for s, k in subsets.items()]
        for f in FAMS:
            mine, targets = tfam == f, TARGETS[f]
            truth = np.log(test[targets].to_numpy(float))
            sd = truth[mine].std(0)
            for m in (a, b):
                if (f, m) not in e2:
                    x = (np.log(C.estimator(f, m).reindex(tid)[targets].to_numpy()) - truth) ** 2
                    x[~mine] = 0.0
                    e2[(f, m)] = x
            out += [row(name, f, "rmse_over_sd", None, f"estimate/{f}/*/predictions.npz", s)
                    | ci(boot_ci(rmse_sd(boot, e2[(f, a)], mine & k, sd) - rmse_sd(boot, e2[(f, b)], mine & k, sd)))
                    for s, k in subsets.items()]
    return out


# ------------------------------------------------------------------------------------ end to end

def end_to_end() -> list[dict]:
    """Every evaluation set's report, as stored (one simulation seed)."""
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


def clouds(path) -> pd.DataFrame:
    """Per-cloud kernel scores; a fit the sampler cannot realise is scored as CSR (regret = gain). A cloud the
    kernel score could not score at all (no gain: its held-out replicate is too small) is dropped for every
    variant, as endtoend.py's own sums skip it."""
    d = pd.read_csv(path)
    d = d[d.groupby("scored_case_id").kernel_gain.transform(lambda g: g.notna().all())]
    failed = d.failed.fillna(False).astype(bool) if "failed" in d else np.zeros(len(d), bool)
    return d.assign(reg=d.kernel_regret.where(~failed, d.kernel_gain))


def groups(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The clouds a skill is pooled over: everything, structured, and by what the classifier did."""
    s = d[d.family != "poisson"]
    right, fam = s.family_hat == s.family, s.ended == "family"
    return {"all": d, "structured": s, "identified True": s[right], "identified False": s[~right],
            "wrong family": s[~right & fam], "ended poisson": s[~fam]}


def skill(d: pd.DataFrame) -> float:
    return float(1 - d.reg.sum() / d.kernel_gain.sum())


def resolved(gain: pd.Series, min_z: float) -> bool:
    return len(gain) > 1 and gain.mean() > min_z * gain.std(ddof=1) / np.sqrt(len(gain))


def boot_skill(reg_a: np.ndarray, gain: np.ndarray, reg_b: np.ndarray | None = None, seed: int = 0) -> tuple[float, float]:
    """95% interval over clouds of a pooled skill, or of the paired difference skill(a) - skill(b)."""
    idx = np.random.default_rng(seed).integers(0, len(gain), (B_CLOUDS, len(gain)))
    g = gain[idx].sum(1)
    s = (reg_b[idx].sum(1) - reg_a[idx].sum(1)) / g if reg_b is not None else 1 - reg_a[idx].sum(1) / g
    return tuple(np.percentile(s, [2.5, 97.5]))


def per_cloud(d: pd.DataFrame, variant: str) -> pd.DataFrame:
    """One row per cloud: regret and gain averaged over simulation seeds (one seed: as stored)."""
    return d[d.variant == variant].groupby("scored_case_id")[["reg", "kernel_gain"]].mean()


def seeded(d: pd.DataFrame, source: str, pairs: list[tuple[str, str]], min_z: float) -> list[dict]:
    """Kernel skill of every variant and group, and paired skill differences. With a `seed` column:
    mean and s.d. over seeds of the per-seed value; the interval pools the seeds per cloud."""
    by_seed = list(d.groupby("seed")) if "seed" in d else [(0, d)]
    kind = "simulation" if "seed" in d else "training"
    out = []
    for v in d.variant.unique():
        for gname, g in groups(d[d.variant == v]).items():
            if not len(g):
                continue
            pooled = g.groupby("scored_case_id")[["reg", "kernel_gain"]].mean()
            n, gain = len(pooled), pooled.kernel_gain
            base = row(f"pipeline:{v}", "all", "", None, source, gname, n=n, n_seeds=len(by_seed), seeds=kind)
            out.append(base | {"metric": "kernel_gain", "mean": gain.mean(), "se": gain.std(ddof=1) / np.sqrt(n) if n > 1 else None})
            if not resolved(gain, min_z):
                continue
            per = [skill(groups(h[h.variant == v])[gname]) for _, h in by_seed]
            lo, hi = boot_skill(pooled.reg.to_numpy(), gain.to_numpy())
            out.append(base | {"metric": "kernel_skill", "mean": np.mean(per), "lo": lo, "hi": hi,
                               "std": np.std(per, ddof=1) if len(per) > 1 else None})
    for a, b in pairs:
        A, Bv = per_cloud(d, a), per_cloud(d, b).loc[per_cloud(d, a).index]
        per = [(h[h.variant == b].reg.sum() - h[h.variant == a].reg.sum()) / h[h.variant == a].kernel_gain.sum() for _, h in by_seed]
        lo, hi = boot_skill(A.reg.to_numpy(), A.kernel_gain.to_numpy(), Bv.reg.to_numpy())
        out.append(row(f"pipeline:{a} - pipeline:{b}", "all", "kernel_skill", np.mean(per), source, "all", lo=lo, hi=hi,
                       std=np.std(per, ddof=1) if len(per) > 1 else None, n=len(A), n_seeds=len(by_seed), seeds=kind))
    return out


def fidelity_source() -> str:
    """Where the story's fidelity rows come from: the 5-seed rescoring if the config has one, else the main
    evaluation set at its one simulation seed. A substring of those rows' `source`."""
    return "scorecheck/repeat" if C.cfg().get("repeat") else str((C.evaluation("main") / "clouds.csv").relative_to(C.run()))


def repeats() -> list[dict]:
    """The headline clouds over 5 simulation seeds (else the main set's one), and the cell clouds (one seed)."""
    min_z, st = C.run_config()["evaluation"]["skill_min_z"], C.cfg()["story"]
    pairs = [("fusion", "curves"), ("ph", "classical"), ("fusion", "ph"), ("curves", "classical"),
             ("fusion", "mincontrast"), ("ph", "mincontrast"), ("oracle_ph", "fusion"), ("oracle_ph", "ph"),
             ("oracle_fusion", "fusion"), ("oracle_mincontrast", "mincontrast"), ("fusion", "oracle_mincontrast")]
    pairs = [p for p in pairs if set(p) <= set(st["pipelines"])]
    if C.cfg().get("repeat"):
        files = sorted(glob.glob(str(C.ROOT / C.cfg()["repeat"])))
        d = pd.concat([clouds(p).assign(seed=int(p.rsplit("seed", 1)[1].split(".")[0])) for p in files])
        out = seeded(d, rel(C.ROOT / C.cfg()["repeat"]), pairs, min_z)
    else:
        path = C.evaluation("main") / "clouds.csv"
        out = seeded(clouds(path), str(path.relative_to(C.run())), pairs, min_z)
    if "networks_cell" not in C.cfg()["evaluation"]:
        return out
    cell = C.evaluation("networks_cell") / "clouds.csv"
    cp = [("fusion_estimators", "curves_estimators"), ("ph_estimators", "classical"),
          ("fusion_estimators", "ph_estimators"), ("curves_estimators", "classical")]
    return out + [r | {"family": "cell"} for r in seeded(clouds(cell), str(cell.relative_to(C.run())), cp, min_z)]


def contrast_sets() -> list[dict]:
    """Evaluation sets that re-score the main set's clouds under a changed input (story `contrasts`:
    set -> the pairs to difference). Scored on the same clouds, so the differences are paired and their
    interval is the bootstrap over clouds; each set carries its own `fusion` rows, identical to the main
    set's, which is what the pairs are taken against."""
    min_z, st = C.run_config()["evaluation"]["skill_min_z"], C.cfg()["story"]
    out = []
    for name, cps in (st.get("contrasts") or {}).items():
        if name not in C.cfg()["evaluation"]:
            continue
        path = C.evaluation(name) / "clouds.csv"
        if not path.exists():
            continue
        out += seeded(clouds(path), str(path.relative_to(C.run())), [tuple(p) for p in cps], min_z)
    return out


def score_checks() -> list[dict]:
    """Wrong-model ladder: regret of the truth perturbed by d s.d. of log theta (and cruder guesses),
    by stratum tier; and the oracle's gain over CSR per stratum with its minimum detectable gain."""
    if not C.cfg().get("dose"):
        return []
    path = C.ROOT / C.cfg()["dose"]
    d, src = clouds(path), rel(path)
    tier = {"above 0.9": "strong", "k 5-30": "strong", "0.5-0.9": "middle", "k 3-4": "middle"}
    d["tier"] = d.stratum.map(tier).fillna("weak")
    out = []
    for (v, t), g in d[~d.failed.fillna(False).astype(bool)].groupby(["variant", "tier"]):
        n = len(g)
        out.append(row(f"wrong model: {v}", "all", "kernel_regret", g.reg.mean(), src, t, se=g.reg.std(ddof=1) / np.sqrt(n), n=n))
        out.append(row(f"wrong model: {v}", "all", "kernel_skill", skill(g), src, t, n=n))
    o = d.drop_duplicates("scored_case_id")
    for (f, s), g in o.groupby(["family", "stratum"]):
        sd, n = g.kernel_gain.std(ddof=1), len(g)
        out.append(row("oracle", f, "kernel_gain", g.kernel_gain.mean(), src, s, se=sd / np.sqrt(n), n=n))
        out.append(row("oracle", f, "kernel_mdg100", 2 * sd / 10, src, s, n=n))    # 2 s.e. at 100 clouds
    return out


def out_of_distribution() -> list[dict]:
    """What each classifier calls the test clouds of a family the run never trained on (scripts/ood.py)."""
    if not C.cfg().get("ood"):
        return []
    out = []
    for fam_dir in sorted((C.run() / C.cfg()["ood"]).glob("*/")):
        for p in sorted(fam_dir.glob("classify/*/predictions.npz")):
            z = np.load(p)
            call = np.array(z["classes"])[z["posterior"].argmax(1)]
            src = str(p.relative_to(C.run()))
            out += [row(p.parent.name, fam_dir.name, f"called {f}", float(np.mean(call == f)), src, n=len(call))
                    for f in z["classes"]]
    return out


def build() -> pd.DataFrame:
    rows_ = (classifiers() + estimators() + contrasts() + end_to_end() + repeats() + contrast_sets()
             + score_checks() + out_of_distribution())
    df = pd.DataFrame(rows_).reindex(columns=COLUMNS)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY, index=False, float_format="%.6g")
    print(f"-> {SUMMARY.relative_to(C.ROOT)}  ({len(df)} rows)")
    return df


# ------------------------------------------------------------------------------------ the views

def pick(df, method, metric, family="all", subset="all", source=None) -> pd.Series | None:
    m = (df.method == method) & (df.metric == metric) & (df.family == family) & (df.subset == subset)
    if source is not None:
        m &= df.source.str.contains(source, regex=False)
    hit = df[m]
    return None if hit.empty else hit.iloc[0]


def num(x, d=3) -> str:
    return "---" if x is None or pd.isna(x) else f"{x:.{d}f}"


def interval(r, d=3, sign=True) -> str:
    if r is None or pd.isna(r["mean"]):
        return "---"
    f = f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}"
    s = f.format(r["mean"])
    return s if pd.isna(r["lo"]) else f"{s} [{f.format(r['lo'])}, {f.format(r['hi'])}]"


def seeds_pm(r, d=3) -> str:
    if r is None or pd.isna(r["mean"]):
        return "---"
    return f"{r['mean']:.{d}f}" + ("" if pd.isna(r["std"]) else f" ± {r['std']:.{d}f}")


def tex_cell(s: str) -> str:
    """Markdown cell -> LaTeX: intervals in small type, ± as math."""
    s = s.replace("±", "$\\pm$").replace("≥", "$\\ge$").replace("τ", "$\\tau$").replace("_", "\\_").replace("%", "\\%")
    s = s.replace("−", "$-$")
    s = s.replace("é", "\\'e")                                  # the .tex files carry no inputenc
    if " [" in s:
        head, _, tail = s.partition(" [")
        s = f"{head} {{\\scriptsize [{tail}}}"
    return s


class Page:
    """results/story.md: the views, section by section, as Markdown; each also written as LaTeX."""

    def __init__(self):
        self.lines = ["# The story in numbers", "",
                      "Generated by `paper/scripts/summary.py`" + ("" if C.CONFIG.name == "paper.yaml" else f" ({C.CONFIG.name})")
                      + f" from `{SUMMARY.relative_to(C.ROOT)}`; "
                      f"do not edit. Each table is also in `{C.TABLES.relative_to(C.ROOT)}/<name>.tex`. Intervals are "
                      "95% bootstrap" + ("; `±` is the s.d. over 5 simulation seeds." if C.cfg().get("repeat") else "."), ""]

    def section(self, title: str, notes: list[str]):
        self.lines += [f"## {title}", ""] + [f"- {n}" for n in notes] + [""]

    def table(self, name: str, header: list[str], body: list[list[str]], rules_after: tuple[int, ...] = ()):
        self.lines += [f"`{name}`", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        self.lines += ["| " + " | ".join(r) + " |" for r in body] + [""]
        C.write_table(name, [tex_cell(h) for h in header], [[tex_cell(c) for c in r] for r in body],
                      "@{}l" + "r" * (len(header) - 1) + "@{}", rules_after)

    def write(self):
        STORY.write_text("\n".join(self.lines) + "\n")
        print(f"-> {STORY.relative_to(C.ROOT)}")


def model_name(m: str) -> str:
    return m if " " in m else f"`{m}`"


def s3_grid(df: pd.DataFrame, page: Page) -> None:
    st = C.cfg()["story"]
    err = lambda m: np.nanmean([pick(df, m, "rmse_over_sd", f)["mean"] if pick(df, m, "rmse_over_sd", f) is not None else np.nan
                                for f in FAMS])
    hours = lambda m: df[(df.method == m) & (df.metric == "train_seconds")]["mean"].sum() / 3600
    body = []
    for inp, learners in st["grid"].items():
        for learner, m in learners.items():
            body.append([inp, learner, model_name(m), interval(pick(df, m, "accuracy"), sign=False),
                         num(pick(df, m, "accuracy", subset="regime 0.9")["mean"]), num(pick(df, m, "ece")["mean"]),
                         num(err(m)), f"{hours(m):.2f}"])
    for m in st["floor"]:
        body.append(["classical", "linear", model_name(m), interval(pick(df, m, "accuracy"), sign=False),
                     num(pick(df, m, "accuracy", subset="regime 0.9")["mean"]), num(pick(df, m, "ece")["mean"]), "---",
                     f"{hours(m):.2f}"])
    body.append(["$\\hat K$", "min. contrast", "mincontrast (tuned)", num(pick(df, "mincontrast (tuned)", "accuracy")["mean"]),
                 "---", "---", num(err("mincontrast (tuned)")), "---"])
    page.section("Section 3. The model and its controls",
                 [f"Headline: `{st['headline']}`, one network for the classifier and every family's estimator.",
                  "Controls: the same inputs to gradient-boosted trees, each input alone, a linear model, minimum contrast.",
                  "Error = RMSE(log θ)/s.d., mean over the 7 structured families (per family: `sA_models`). "
                  "ECE: 15 equal-width bins on the top call. Train: classifier + 7 estimators, hours."])
    page.table("s3_grid", ["Input", "Learner", "Model", "Accuracy", "Acc. τ=0.9", "ECE", "Error", "Train (h)"], body,
               rules_after=(len(body) - 3,))


def s4_ceiling(df: pd.DataFrame, page: Page) -> None:
    acc = df[(df.metric == "accuracy") & (df.family == "all") & (df.subset == "all") & ~df.method.str.contains(" - ")]
    learned = acc[~acc.method.str.startswith(("mincontrast", "ensemble"))].sort_values("mean", ascending=False)
    ens = acc[acc.method.str.startswith("ensemble")].iloc[0]
    top = learned.iloc[0]
    reg = df[(df.metric == "accuracy") & (df.subset == "regime 0.9") & df.method.isin(learned.method)]
    page.section("Section 4. The ceiling", [
        f"{len(learned)} learned classifiers span {learned['mean'].min():.3f}–{learned['mean'].max():.3f}; "
        f"the top {int((learned['mean'] > top['mean'] - 0.005).sum())} are within 0.005 of the best "
        f"(`{top.method}` {top['mean']:.3f} [{top.lo:.3f}, {top.hi:.3f}]).",
        f"The mean posterior of all {len(learned)} ({ens.method}): {ens['mean']:.3f}.",
        f"In regime (τ = 0.9) the same models span {reg['mean'].min():.3f}–{reg['mean'].max():.3f}.",
        "Per-model table with intervals: `sA_models`; regime boundary and confusion: T5, F3, F4 (`paper/scripts/make.py`)."])


def s5_fidelity(df: pd.DataFrame, page: Page) -> None:
    st, src = C.cfg()["story"], fidelity_source()
    seeds = bool(C.cfg().get("repeat"))
    body = []
    for v, label in st["pipelines"].items():
        g = lambda subset: pick(df, f"pipeline:{v}", "kernel_skill", subset=subset, source=src)
        sent = pick(df, f"pipeline:{v}", "kernel_gain", subset="ended poisson", source=src)
        body.append([label, f"`{v}`", seeds_pm(g("all")), f"[{g('all').lo:.3f}, {g('all').hi:.3f}]",
                     seeds_pm(g("identified True")), seeds_pm(g("wrong family")), "0" if sent is None else str(int(sent.n))])
    pairs = df[df.method.str.contains(" - pipeline:") & df.source.str.contains(src, regex=False) & (df.family == "all")]
    pbody = [[f"`{r.method.replace('pipeline:', '')}`", f"{r['mean']:+.3f}" + (f" ± {r['std']:.3f}" if seeds else ""),
              f"[{r.lo:+.3f}, {r.hi:+.3f}]"] for _, r in pairs.iterrows()]
    sent = {v: pick(df, f"pipeline:{v}", "kernel_gain", subset="ended poisson", source=src) for v in st["pipelines"]}
    sent = {v: s for v, s in sent.items() if s is not None}
    hit = {v: pick(df, f"pipeline:{v}", "kernel_gain", subset="identified True", source=src) for v in sent}
    share = [sent[v]["mean"] / hit[v]["mean"] for v in sent]
    z = [s["mean"] / s["se"] for s in sent.values()]
    n_clouds = int(pick(df, f"pipeline:{next(iter(st['pipelines']))}", "kernel_gain", subset="all", source=src).n)
    page.section(f"Section 5. Fidelity ({n_clouds} headline clouds, " + ("5 simulation seeds)" if seeds else "one simulation seed)"), [
        "Skill: 1 = as good as the true model, 0 = no better than a Poisson fit. "
        + ("Mean ± s.d. over 5 simulation seeds (same fits); interval: bootstrap over clouds, seeds pooled."
           if seeds else "One simulation seed; interval: bootstrap over clouds."),
        "Identified / wrong family: structured clouds the classifier called correctly / called another structured family.",
        "Sent to Poisson: structured clouds the classifier called Poisson. What the true model would have gained over "
        f"the Poisson fit there is {min(share):.1%}–{max(share):.1%} of its gain on identified clouds "
        f"(z {min(z):.1f}–{max(z):.1f}" + (" with the seeds pooled; unresolved at one seed)." if seeds else ")."),
        *(["Per-family skill is not reported here: over seeds it moves by 0.15–0.7 for Matérn I, Matérn II and cell "
           "(60 clouds each)."] if seeds else
          ["Per-family skill is not reported here: 60 clouds per family at one seed cannot rank pipelines within a family."])])
    page.table("s5_fidelity", ["Pipeline", "Variant", "Skill", "95% CI", "Identified", "Wrong family", "Sent to Poisson (n)"], body)
    page.lines += ["Paired differences, same clouds and simulations (mean ± s.d. over seeds; 95% CI over clouds):", ""]
    page.table("s5_pairs", ["Difference", "Δ skill", "95% CI"], pbody)


def s6_topology(df: pd.DataFrame, page: Page) -> None:
    g = C.cfg()["story"]["grid"]
    if not all(k in g["classical + PH"] and k in g["classical"] for k in ("network", "trees")):
        page.section("Section 6. What topology adds", ["Skipped: the grid has no network without PH to pair the headline "
                                                       "with (story.grid `classical`). Trees + PH − trees: `sA_models`, T10."])
        return
    net, trees = (f"{g['classical + PH'][k]} - {g['classical'][k]}" for k in ("network", "trees"))
    body = [[C.NAMES[f], interval(pick(df, net, "rmse_over_sd", f)), interval(pick(df, trees, "rmse_over_sd", f))] for f in FAMS]
    body.append(["accuracy, all", interval(pick(df, net, "accuracy")), interval(pick(df, trees, "accuracy"))])
    body.append(["accuracy, τ=0.9", interval(pick(df, net, "accuracy", subset="regime 0.9")),
                 interval(pick(df, trees, "accuracy", subset="regime 0.9"))])
    rep = lambda a, b, fam="all": pick(df, f"pipeline:{a} - pipeline:{b}", "kernel_skill", fam)
    e2e = [rep("fusion", "curves"), rep("ph", "classical")]
    n_e2e = next((int(r.n) for r in e2e if r is not None), None)
    body.append([f"skill, {n_e2e} clouds" if n_e2e else "skill, end to end", *(interval(r) for r in e2e)])
    cell = [rep("fusion_estimators", "curves_estimators", "cell"), rep("ph_estimators", "classical", "cell")]
    body.append(["skill, cell k ≥ 5 (1000)", *(interval(r) for r in cell)])
    seeds = "mean over 5 simulation seeds" if C.cfg().get("repeat") else "one simulation seed"
    page.section("Section 6. What topology adds (paired: with PH − without, same learner)", [
        f"Network: `{net}`. Trees: `{trees}`.",
        "Error rows: RMSE(log θ)/s.d., negative = PH helps; 95% interval over test θ (compare's resamples). "
        "One training seed: between two networks, differences under about 0.005 are within seed noise.",
        f"Skill rows: positive = PH helps; paired on the same clouds, {seeds}. Cell: one seed, classifier fixed "
        "(`hgb_classical`), only the estimator changes."])
    page.table("s6_topology", ["", "Network + PH − network", "Trees + PH − trees"], body, rules_after=(len(FAMS) - 1, len(FAMS) + 1))


def s6_filtrations(df: pd.DataFrame, page: Page) -> None:
    """Which of the two filtrations carries what topology adds: the same network and trainer on both images,
    on each alone, and on neither (story `filtrations`: label -> model)."""
    st = C.cfg()["story"]
    cols = st.get("filtrations") or {}
    if len(cols) < 3:
        page.section("Section 6b. Which filtration carries it",
                     ["Skipped: story `filtrations` names fewer than three models to contrast."])
        return
    labels, models = list(cols), list(cols.values())
    err = lambda m, f: interval(pick(df, m, "rmse_over_sd", f, subset="regime 0.9"), sign=False)
    body = [[C.NAMES[f], *[err(m, f) for m in models]] for f in FAMS]
    body.append(["accuracy, τ=0.9", *[interval(pick(df, m, "accuracy", subset="regime 0.9"), sign=False)
                                      for m in models]])
    src, pipes = "fidelity_filtrations", st.get("filtration_pipelines") or {}
    if pipes:                                       # paired, the first column minus each other, on the same clouds
        first = pipes[labels[0]]
        paired = {l: pick(df, f"pipeline:{first} - pipeline:{pipes[l]}", "kernel_skill", source=src)
                  for l in labels[1:] if l in pipes}
        neither = next((l for l in labels if l not in pipes), None)          # the network without PH: its own set
        if neither is not None:
            paired[neither] = pick(df, f"pipeline:{first} - pipeline:curves", "kernel_skill")
        body.append([f"skill, {labels[0]} − column", "---", *[interval(paired.get(l)) for l in labels[1:]]])
    page.section("Section 6b. Which filtration carries it (same network and trainer, inputs removed)", [
        "Columns: " + ", ".join(f"{l} = `{m}`" for l, m in cols.items()) + ".",
        "Error rows: RMSE(log θ)/s.d. in the structured regime (τ = 0.9), lower is better; 95% interval over "
        "test θ (compare's resamples). One training seed: differences under about 0.005 are within seed noise.",
        f"Skill row: the first column's end-to-end skill minus each column's, paired on the same clouds "
        f"(positive = {labels[0]} is better); 95% interval over clouds."])
    page.table("s6_filtrations", ["", *labels], body, rules_after=(len(FAMS) - 1,))
    pairs = df[df.method.str.contains(" - pipeline:") & df.source.str.contains(src, regex=False) & (df.family == "all")]
    if pairs.empty:
        return
    page.lines += [f"Paired end-to-end differences, same {int(pairs.n.iloc[0])} clouds:", ""]
    page.table("s6_filtration_pairs", ["Difference", "Δ skill", "95% CI"],
               [[f"`{r.method.replace('pipeline:', '')}`", f"{r['mean']:+.3f}", f"[{r.lo:+.3f}, {r.hi:+.3f}]"]
                for _, r in pairs.iterrows()])


def m_topology(df: pd.DataFrame, page: Page) -> None:
    """Main body: what the persistence images add to the same network and trainer, as the relative change in
    estimation error (structured regime, tau = 0.9) against the network on the summary curves alone (story
    `filtrations`: `neither` is the baseline, every other label one input set). The families minimum contrast
    cannot fit come first. Absolute errors with intervals: s6_filtrations (appendix)."""
    cols = C.cfg()["story"].get("filtrations") or {}
    if "neither" not in cols or len(cols) < 2:
        page.section("Main body. What topology adds", ["Skipped: story `filtrations` has no `neither` baseline."])
        return
    base, rest = cols["neither"], {l: m for l, m in cols.items() if l != "neither"}
    err = lambda m, f: pick(df, m, "rmse_over_sd", f, subset="regime 0.9")["mean"]
    fams = [f for f in PAPER_ORDER if f not in FITTABLE] + [f for f in PAPER_ORDER if f in FITTABLE]
    pct = lambda x: "0%" if round(100 * x) == 0 else f"{round(100 * x):+d}%".replace("-", "−")
    body = [[C.NAMES[f], num(err(base, f)), *[pct(err(m, f) / err(base, f) - 1) for m in rest.values()]] for f in fams]
    page.section("Main body. What topology adds (relative change in error against the curves alone)", [
        f"Baseline: `{base}`, error RMSE(log θ)/s.d. in the structured regime (τ = 0.9). Columns: "
        + ", ".join(f"+ {l} = `{m}`" for l, m in rest.items()) + "; negative = the images help.",
        "One training seed: differences under about 0.005 in absolute error are within seed noise."])
    page.table("m_topology", ["", "Curves alone", *[f"+ {l}" for l in rest]], body,
               rules_after=(sum(f not in FITTABLE for f in fams) - 1,))


def sA_score(df: pd.DataFrame, page: Page) -> None:
    if not C.cfg().get("dose"):
        return
    ladder = df[df.method.str.startswith("wrong model:") & (df.metric == "kernel_regret")]
    order = ["noise_0.1", "noise_0.25", "noise_0.5", "noise_1.0", "noise_2.0", "median", "prior_draw", "wrong_family"]
    body = []
    for v in order:
        cells = [f"`{v}`"]
        for t in ("strong", "middle", "weak"):
            r = ladder[(ladder.method == f"wrong model: {v}") & (ladder.subset == t)].iloc[0]
            cells.append(f"{r['mean'] * 1e4:.1f} (z {r['mean'] / r['se']:.1f})")
        body.append(cells)
    mdg = df[(df.method == "oracle") & (df.metric == "kernel_gain") & df.source.str.contains("dose")]
    weak = mdg[~mdg.subset.isin(["above 0.9", "k 5-30"])]
    hits = weak[weak["mean"] / weak.se > 2]
    page.section("Appendix. Does the score see errors? (wrong-model ladder, kernel score, 100 clouds per stratum)", [
        "Regret ×1e-4 of the true family at a perturbed θ: `noise_d` adds Gaussian noise of d s.d. to log θ; "
        "`median` = train median; `prior_draw` = another θ of the family; `wrong_family` = a θ of another family.",
        "Tiers: strong = τ ≥ 0.9 and cell k ≥ 5; middle = 0.5–0.9 and cell k 3–4; weak = below 0.5 and cell k 2.",
        f"Oracle gain over CSR outside the strong tier: {len(hits)} of {len(weak)} strata resolved (z > 2): "
        + ", ".join(f"{r.family} {r.subset} (z {r['mean'] / r.se:.1f})" for _, r in hits.iterrows())
        + ". Minimum detectable gain per stratum: `kernel_mdg100` rows of summary.csv."])
    page.table("sA_score_ladder", ["Wrong model", "Strong", "Middle", "Weak"], body)


def sA_models(df: pd.DataFrame, page: Page) -> None:
    acc = df[(df.metric == "accuracy") & (df.family == "all") & ~df.method.str.contains(" - ")]
    acc = acc.pivot_table(index="method", columns="subset", values="mean")
    err = df[(df.metric == "rmse_over_sd") & (df.subset == "all") & ~df.method.str.contains(" - ")]
    err = err.pivot_table(index="method", columns="family", values="mean")
    cell = lambda t, m, c: "---" if m not in t.index or c not in t or pd.isna(t.loc[m, c]) else f"{t.loc[m, c]:.3f}"
    order = list(acc.sort_values("all", ascending=False).index) + [m for m in err.index if m not in acc.index]
    body = [[model_name(m), *[cell(acc, m, s) for s in REG], *[cell(err, m, f) for f in FAMS]] for m in order]
    page.section("Appendix. Every model", [f"Accuracy (balanced, {len(C.FAMILIES)} families) and RMSE(log θ)/s.d. per family, test split."])
    page.table("sA_models", ["Model", "Acc. all", "τ=0.5", "τ=0.9", *[C.NAMES[f] for f in FAMS]], body)


def sB_ood(df: pd.DataFrame, page: Page) -> None:
    rows_ = df[df.metric.str.startswith("called ")]
    if rows_.empty:
        return
    for fam, d in rows_.groupby("family"):
        t = d.assign(call=d.metric.str.removeprefix("called ")).pivot_table(index="method", columns="call", values="mean")
        t = t.reindex(columns=C.FAMILIES).sort_values("poisson")
        body = [[model_name(m), *[num(t.loc[m, f], 2) for f in C.FAMILIES]] for m in t.index]
        page.section(f"Appendix. Out of distribution: {C.NAMES.get(fam, fam)}", [
            f"Share of {C.NAMES.get(fam, fam)}'s {int(d.n.iloc[0])} test clouds each classifier calls each family; "
            f"the run never trained on {C.NAMES.get(fam, fam)}. Along its regime coordinate: <run>/ood/{fam}/summary.md."])
        page.table(f"sB_ood_{fam}", ["Classifier", *[C.NAMES[f] for f in C.FAMILIES]], body)


def main() -> None:
    df = build()
    page = Page()
    for view in (s3_grid, s4_ceiling, s5_fidelity, s6_topology, s6_filtrations, m_topology, sA_score, sA_models, sB_ood):
        view(df, page)
    page.write()


if __name__ == "__main__":
    main()
