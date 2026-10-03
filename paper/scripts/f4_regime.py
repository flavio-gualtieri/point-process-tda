"""F4: the regime boundary on one axis -- detection, ParamNet's calls and the gain over CSR.

u~ = (u - u*(0.5)) / (u*(0.9) - u*(0.5)), u and u* from compare's frozen cutoffs (common.x_of): 0 at the
tau = 0.5 boundary, 1 at tau = 0.9, positive on the structured side, so the six families with a
coordinate share one axis (cell has none: s_regime). Families are pooled with equal weight.
    (a) detection power on the test split: the CSR test and the DETECTORS as tests of the cutoffs' size
        (common.detected), in bins of x where every family has patterns. Thin grey: the CSR test per family.
    (b) the headline classifier's argmax call on the test split: the true family, another family of the
        same mechanism (paper.yaml coarse), another mechanism, Poisson.
    (c) gain over CSR, S(csr) - S(model), on the main evaluation's clouds per stratum, as a share of the
        true model's mean gain beyond tau = 0.9 in the same family; plotted at the stratum's median x.
        95% bootstrap over clouds, resampled within (family, stratum) and shared by the three series.

f4_detection is (a) alone at column width, for the main body.

    python paper/scripts/f4_regime.py        # -> paper/figs/f4_regime.pdf, f4_detection.pdf
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C  # first: it sets the bank (paper config `data`) before cloudforger reads it

DETECTORS = [C.CSR_TEST, "hgb_ph", "fusion_curves_alpha_dtm10"]
LABELS = {C.CSR_TEST: "$L$ test", "hgb_ph": "Trees, PH"}
COLOURS = {C.CSR_TEST: C.SERIES[1], "hgb_ph": C.SERIES[2], "fusion_curves_alpha_dtm10": C.SERIES[0]}
XLABEL = r"$\tilde u$"
STRATA = ["below 0.5", "0.5-0.9", "above 0.9"]
EDGES = np.arange(-3, 3.01, 0.5)
MIN_BIN = 30                     # patterns per (family, bin) for the bin to count
BOOT = 1000


def regime_families() -> list[str]:
    return [f for f in C.FAMILIES if f in C.cutoffs()]


def test_rows() -> pd.DataFrame:
    """Test patterns of the families with a coordinate, with their x."""
    r, cuts = C.rows(), C.cutoffs()
    t = r[(r.split == "test") & r.family.isin(regime_families())].copy()
    t["x"] = np.nan
    for f in regime_families():
        m = (t.family == f).to_numpy()
        t.loc[m, "x"] = C.x_of(t[m], f, cuts)
    return t


def pooled_bins(t: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Bin centres and the equal-weight family mean of y, where every family has MIN_BIN patterns."""
    at = np.digitize(t.x.to_numpy(), EDGES) - 1
    per = []
    for f in regime_families():
        m = (t.family == f).to_numpy()
        per.append([y[m & (at == b)].mean() if (m & (at == b)).sum() >= MIN_BIN else np.nan
                    for b in range(len(EDGES) - 1)])
    per = np.array(per)
    ok = np.isfinite(per).all(0)
    return (0.5 * (EDGES[1:] + EDGES[:-1]))[ok], per[:, ok].mean(0)


def detection(ax, t: pd.DataFrame) -> None:
    names = C.cfg().get("names", {})
    for model in DETECTORS:
        y = C.detected(model).reindex(t.index).to_numpy(float)
        if model == C.CSR_TEST:                                  # each family alone, behind the pooled lines
            for f in regime_families():
                m = (t.family == f).to_numpy()
                at = np.digitize(t.x[m], EDGES) - 1
                keep = [b for b in range(len(EDGES) - 1) if (at == b).sum() >= MIN_BIN]
                ax.plot([t.x[m][at == b].median() for b in keep], [y[m][at == b].mean() for b in keep],
                        color=COLOURS[model], lw=0.5, alpha=0.3, zorder=1)
        ax.plot(*pooled_bins(t, y), color=COLOURS[model], lw=1.6, zorder=3)
    size = next(c["size"] for c in C.cutoffs().values())
    ax.axhline(size, color=C.AXIS, lw=0.6, ls=":")
    ax.set(ylim=(0, 1.02), yticks=[0, 0.5, 1])
    C.legend_below(ax, [LABELS.get(m, names.get(m, m)) for m in DETECTORS], [COLOURS[m] for m in DETECTORS])


def calls(ax, t: pd.DataFrame) -> None:
    coarse, model = C.cfg()["coarse"], C.cfg()["story"]["headline"]
    post = C.classifier(model).reindex(t.index)
    call = pd.Series(np.array(C.FAMILIES)[post[C.FAMILIES].to_numpy().argmax(1)], index=t.index)
    kinds = {"True family": call == t.family,
             "Same mechanism": (call != t.family) & (call.map(coarse) == t.family.map(coarse)),
             "Other mechanism": (call != "poisson") & (call.map(coarse) != t.family.map(coarse)),
             "Poisson": call == "poisson"}
    colours = [C.SERIES[0], C.BLUE_RAMP[3], C.SERIES[3], C.GRID]
    xs, ys = zip(*(pooled_bins(t, v.to_numpy(float)) for v in kinds.values()))
    ax.stackplot(xs[0], *ys, colors=colours, lw=0)
    ax.set(ylim=(0, 1), yticks=[0, 0.5, 1])
    C.legend_below(ax, list(kinds), colours, ncol=2)


def gain_shares(strata: dict[str, list[str]]) -> tuple[list[str], pd.DataFrame, np.ndarray, np.ndarray]:
    """Gain over CSR, S(csr) - S(model), on the main evaluation's clouds of the families in `strata`
    (family -> its strata, weakest first): the true model, then paper.yaml family_pair's pipelines, failed
    fits dropped so every series has the same clouds. Each (family, stratum, series) is the series' mean
    gain as a share of the true model's in the family's last stratum. Returns the series names, the
    clouds (family, stratum, fit_case_id), the estimate (family, stratum, series) and BOOT resamples of
    it, over clouds within (family, stratum), shared by the series."""
    cfg, fams = C.cfg(), list(strata)
    cl = pd.read_csv(C.evaluation("main") / "clouds.csv")
    cl = cl[cl.family.isin(fams) & ~cl.failed.astype(bool)]
    pipe = cfg["family_pair"]                                   # [ParamNet, minimum contrast]
    wide = {"True model": cl[cl.variant == pipe[0]].set_index("scored_case_id").kernel_gain}
    for v in pipe:
        g = cl[cl.variant == v].set_index("scored_case_id")
        wide[cfg["names"].get(v, v)] = g.kernel_gain - g.kernel_regret
    wide = pd.DataFrame(wide).dropna()
    meta = cl[cl.variant == pipe[0]].set_index("scored_case_id").loc[wide.index, ["family", "stratum", "fit_case_id"]]
    groups = {(f, s): np.flatnonzero(((meta.family == f) & (meta.stratum == s)).to_numpy())
              for f in fams for s in strata[f]}
    vals = wide.to_numpy()

    def shares(pick: dict) -> np.ndarray:
        out = np.zeros((len(fams), max(map(len, strata.values())), vals.shape[1]))
        for i, f in enumerate(fams):
            norm = vals[pick[(f, strata[f][-1])], 0].mean()
            for j, s in enumerate(strata[f]):
                out[i, j] = vals[pick[(f, s)]].mean(0) / norm
        return out

    rng = np.random.default_rng(0)
    boot = np.array([shares({k: rng.choice(v, len(v)) for k, v in groups.items()}) for _ in range(BOOT)])
    return list(wide.columns), meta, shares(groups), boot


GAIN_COLOURS = [C.INK, C.SERIES[0], C.SERIES[1]]


def gains(ax) -> None:
    """Gain over CSR per stratum, families averaged (see the module doc)."""
    cuts, r, fams = C.cutoffs(), C.rows(), regime_families()
    names, meta, est, boot = gain_shares({f: STRATA for f in fams})
    est, (lo, hi) = est.mean(0), np.percentile(boot.mean(1), [2.5, 97.5], axis=0)
    x = pd.concat([pd.Series(C.x_of(r.loc[g.fit_case_id], f, cuts), index=g.fit_case_id)
                   for f, g in meta.groupby("family")])
    at = np.array([np.median(x[meta.fit_case_id[meta.stratum == s]]) for s in STRATA])
    for j, name in enumerate(names):
        ax.errorbar(at + (j - 1) * 0.12, est[:, j], yerr=[est[:, j] - lo[:, j], hi[:, j] - est[:, j]],
                    color=GAIN_COLOURS[j], marker="o", ms=3.5, mec="white", mew=0.5, lw=1.4, elinewidth=1.0,
                    capsize=0, zorder=3 + j)
    ax.axhline(0, color=C.AXIS, lw=0.6)
    ax.set(yticks=[0, 0.5, 1])
    C.legend_below(ax, names, GAIN_COLOURS)


def boundaries(ax) -> None:
    ax.set(xlim=(EDGES[0], EDGES[-1]))
    ax.set_xlabel(XLABEL, labelpad=1)
    for v, ls in ((0, "-"), (1, "--")):
        ax.axvline(v, color=C.INK2, lw=0.6, ls=ls, zorder=5)


def detection_only(t: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(C.COLUMN_WIDTH, 1.75), layout="constrained")
    detection(ax, t)
    ax.set_ylabel("Detection power", labelpad=2)
    boundaries(ax)
    C.save(fig, "f4_detection")


def main() -> None:
    C.style()
    t = test_rows()
    detection_only(t)
    fig, axes = plt.subplots(1, 3, figsize=(C.TEXT_WIDTH, 2.2), layout="constrained")
    detection(axes[0], t)
    calls(axes[1], t)
    gains(axes[2])
    for ax, title in zip(axes, ["(a) Detection power", "(b) ParamNet's call", "(c) Relative gain over CSR"]):
        ax.set_title(title, loc="left")
        boundaries(ax)
    C.save(fig, "f4_regime")


if __name__ == "__main__":
    main()
