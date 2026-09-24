#!/usr/bin/env python3
"""Regime analysis of stage 1: in which coordinates is "stage 1 sends this cloud to its own group"
best explained, and where is the boundary?

    python cascade/regime_analysis.py                                   # run `default`, train OOF
    python cascade/regime_analysis.py --run default --stability logreg nn_curves
    sbatch cascade/regime_analysis.sh

For every non-poisson family the target is y = 1{stage 1 routes the cloud to its own group}, on the
run's OUT-OF-FOLD train predictions (370k clouds, never the test set: the cutoff chosen here
defines training pools downstream). Every candidate coordinate is a function of theta (the two
replicates of a theta share it), and is scored in 5-fold cross-validation grouped by theta:

  explained   (LL_null - LL_coord) / (LL_null - LL_ceiling), out-of-fold log-loss, where
              LL_null    = the family's constant routing rate (knows nothing),
              LL_coord   = a monotone (isotonic) fit on a 1-D coordinate, or a small gradient-boosted
                           fit on a 2-D one,
              LL_ceiling = gradient boosting on ALL of theta (+ nbar, + the delta-tildes): no
                           function of theta can do better, because replicates of one theta
                           disagree by chance. 1 = the coordinate is a sufficient description.
  nbar_gain   (1-D only) the share of the explainable log-loss that adding nbar recovers on top of
              the coordinate: 0 = the detection curves at different nbar collapse onto one.
  ambiguous   fraction of the family's clouds whose fitted routing probability lies in [0.1, 0.9]:
              how wide the transition is, in units of clouds (lower = sharper).
Boundaries (1-D only): where the isotonic fit crosses tau in {0.3, 0.5, 0.7, 0.9}, and the fraction
of the family's train clouds past it -- the stage-2/3 training pool that cutoff would leave.

Stability (--stability): the same curves from other stage-1 models, on the VAL split (the only
held-out split every model has; 2000 clouds per family, so these are coarser).

Kinds of coordinate: raw (a model parameter), physical (dimensionless, e.g. cluster overlap),
snr (regime.DERIVED zetas, a pair-count signal-to-noise estimate), delta (delta-tilde; calibrated
on the L-function, which stage 1 also sees -- read a narrow delta win with that in mind).

Output  cascade/results/<run>/regime_analysis/
            cutoffs.json   the frozen training-data rule per family (regime.Cutoffs), from config
                           regime.coordinates and regime.taus
            scores.csv, boundaries.csv, stability.csv, summary.md
            explained.png, detection.png, stability.png
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from common import DEFAULT_CONFIG, REGIME, RESULTS, TARGETS, load_config, load_predictions, manifest, write_json
from regime import fit_cutoff, values

FOLDS, TAUS, CLIP = 5, (0.3, 0.5, 0.7, 0.9), 1e-4
KINDS = ("raw", "physical", "snr", "delta", "combined")
# (coordinate names, kind). A tuple of two names is a 2-D coordinate. kind `combined` is the
# cutoff rule itself (regime.fit_cutoff): x * nbar^a, a fitted, then monotone -- scored like the rest.
CANDIDATES = {
    "thomas": [("sigma", "raw"), ("mu", "raw"), ("kappa", "raw"),
               ("omega", "physical"), (("omega", "nbar"), "physical"), (("omega", "mu"), "physical"),
               ("zeta", "snr"), ("delta_tilde", "delta"), ("delta_tilde_hi", "delta"),
               (("omega", "nbar"), "combined")],
    "nested": [("sigma1", "raw"), ("sigma2", "raw"), ("mu1", "raw"), ("mu2", "raw"),
               ("omega_outer", "physical"), ("omega_inner", "physical"), ("meta_size", "physical"),
               (("omega_inner", "omega_outer"), "physical"),
               ("zeta_outer", "snr"), ("zeta_inner", "snr"), ("zeta", "snr"),
               ("delta_tilde", "delta"), ("delta_tilde_hi", "delta"),
               (("omega_inner", "nbar"), "combined")],
    "matern2": [("R", "raw"), ("lam_p", "raw"), ("core", "physical"), (("core", "nbar"), "physical"),
                ("zeta", "snr"), ("delta_tilde", "delta"), ("delta_tilde_lo", "delta"),
                (("zeta", "nbar"), "combined")],
    "lgcp": [("sigma2", "raw"), ("s", "raw"), ("s_rel", "physical"), (("sigma2", "s_rel"), "physical"),
             ("zeta", "snr"), ("delta_tilde", "delta"), ("delta_tilde_hi", "delta"),
             (("zeta", "nbar"), "combined")],
}
DELTAS = ["delta_tilde", "delta_tilde_lo", "delta_tilde_hi", "delta_tilde_sup"]

# Reference categorical palette, fixed order (dataviz skill), text in ink, never in series colour.
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
KIND_COLOR = dict(zip(KINDS, SERIES))
NBAR_SHADES = ["#9ec5f0", "#4f94e0", "#1a5aa6"]          # one hue, light -> dark: low -> high nbar


# ------------------------------------------------------------------------------------------ data

def load(run: str, split: str) -> pd.DataFrame:
    """Stage-1 predictions of one split joined to the manifest; y = routed to own group."""
    s1 = load_predictions(RESULTS / run / "stage1" / "predictions.npz")
    s1 = s1[s1.split == split]
    if s1.empty:
        raise SystemExit(f"{run}: no stage-1 predictions on `{split}`")
    df = manifest().loc[s1.index].assign(pred=s1.pred)
    df = df[df.family != "poisson"]
    df["y"] = (df.pred == df.family.map(REGIME)).astype(int)
    return df


def coord(df: pd.DataFrame, family: str, names) -> np.ndarray:
    names = (names,) if isinstance(names, str) else names
    return np.column_stack([values(df, family, n) for n in names])


def label(names, kind: str = "") -> str:
    if kind == "combined":
        return f"{names[0]} · n̄^a"
    return names if isinstance(names, str) else " + ".join(names)


# --------------------------------------------------------------------------------------- scoring

def folds(theta: np.ndarray, seed: int = 0) -> np.ndarray:
    uniq = np.unique(theta)
    fold_of = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % FOLDS))
    return np.array([fold_of[t] for t in theta])


def small_hgb():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15,
                                          early_stopping=False, random_state=0)


def oof(X: np.ndarray, y: np.ndarray, fold: np.ndarray, kind: str) -> np.ndarray:
    """Out-of-fold P(y = 1). kind: isotonic (1-D, direction chosen on the training fold) | hgb."""
    p = np.zeros(len(y))
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        if kind == "isotonic":
            p[te] = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(X[tr, 0], y[tr]).predict(X[te, 0])
        elif kind == "null":
            p[te] = y[tr].mean()
        elif kind == "combined":
            from sklearn.linear_model import LogisticRegression
            Z = np.log(X)
            b = LogisticRegression(C=1e6, max_iter=1000).fit(Z[tr], y[tr]).coef_[0]
            u = Z[:, 0] + b[1] / b[0] * Z[:, 1]
            p[te] = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(u[tr], y[tr]).predict(u[te])
        else:
            p[te] = small_hgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return np.clip(p, CLIP, 1 - CLIP)


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ceiling_features(df: pd.DataFrame, family: str) -> np.ndarray:
    cols = [c for c in dict.fromkeys(TARGETS[family] + ["nbar"]) if c in df]
    X = np.log(df[cols].to_numpy(float))
    return np.column_stack([X, df[[c for c in DELTAS if c in df]].to_numpy(float)])


def score_family(df: pd.DataFrame, family: str) -> tuple[list[dict], list[dict], dict]:
    g = df[df.family == family]
    y, fold = g.y.to_numpy(), folds(g.theta.to_numpy())
    ll0 = logloss(oof(np.zeros((len(y), 1)), y, fold, "null"), y)
    llc = logloss(oof(ceiling_features(g, family), y, fold, "hgb"), y)
    span = ll0 - llc
    scores, bounds, fits = [], [], {}
    for names, kind in CANDIDATES[family]:
        X = coord(g, family, names)
        one_d = X.shape[1] == 1
        ll = logloss(oof(X, y, fold, "combined" if kind == "combined" else "isotonic" if one_d else "hgb"), y)
        row = {"family": family, "coordinate": label(names, kind), "kind": kind, "dims": X.shape[1],
               "explained": (ll0 - ll) / span, "logloss": ll, "logloss_null": ll0, "logloss_ceiling": llc,
               "nbar_gain": np.nan, "ambiguous": np.nan}
        if one_d:
            if names != "nbar":
                X2 = np.column_stack([X, g.nbar.to_numpy()])
                row["nbar_gain"] = (ll - logloss(oof(X2, y, fold, "hgb"), y)) / span
            iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(X[:, 0], y)
            p = iso.predict(X[:, 0])
            row["ambiguous"] = float(((p >= 0.1) & (p <= 0.9)).mean())
            fits[label(names)] = (X[:, 0], iso)
            for tau in TAUS:
                b = boundary(iso, X[:, 0], tau)
                past = (p >= tau).mean()
                bounds.append({"family": family, "coordinate": label(names), "kind": kind, "tau": tau,
                               "boundary": b, "direction": "increasing" if iso.increasing_ else "decreasing",
                               "pool_fraction": float(past)})
        scores.append(row)
    return scores, bounds, fits


def boundary(iso: IsotonicRegression, x: np.ndarray, tau: float) -> float:
    """Coordinate value where the monotone fit first reaches tau (NaN if it never does)."""
    grid = np.sort(np.unique(x))
    p = iso.predict(grid)
    hit = np.flatnonzero(p >= tau)
    if not len(hit):
        return float("nan")
    return float(grid[hit[0]] if iso.increasing_ else grid[hit[-1]])


# ----------------------------------------------------------------------------------------- plots

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _xscale(ax, x):
    from matplotlib.ticker import NullFormatter
    if np.nanmin(x) > 0:
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())     # minor labels collide on short ranges
    else:
        ax.set_xscale("symlog", linthresh=1.0)


def binned(x: np.ndarray, y: np.ndarray, bins: int = 15) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    keep = [i for i in range(len(edges) - 1) if (b == i).sum() >= 20]
    return np.array([np.median(x[b == i]) for i in keep]), np.array([y[b == i].mean() for i in keep])


def plot_explained(scores: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt
    fams = list(CANDIDATES)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    for ax, f in zip(axes.flat, fams):
        s = scores[scores.family == f].sort_values("explained")
        ax.barh(range(len(s)), s.explained.clip(lower=0), height=0.62,
                color=[KIND_COLOR[k] for k in s.kind], edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(range(len(s)), s.coordinate, fontsize=8, color=INK)
        for i, v in enumerate(s.explained):
            ax.text(max(v, 0) + 0.01, i, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
        ax.axvline(1.0, color=INK2, linewidth=1, linestyle="--")
        ax.set_xlim(0, 1.12)
        ax.set_title(f, fontsize=11, color=INK, loc="left")
        _style(ax)
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_COLOR[k]) for k in KINDS]
    fig.legend(handles, KINDS, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("How much of stage 1's routing each coordinate explains  (1 = everything θ can explain)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def best_physical(scores: pd.DataFrame, family: str) -> str:
    s = scores[(scores.family == family) & (scores.dims == 1) & (scores.kind != "delta")]
    return s.sort_values("explained").coordinate.iloc[-1]


def plot_detection(df: pd.DataFrame, scores: pd.DataFrame, cutoffs: dict, path) -> None:
    """Per family: routing rate vs the chosen coordinate x, vs the cutoff's combined coordinate
    x * nbar^a, and vs delta-tilde, one line per nbar tercile. Lines that coincide mean the
    coordinate absorbs nbar (collapse); the middle column is what the components are cut on."""
    import matplotlib.pyplot as plt
    fams = list(cutoffs)
    fig, axes = plt.subplots(len(fams), 3, figsize=(14, 3.0 * len(fams)), facecolor=SURFACE)
    for row, f in zip(axes, fams):
        g = df[df.family == f]
        c = cutoffs[f]
        x = values(g, f, c["coordinate"])
        panels = [(c["coordinate"], x),
                  (f"{c['coordinate']} · n̄^{c['nbar_exponent']:.2f}", x * g.nbar.to_numpy() ** c["nbar_exponent"]),
                  ("delta_tilde", g.delta_tilde.to_numpy())]
        terc = pd.qcut(g.nbar, 3, labels=False).to_numpy()
        for ax, (name, v) in zip(row, panels):
            for t in range(3):
                bx, by = binned(v[terc == t], g.y.to_numpy()[terc == t])
                ax.plot(bx, by, color=NBAR_SHADES[t], linewidth=2, marker="o", markersize=4,
                        label=["low n̄", "mid n̄", "high n̄"][t])
            ax.axhline(0.5, color=INK2, linewidth=1, linestyle="--")
            if name.startswith(c["coordinate"] + " ·") and c["u_boundary"].get("0.5") is not None:
                b = np.exp(c["u_boundary"]["0.5"])
                ax.axvline(b, color=INK, linewidth=1)
                ax.text(b, 0.04, f" cutoff at τ=0.5: {b:.3g}", fontsize=7.5, color=INK)
            _xscale(ax, v)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{f}: {name}", fontsize=9.5, color=INK, loc="left")
            ax.set_ylabel("P(routed to own group)", fontsize=8, color=INK2)
            _style(ax)
        row[0].legend(frameon=False, fontsize=7.5, loc="best")
    fig.suptitle("Stage-1 detection curves, split by expected count n̄  (middle: the cutoff coordinate)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stability(curves: dict, scores: pd.DataFrame, path) -> None:
    import matplotlib.pyplot as plt
    fams = list(CANDIDATES)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), facecolor=SURFACE)
    for ax, f in zip(axes.flat, fams):
        name = best_physical(scores, f)
        for i, (run, df) in enumerate(curves.items()):
            g = df[df.family == f]
            x = values(g, f, name)
            bx, by = binned(x, g.y.to_numpy(), bins=10)
            ax.plot(bx, by, color=SERIES[i % len(SERIES)], linewidth=2, marker="o", markersize=4, label=run)
            _xscale(ax, x)
        ax.axhline(0.5, color=INK2, linewidth=1, linestyle="--")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{f}: {name}", fontsize=9.5, color=INK, loc="left")
        ax.set_ylabel("P(routed to own group)", fontsize=8, color=INK2)
        _style(ax)
    axes.flat[0].legend(frameon=False, fontsize=8, title="stage-1 model", title_fontsize=8)
    fig.suptitle("Is the boundary a property of the problem or of the model?  (val split)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------ main

def summary(scores: pd.DataFrame, bounds: pd.DataFrame, stability: pd.DataFrame | None,
            cutoffs: dict) -> str:
    lines = ["# Regime analysis of stage 1", "", "## Cutoffs (cutoffs.json, used to select component training data)", "",
             "| family | rule | explained | " + " | ".join(f"tau {t}: boundary (pool)" for t in
                                                         next(iter(cutoffs.values()))["u_boundary"]) + " |",
             "|---|---|---|" + "---|" * len(next(iter(cutoffs.values()))["u_boundary"])]
    for f, c in cutoffs.items():
        op = ">=" if c["direction"] == "increasing" else "<="
        cells = [f"{np.exp(u):.3g} ({c['pool_fraction'][t]:.0%})" if u is not None else "none"
                 for t, u in c["u_boundary"].items()]
        lines.append(f"| {f} | {c['coordinate']} · n̄^{c['nbar_exponent']:.2f} {op} boundary | "
                     f"{'' if c['explained'] is None else format(c['explained'], '.3f')} | " + " | ".join(cells) + " |")
    lines.append("")
    for f in CANDIDATES:
        s = scores[scores.family == f].sort_values("explained", ascending=False)
        top = s.iloc[0]
        phys = best_physical(scores, f)
        b = bounds[(bounds.family == f) & (bounds.coordinate == phys) & (bounds.tau == 0.5)].iloc[0]
        lines += [f"## {f}",
                  f"- best overall: **{top.coordinate}** ({top.kind}), explained {top.explained:.3f}",
                  f"- best 1-D non-delta: **{phys}**, explained "
                  f"{s[s.coordinate == phys].explained.iloc[0]:.3f}, n̄ adds "
                  f"{s[s.coordinate == phys].nbar_gain.iloc[0]:.3f}, ambiguous {s[s.coordinate == phys].ambiguous.iloc[0]:.2f}",
                  f"- its boundary at tau = 0.5: {phys} {'>=' if b.direction == 'increasing' else '<='} "
                  f"{b.boundary:.4g}, leaving {b.pool_fraction:.0%} of the family's train clouds", ""]
        lines += ["| coordinate | kind | dims | explained | n̄ gain | ambiguous |", "|---|---|---|---|---|---|"]
        lines += [f"| {r.coordinate} | {r.kind} | {r.dims} | {r.explained:.3f} | "
                  f"{'' if np.isnan(r.nbar_gain) else f'{r.nbar_gain:.3f}'} | "
                  f"{'' if np.isnan(r.ambiguous) else f'{r.ambiguous:.2f}'} |" for r in s.itertuples()]
        lines.append("")
    if stability is not None:
        lines += ["## Stability across stage-1 models (val split, best 1-D non-delta coordinate)", "",
                  "| run | family | coordinate | boundary (tau 0.5) | routed to own group |", "|---|---|---|---|---|"]
        lines += [f"| {r.run} | {r.family} | {r.coordinate} | {r._4:.4g} | {r.routed_own:.3f} |"
                  for r in stability.itertuples()]
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="default", help="run whose stage 1 is analysed (needs train OOF)")
    p.add_argument("--split", default="train", help="train (out-of-fold) | val")
    p.add_argument("--stability", nargs="*", default=["logreg", "nn_curves"],
                   help="other runs whose stage 1 is compared on the val split")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="regime.coordinates and regime.taus for the cutoffs")
    args = p.parse_args(argv)
    rc = load_config(args.config)["regime"]
    out = RESULTS / args.run / "regime_analysis"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load(args.run, args.split)
    all_scores, all_bounds, fits = [], [], {}
    for f in CANDIDATES:
        s, b, fits[f] = score_family(df, f)
        all_scores += s
        all_bounds += b
        print(f"{f:8s} scored {len(s)} coordinates ({time.time() - t0:.0f}s)", flush=True)
    scores, bounds = pd.DataFrame(all_scores), pd.DataFrame(all_bounds)
    scores.to_csv(out / "scores.csv", index=False)
    bounds.to_csv(out / "boundaries.csv", index=False)

    stability, curves = None, {}
    if args.stability:
        rows = []
        for run in [args.run] + args.stability:
            v = load(run, "val")
            curves[run] = v
            for f in CANDIDATES:
                name = best_physical(scores, f)
                g = v[v.family == f]
                x = values(g, f, name)
                iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(x, g.y)
                rows.append({"run": run, "family": f, "coordinate": name,
                             "boundary_tau_0.5": boundary(iso, x, 0.5), "routed_own": g.y.mean()})
        stability = pd.DataFrame(rows)
        stability.to_csv(out / "stability.csv", index=False)

    cutoffs = {}
    for f, name in rc["coordinates"].items():
        g = df[df.family == f]
        cutoffs[f] = {"coordinate": name, **fit_cutoff(values(g, f, name), g.nbar.to_numpy(float), g.y.to_numpy(),
                                                       [t for t in rc["taus"] if t > 0])}
        hit = scores[(scores.family == f) & (scores.coordinate == label((name, "nbar"), "combined"))]
        cutoffs[f]["explained"] = float(hit.explained.iloc[0]) if len(hit) else None   # None: not a candidate
    write_json(out / "cutoffs.json", cutoffs)

    (out / "summary.md").write_text(summary(scores, bounds, stability, cutoffs))
    if not args.no_plots:
        plot_explained(scores, out / "explained.png")
        plot_detection(df, scores, cutoffs, out / "detection.png")
        if curves:
            plot_stability(curves, scores, out / "stability.png")
    print((out / "summary.md").read_text())
    print(f"-> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
