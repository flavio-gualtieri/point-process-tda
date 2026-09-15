#!/usr/bin/env python3
# scripts/evaluate_regimes.py
"""Performance across regimes: turn every method's per-pattern DV3
prediction bundles into per-regime tables, across seeds, with paired tests.

Where scripts/evaluate.py compares one scalar per (method, seed), this reads
the predictions_<set>.npz each DV3 run wrote (scripts/train.py with
data.source: dv3; scripts/classical_detection.py), joins them to the DV3
manifests by case_id, and reports -- per method, mean +- s.d. over seeds:

  A  prior-integrated risk per family, overall and cut by delta-tilde band,
     nbar band and their grid (docs/generation_procedure.tex, set A)
  B  every fixed-theta cell: normalised loss, bias and s.d. per target,
     relative bias of the natural parameter; accuracy for classifiers
  C  every rung of every ladder to CSR: the same, as a function of
     delta-tilde; plus calibrated detection power, and the realised size on
     the held-out half of the CSR anchors
  paired  per cell, the seed-wise difference of each method against
     --reference (Wilcoxon signed-rank), Holm-adjusted within each
     (set, family) block of cells -- many cells are tested, and the regime
     claim is "where does it win", so the family-wise correction is part of
     the result, not an afterthought

Output: results/<process>/_regimes/<name>/{summary.json, <task>_<set>.csv,
paired_<task>_<set>.csv, figures/*.png}.

Method specs: `method`, `method@run_tag` (an archived variant) and
`method#tag` (read from results/<process>/<tag>/ instead of the tag the
config's filtration implies -- e.g. a classical detector under raw). Methods
in evaluate.py's RAW_TAG_METHODS and envelope_* default to raw.

A deterministic method (a classical fit/test: one seed dir, or bundles
flagged deterministic) is paired with every seed of the other method.

Usage:
    python scripts/evaluate_regimes.py configs/runs/dv3/thomas/pi_multik.yaml \\
        --methods pi_multik vihrs_checkpointed mincontrast --reference pi_multik
    python scripts/evaluate_regimes.py configs/runs/dv3/classify/pi_multik.yaml \\
        --methods pi_multik vihrs envelope_L envelope_LGF --name detect
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cloudforger.config import load_config  # noqa: E402
from cloudforger.data_generation.filtration import REGISTRY as FILTRATION_REGISTRY  # noqa: E402
from cloudforger.evaluation import regimes as R  # noqa: E402
from cloudforger.evaluation.dv3 import DEFAULT_DV3_ROOT, EVAL_SETS, load_regimes  # noqa: E402
from cloudforger.evaluation.predictions import find_predictions, load_predictions  # noqa: E402
from cloudforger.paths import DEFAULT_RESULTS_ROOT, PROJECT_ROOT, RAW_TAG, ResultsPaths, combined_filtration_tag  # noqa: E402
from evaluate import RAW_TAG_METHODS  # noqa: E402

# Headline metric per task: what a cell is ranked on, and its direction.
HEADLINE = {"params": ("loss", "lower"), "classify": ("accuracy", "higher"), "detect": ("power", "higher")}

# Validated categorical order (dataviz reference palette, light surface):
# method i always gets slot i, in the order given on the command line.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
SCALE_STYLES = ["-", "--", ":", "-."]


# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════

def parse_spec(spec: str, default_tag: str) -> tuple[str, str, str | None]:
    """'method[#tag][@run_tag]' -> (method, filtration_tag, run_tag)."""
    run_tag = None
    if "@" in spec:
        spec, run_tag = spec.split("@", 1)
    tag = default_tag
    if "#" in spec:
        spec, tag = spec.split("#", 1)
    elif spec in RAW_TAG_METHODS or spec.startswith("envelope_") or spec.startswith("vihrs"):
        tag = RAW_TAG
    return spec, tag, run_tag


def seed_dirs(results_root: Path, process: str, tag: str, method: str, run_tag: str | None) -> dict[int, Path]:
    base = results_root / process / tag / method
    if run_tag:
        base = base / "_runs" / run_tag
    out = {}
    for d in sorted(base.glob("seed_*")):
        try:
            out[int(d.name.split("_", 1)[1])] = d
        except ValueError:
            continue
    return out


class RegimeCache:
    """Manifest-derived regimes, loaded once per (set, family set)."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[tuple[str, tuple[str, ...]], Any] = {}

    def for_bundle(self, bundle: dict[str, Any]):
        fams = tuple(dict.fromkeys(str(f) for f in bundle["family"]))
        key = (bundle["set"], fams)
        if key not in self._cache:
            self._cache[key] = load_regimes(bundle["set"], list(fams), self.root)
        return self._cache[key].select(list(bundle["case_id"]))


def method_seed_tables(seed_dir: Path, sets: list[str], cache: RegimeCache) -> tuple[dict, bool, dict]:
    """{task: {row_key: {...}}} for one (method, seed), merged over sets.
    Detection thresholds are calibrated on this same method+seed's C bundle."""
    paths = find_predictions(seed_dir, sets + (["C"] if "C" not in sets else []))
    bundles = {s: load_predictions(p) for s, p in paths.items()}
    deterministic = any(b["meta"].get("deterministic") for b in bundles.values())
    threshold, thr_json = None, {}
    if "C" in bundles:
        score = R.detection_score_of(bundles["C"])
        if score is not None:
            reg_c = cache.for_bundle(bundles["C"])
            if (reg_c["family"] == "poisson").any():
                threshold = R.calibrate_threshold(score, reg_c)
                thr_json = threshold.to_json()
    tables: dict[str, dict] = defaultdict(dict)
    for s in sets:
        if s not in bundles:
            continue
        b = bundles[s]
        for task, rows in R.tables_for_bundle(b, cache.for_bundle(b), threshold).items():
            tables[task].update(rows)
    return dict(tables), deterministic, thr_json


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def flatten(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def aggregate(per_seed: dict[int, dict]) -> dict[str, dict[str, dict]]:
    """{task: {row_key: {"coords", "small", "metrics": {name: {mean, sd, n, values}}}}}."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for seed, tables in per_seed.items():
        for task, rows in tables.items():
            for key, row in rows.items():
                slot = out[task].setdefault(key, {"coords": row["coords"], "small": row["small"], "metrics": {}})
                for name, value in flatten(row["metrics"]).items():
                    slot["metrics"].setdefault(name, {})[seed] = value
    for task in out.values():
        for slot in task.values():
            for name, by_seed in list(slot["metrics"].items()):
                vals = np.array([v for v in by_seed.values() if np.isfinite(v)])
                slot["metrics"][name] = {
                    "mean": float(vals.mean()) if len(vals) else float("nan"),
                    "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "n": int(len(vals)), "values": by_seed,
                }
    return out


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    adj = np.full_like(p, np.nan)
    running, m = 0.0, int(np.isfinite(p).sum())
    for rank, i in enumerate(order):
        if not np.isfinite(p[i]):
            continue
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def paired(agg: dict[str, dict], reference: str, deterministic: dict[str, bool]) -> dict[str, list[dict]]:
    """Per task, one row per (method != reference, row_key) on the headline metric."""
    out: dict[str, list[dict]] = defaultdict(list)
    ref = agg.get(reference, {})
    for method, tasks in agg.items():
        if method == reference:
            continue
        for task, rows in tasks.items():
            metric = HEADLINE[task][0]
            block: dict[str, list[dict]] = defaultdict(list)
            for key, slot in rows.items():
                if key not in ref.get(task, {}) or metric not in slot["metrics"]:
                    continue
                a = slot["metrics"][metric]["values"]
                b = ref[task][key]["metrics"].get(metric, {}).get("values", {})
                if deterministic.get(method) and len(a) == 1:
                    a = {s: next(iter(a.values())) for s in b}
                if deterministic.get(reference) and len(b) == 1:
                    b = {s: next(iter(b.values())) for s in a}
                common = sorted(set(a) & set(b))
                d = np.array([a[s] - b[s] for s in common if np.isfinite(a[s]) and np.isfinite(b[s])])
                p = float("nan")
                if len(d) >= 5 and np.any(d != 0):
                    try:
                        p = float(stats.wilcoxon(d).pvalue)
                    except ValueError:
                        pass
                row = {"method": method, "reference": reference, "row": key, "metric": metric,
                       "n_pairs": int(len(d)), "mean_delta": float(d.mean()) if len(d) else float("nan"),
                       "wins": int((d < 0).sum() if HEADLINE[task][1] == "lower" else (d > 0).sum()),
                       "p": p}
                block["/".join(key.split("/")[:2])].append(row)
            for rows_ in block.values():
                for row, adj in zip(rows_, holm([r["p"] for r in rows_])):
                    row["p_holm"] = adj
                out[task] += rows_
    return dict(out)


# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════

COORD_COLS = ("n_rows", "nbar", "delta_tilde", "scale", "amplitude", "tau_K")


def write_csvs(agg: dict[str, dict], pairs: dict[str, list[dict]], out_dir: Path) -> None:
    rows_by_file: dict[str, list[list]] = defaultdict(list)
    for method, tasks in agg.items():
        for task, rows in tasks.items():
            for key, slot in rows.items():
                set_ = key.split("/")[0]
                coords = [slot["coords"].get(c, "") for c in COORD_COLS]
                for name, m in slot["metrics"].items():
                    rows_by_file[f"{task}_{set_}.csv"].append(
                        [method, key, *coords, int(slot["small"]), name, m["mean"], m["sd"], m["n"]])
    for fname, rows in rows_by_file.items():
        with open(out_dir / fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "row", *COORD_COLS, "small", "metric", "mean", "sd", "n_seeds"])
            w.writerows(sorted(rows, key=lambda r: (r[1], r[0], r[-4])))
    for task, rows in pairs.items():
        by_set: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_set[r["row"].split("/")[0]].append(r)
        for set_, rs in by_set.items():
            with open(out_dir / f"paired_{task}_{set_}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
                w.writeheader()
                w.writerows(rs)


def _fmt(m: dict | None, digits: int = 3) -> str:
    if not m or not np.isfinite(m["mean"]):
        return "—"
    return f"{m['mean']:.{digits}f}±{m['sd']:.{digits}f}" if m["n"] > 1 else f"{m['mean']:.{digits}f}"


def print_tables(agg: dict[str, dict], methods: list[str]) -> None:
    tasks = sorted({t for m in agg.values() for t in m})
    for task in tasks:
        metric = HEADLINE[task][0]
        keys = sorted({k for m in agg.values() for k in m.get(task, {})},
                      key=lambda k: (k.split("/")[0], k.split("/")[1], _sort_key(k)))
        print(f"\n{'=' * 110}\n  {task}: {metric} (mean ± s.d. over seeds)\n{'=' * 110}")
        head = f"  {'row':<46}{'nbar':>7}{'delta':>8}{'scale':>7}"
        print(head + "".join(f"{m[:18]:>20}" for m in methods))
        last_block = None
        for key in keys:
            if "/delta=" in key and "/nbar=" in key:
                continue  # the joint A grid is in the CSV; the console shows the marginals
            block = "/".join(key.split("/")[:2])
            if block != last_block:
                print(f"  {'-' * (39 + 20 * len(methods))}")
                last_block = block
            slot = next((agg[m][task][key] for m in methods if key in agg.get(m, {}).get(task, {})), None)
            c = slot["coords"] if slot else {}
            coords = (f"{c.get('nbar', float('nan')):>7.0f}{c.get('delta_tilde', float('nan')):>8.2f}"
                      f"{c.get('scale', float('nan')):>7.2f}")
            cells = "".join(f"{_fmt(agg.get(m, {}).get(task, {}).get(key, {}).get('metrics', {}).get(metric)):>20}"
                            for m in methods)
            flag = " *" if slot and slot["small"] else ""
            print(f"  {key:<46}{coords}{cells}{flag}")


def _sort_key(key: str) -> tuple:
    parts = dict(p.split("=", 1) for p in key.split("/")[2:] if "=" in p)
    nums = []
    for k in ("cell", "ladder", "level", "nbar", "delta"):
        v = parts.get(k, "")
        try:
            nums.append(float(v.strip("[").split(",")[0]) if v else -1.0)
        except ValueError:
            nums.append(-1.0)
    return tuple(nums)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)


def _delta_axis(ax, ticks=(0.1, 0.25, 0.5, 1, 2, 4, 8)) -> None:
    """Log delta axis with plain labels at the design values (the default
    log formatter prints colliding minor labels like 6x10^-1)."""
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi] or list(ticks)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _metric_axis(ax, metric: str) -> None:
    # Loss spans orders of magnitude once a classical fit diverges near CSR;
    # on a linear axis that flattens every other method onto the baseline.
    if metric in ("loss", "medae"):
        ax.set_yscale("log")


def plot_ladders(agg: dict[str, dict], methods: list[str], task: str, metric: str, fig_dir: Path) -> None:
    """C: metric vs delta-tilde along each ladder; one panel per (family, ladder)."""
    panels: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for m in methods:
        for key, slot in agg.get(m, {}).get(task, {}).items():
            parts = key.split("/")
            if parts[0] != "C" or len(parts) < 4 or not parts[2].startswith("ladder=") or metric not in slot["metrics"]:
                continue  # CSR anchors and the pooled C/_all row are not ladder rungs
            fam, ladder = parts[1], parts[2]
            mm = slot["metrics"][metric]
            panels[(fam, ladder)][m].append((slot["coords"].get("delta_tilde"), mm["mean"], mm["sd"],
                                             slot["coords"].get("nbar")))
    if not panels:
        return
    keys = sorted(panels)
    ncol = min(4, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    width = max(3.6 * ncol, 2.2 * min(len(methods), 3) + 1.0)  # room for the shared legend
    legend_rows = int(np.ceil(len(methods) / max(1, int(width // 2.2))))
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, 3.0 * nrow + 0.3 * legend_rows), squeeze=False)
    for ax, (fam, ladder) in zip(axes.flat, keys):
        nbar = None
        for i, m in enumerate(methods):
            pts = sorted(p for p in panels[(fam, ladder)].get(m, []) if p[0] is not None)
            if not pts:
                continue
            x, y, sd = (np.array([p[j] for p in pts], dtype=float) for j in range(3))
            nbar = pts[0][3]
            color = PALETTE[i % len(PALETTE)]
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4, label=m)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.15, linewidth=0)
        _delta_axis(ax)
        _metric_axis(ax, metric)
        ax.axvline(1.0, color=INK_2, linewidth=0.8, linestyle=":")
        if metric == "power":
            ax.axhline(0.05, color=INK_2, linewidth=0.8, linestyle="--")
            ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{fam}, {ladder}" + (f" (n̄={nbar:g})" if nbar else ""), fontsize=9, color=INK)
        ax.set_xlabel("δ̃ (distance from CSR)", fontsize=8, color=INK_2)
        ax.set_ylabel(metric, fontsize=8, color=INK_2)
        _style(ax)
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, int(width // 2.2)), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.3 * legend_rows + 0.15) / fig.get_figheight()))
    fig.savefig(fig_dir / f"C_{task}_{metric}.png", dpi=150)
    plt.close(fig)


def plot_cells(agg: dict[str, dict], methods: list[str], task: str, metric: str, fig_dir: Path) -> None:
    """B: metric vs target delta per family (rows) and nbar (columns); colour =
    method, line style = scale level (secondary encoding, also in the legend)."""
    data: dict[tuple[str, float], dict[tuple[str, float], list]] = defaultdict(lambda: defaultdict(list))
    for m in methods:
        for key, slot in agg.get(m, {}).get(task, {}).items():
            parts = key.split("/")
            if parts[0] != "B" or len(parts) < 3 or not parts[2].startswith("cell=") or metric not in slot["metrics"]:
                continue
            c = slot["coords"]
            data[(parts[1], c.get("nbar"))][(m, c.get("scale", float("nan")))].append(
                (c.get("delta_tilde"), slot["metrics"][metric]["mean"]))
    if not data:
        return
    fams = sorted({f for f, _ in data})
    nbars = sorted({n for _, n in data if n is not None})
    fig, axes = plt.subplots(len(fams), len(nbars), figsize=(3.6 * len(nbars), 2.9 * len(fams)), squeeze=False)
    for r, fam in enumerate(fams):
        scales = sorted({s for (f, n) in data if f == fam for (_, s) in data[(f, n)]})
        for c, nb in enumerate(nbars):
            ax = axes[r, c]
            for (m, sc), pts in data.get((fam, nb), {}).items():
                pts = sorted(p for p in pts if p[0] is not None)
                if not pts:
                    continue
                i = methods.index(m)
                j = scales.index(sc) if sc in scales else 0
                x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                ax.plot(x, y, color=PALETTE[i % len(PALETTE)], linewidth=2,
                        linestyle=SCALE_STYLES[j % len(SCALE_STYLES)], marker="o", markersize=4,
                        label=f"{m}, scale={sc:g}" if np.isfinite(sc) else m)
            _delta_axis(ax)
            _metric_axis(ax, metric)
            ax.set_title(f"{fam}, n̄={nb:g}", fontsize=9, color=INK)
            ax.set_xlabel("δ̃" if fam != "matern2" else "δ̃ (Matérn: laid out in τ)", fontsize=8, color=INK_2)
            ax.set_ylabel(metric, fontsize=8, color=INK_2)
            _style(ax)
            if c == len(nbars) - 1:
                ax.legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / f"B_{task}_{metric}.png", dpi=150)
    plt.close(fig)


def plot_prior_bands(agg: dict[str, dict], methods: list[str], task: str, metric: str, fig_dir: Path) -> None:
    """A: metric per delta-tilde band, one panel per family, grouped bars by method."""
    fams = sorted({k.split("/")[1] for m in methods for k in agg.get(m, {}).get(task, {})
                   if k.startswith("A/") and "/delta=" in k and "/nbar=" not in k})
    if not fams:
        return
    fig, axes = plt.subplots(1, len(fams), figsize=(3.8 * len(fams), 3.2), squeeze=False)
    width = 0.8 / max(len(methods), 1)
    for ax, fam in zip(axes.flat, fams):
        for i, m in enumerate(methods):
            vals, sds = [], []
            for band in R.DELTA_LABELS:
                slot = agg.get(m, {}).get(task, {}).get(f"A/{fam}/delta={band}")
                mm = slot["metrics"].get(metric) if slot else None
                vals.append(mm["mean"] if mm else np.nan)
                sds.append(mm["sd"] if mm else 0.0)
            x = np.arange(len(R.DELTA_LABELS)) + (i - (len(methods) - 1) / 2) * width
            ax.bar(x, vals, width=width * 0.9, color=PALETTE[i % len(PALETTE)], yerr=sds,
                   error_kw={"elinewidth": 0.8, "ecolor": INK_2}, label=m)
        ax.set_xticks(np.arange(len(R.DELTA_LABELS)))
        ax.set_xticklabels(R.DELTA_LABELS, rotation=35, fontsize=7)
        ax.set_title(f"{fam}: set A by δ̃ band", fontsize=9, color=INK)
        _metric_axis(ax, metric)
        ax.set_ylabel(metric, fontsize=8, color=INK_2)
        _style(ax)
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / f"A_{task}_{metric}.png", dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="a DV3 RunConfig (process name, filtration tag, seeds)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="path.to.field=value")
    ap.add_argument("--methods", nargs="+", required=True, help="method[#tag][@run_tag] ...")
    ap.add_argument("--reference", default=None, help="method spec every other method is paired against "
                                                      "(default: the first of --methods)")
    ap.add_argument("--sets", nargs="+", default=list(EVAL_SETS), choices=list(EVAL_SETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="default: the config's seeds")
    ap.add_argument("--name", default="compare", help="output subdirectory under results/<process>/_regimes/")
    ap.add_argument("--dv3-root", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    filtrations = [FILTRATION_REGISTRY.build(f.name, **f.params) for f in cfg.filtration]
    default_tag = combined_filtration_tag(filtrations)
    results_root = Path(cfg.results_root or DEFAULT_RESULTS_ROOT)
    dv3_root = args.dv3_root or (Path(cfg.data.root) if cfg.data.root else DEFAULT_DV3_ROOT)
    if not dv3_root.is_absolute():
        dv3_root = PROJECT_ROOT / dv3_root
    seeds = set(args.seeds or cfg.seeds)
    reference = args.reference or args.methods[0]
    cache = RegimeCache(dv3_root)

    agg: dict[str, dict] = {}
    deterministic: dict[str, bool] = {}
    thresholds: dict[str, dict] = {}
    for spec in args.methods:
        method, tag, run_tag = parse_spec(spec, default_tag)
        dirs = seed_dirs(results_root, cfg.process.name, tag, method, run_tag)
        per_seed, det = {}, False
        for seed, d in dirs.items():
            if not find_predictions(d, args.sets):
                continue
            is_det_dir = len(dirs) == 1
            if seed not in seeds and not is_det_dir:
                continue
            tables, det_b, thr = method_seed_tables(d, args.sets, cache)
            per_seed[seed] = tables
            det = det or det_b or is_det_dir
            if thr:
                thresholds.setdefault(spec, {})[seed] = thr
        if not per_seed:
            print(f"  ! {spec}: no prediction bundles under {results_root / cfg.process.name / tag / method}"
                  f"{' @' + run_tag if run_tag else ''} for seeds {sorted(seeds)}; skipping")
            continue
        print(f"  {spec}: {len(per_seed)} seed(s) from {tag}/{method}{' (deterministic)' if det else ''}")
        agg[spec] = aggregate(per_seed)
        deterministic[spec] = det

    if not agg:
        raise SystemExit("no prediction bundles found for any method -- run training with data.source: dv3 first")
    methods = [m for m in args.methods if m in agg]
    if reference not in agg:
        print(f"  ! reference {reference!r} has no results; pairing against {methods[0]!r}")
        reference = methods[0]

    out_dir = results_root / cfg.process.name / "_regimes" / args.name
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print_tables(agg, methods)
    pairs = paired(agg, reference, deterministic)
    for task, rows in pairs.items():
        sig = [r for r in rows if np.isfinite(r.get("p_holm", np.nan)) and r["p_holm"] < 0.05]
        print(f"\n  paired vs {reference} ({task}, {HEADLINE[task][0]}): {len(sig)}/{len(rows)} cells differ at "
              f"Holm-adjusted p < 0.05 (seed-paired Wilcoxon; >= 5 pairs needed)")
    write_csvs(agg, pairs, out_dir)

    for task in sorted({t for m in agg.values() for t in m}):
        metric = HEADLINE[task][0]
        plot_ladders(agg, methods, task, metric, fig_dir)
        plot_cells(agg, methods, task, metric, fig_dir)
        plot_prior_bands(agg, methods, task, metric, fig_dir)
        if task == "params":
            plot_ladders(agg, methods, task, "fail_rate", fig_dir)

    summary = {
        "config": str(args.config), "process": cfg.process.name, "methods": methods,
        "reference": reference, "sets": args.sets, "seeds": sorted(seeds),
        "deterministic": deterministic, "detection_thresholds": thresholds,
        "bands": {"delta": R.DELTA_LABELS, "nbar": R.NBAR_LABELS},
        "tables": {m: {t: {k: {"coords": v["coords"], "small": v["small"],
                                "metrics": {n: {kk: vv for kk, vv in mm.items() if kk != "values"}
                                            for n, mm in v["metrics"].items()}}
                            for k, v in rows.items()}
                       for t, rows in tasks.items()}
                   for m, tasks in agg.items()},
        "paired": pairs,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"\nSaved regime tables -> {out_dir}")
    return summary


if __name__ == "__main__":
    main()
