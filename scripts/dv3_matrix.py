#!/usr/bin/env python3
# scripts/dv3_matrix.py
"""(Re)write the "DV3 classical" sheet of docs/status/experiment_matrix.xlsx:
status and per-regime results of the classical baselines on DV3.

The sheet has three blocks, one row per (arm[, family]):

  Parameter estimation  normalised loss L on A overall, cut by delta-tilde band
                        and by nbar band (the regime axes of set A), plus the
                        whole-set loss on B and C
  Classification        A balanced accuracy, per-family recall, accuracy by
                        delta-tilde band (structured families pooled), B and C
  Detection (CSR)       held-out size and power on the C ladders by
                        delta-tilde band, at the empirical 5% threshold

Every number is mean +- sample SD (ddof=1) over seeds of one per-seed value,
computed by the SAME per-seed table builder scripts/evaluate_regimes.py uses
(method_seed_tables), so the sheet and the regime tables cannot disagree.
Pooled band columns weight each regime row by its pattern count. A value
marked * has a regime row with fewer than 20 patterns behind it.

Rows are listed whether or not they have run, so the sheet doubles as the
to-run list: "ready to run" rows name the SLURM task that produces them.
Nothing is trained or recomputed here -- it only reads results/ and writes
the workbook. Re-run it whenever jobs land; the other sheets (the DV2 audit
of 2026-09-11) are left untouched.

Usage:
    python scripts/dv3_matrix.py
    python scripts/dv3_matrix.py --dry-run          # print the tables, write nothing

Needs openpyxl, which cloud-env does not ship: `pip install openpyxl` into the
env, or put a directory containing it on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cloudforger.evaluation.dv3 import DEFAULT_DV3_ROOT, DELTA_EDGES, FAMILIES, band_index  # noqa: E402
from cloudforger.evaluation.predictions import find_predictions  # noqa: E402
from cloudforger.evaluation.regimes import DELTA_LABELS, NBAR_LABELS  # noqa: E402
from cloudforger.paths import DEFAULT_RESULTS_ROOT, RAW_TAG  # noqa: E402
from evaluate_regimes import RegimeCache, method_seed_tables  # noqa: E402

MATRIX = ROOT / "docs" / "status" / "experiment_matrix.xlsx"
SHEET = "DV3 classical"
SEEDS = list(range(9371, 9381))  # every configs/runs/dv3/*.yaml
PARAM_FAMILIES = [f for f in FAMILIES if f != "poisson"]
STRUCTURED = PARAM_FAMILIES
TRAIN_SCRIPT = "slurm/dv3_classical_train.sh"
PREP_SCRIPT = "slurm/dv3_classical_prep.sh"
SUBMIT_SCRIPT = "slurm/dv3_classical_submit.sh"
# Groups in GPU-array order: inference tasks t = arm * 4 + family (0-27) come
# first, classification t = 28 + arm; prep task = pass * 4 + family, 12 for _classify.
GROUPS = [*PARAM_FAMILIES, "classify"]


def task_ids(a_idx: int, group: str) -> str:
    if group == "classify":
        return f"{TRAIN_SCRIPT} task {len(NEURAL_ARMS) * len(PARAM_FAMILIES) + a_idx} (after {PREP_SCRIPT} task 12)"
    g = PARAM_FAMILIES.index(group)
    pass_ = 0 if a_idx == 0 else (1 if NEURAL_ARMS[a_idx].grid == "0.25" else 2)
    return f"{TRAIN_SCRIPT} task {a_idx * len(PARAM_FAMILIES) + g} (after {PREP_SCRIPT} task {pass_ * len(PARAM_FAMILIES) + g})"


@dataclass(frozen=True)
class Arm:
    key: str                  # == ARM_TAG in slurm/dv3_classical_train.sh
    inputs: str
    grid: str                 # F/G/J radius grid r_max ("—" when L only)
    subdir: str | None        # results subdir, same for both tasks unless overridden below
    pair: str | None = None   # arms sharing a pair differ only in the F/G/J grid; choose on validation


# Keep in the order (and with the subdirs) of slurm/dv3_classical_train.sh.
# The 11 pair/triple/J arms below cover the rest of the 15 non-empty subsets
# of {L,F,G,J}; only the original 4 (L, LFGJ, G, F) also have an 0.08 grid.
NEURAL_ARMS = [
    Arm("L", "L(r)−r + n(x)", "—", None),
    Arm("LFGJ", "L, F, G, J + n(x)", "0.25", "vihrs_lfgj", "LFGJ"),
    Arm("LFGJ_r080", "L, F, G, J + n(x)", "0.08", "vihrs_lfgj_r080", "LFGJ"),
    Arm("G", "G + n(x)", "0.25", "vihrs_g", "G"),
    Arm("F", "F + n(x)", "0.25", "vihrs_f", "F"),
    Arm("G_r080", "G + n(x)", "0.08", "vihrs_g_r080", "G"),
    Arm("F_r080", "F + n(x)", "0.08", "vihrs_f_r080", "F"),
    Arm("J", "J + n(x)", "0.25", "vihrs_j"),
    Arm("LF", "L, F + n(x)", "0.25", "vihrs_lf"),
    Arm("LG", "L, G + n(x)", "0.25", "vihrs_lg"),
    Arm("LJ", "L, J + n(x)", "0.25", "vihrs_lj"),
    Arm("FG", "F, G + n(x)", "0.25", "vihrs_fg"),
    Arm("FJ", "F, J + n(x)", "0.25", "vihrs_fj"),
    Arm("GJ", "G, J + n(x)", "0.25", "vihrs_gj"),
    Arm("LFG", "L, F, G + n(x)", "0.25", "vihrs_lfg"),
    Arm("LFJ", "L, F, J + n(x)", "0.25", "vihrs_lfj"),
    Arm("LGJ", "L, G, J + n(x)", "0.25", "vihrs_lgj"),
    Arm("FGJ", "F, G, J + n(x)", "0.25", "vihrs_fgj"),
]
L_SUBDIR = {"params": "vihrs_checkpointed", "classify": "vihrs"}

# Non-neural classical estimators with a DV3 config (CPU; not in the GPU batch).
MINCONTRAST = {"thomas": ["mincontrast", "mincontrast_g"], "nested": ["mincontrast_nested", "mincontrast_g_nested"]}
ENVELOPES = ["envelope_L", "envelope_G", "envelope_F", "envelope_LGF"]

STATUS_FILL = {"complete": "C6EFCE", "partial seeds": "FFEB9C", "ready to run": "DDEBF7", "not scheduled": "FFC7CE"}
STATUS_FONT = {"complete": "006100", "partial seeds": "9C5700", "ready to run": "1F4E78", "not scheduled": "9C0006"}


# ══════════════════════════════════════════════════════════════════════════════
# Reading one seed
# ══════════════════════════════════════════════════════════════════════════════

def _json(path: Path) -> dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def seed_dirs(method_dir: Path, deterministic: bool = False) -> dict[int, Path]:
    """Finished seeds: results.pt for trained/fitted methods, prediction
    bundles for the deterministic envelope tests (seed_0 only)."""
    out = {}
    for d in sorted(method_dir.glob("seed_*")):
        try:
            seed = int(d.name.split("_", 1)[1])
        except ValueError:
            continue
        done = find_predictions(d, ["A", "B", "C"]) if deterministic else (d / "results.pt").exists()
        if done and (deterministic or seed in SEEDS):
            out[seed] = d
    return out


def _tables(seed_dir: Path, sets: list[str], cache: RegimeCache) -> tuple[dict, dict]:
    if not find_predictions(seed_dir, sets):
        return {}, {}
    tables, _, thr = method_seed_tables(seed_dir, sets, cache)
    return tables, thr


def _cell(row: dict | None, metric: str) -> tuple[float | None, bool]:
    if not row:
        return None, False
    return row["metrics"].get(metric), bool(row.get("small"))


def _pooled(rows: list[dict], metric: str) -> tuple[float | None, bool]:
    """Pattern-count-weighted mean of `metric` over regime rows."""
    vals = [(r["metrics"].get(metric), r["coords"].get("n_rows", 0), r.get("small")) for r in rows]
    vals = [(v, n, s) for v, n, s in vals if v is not None and np.isfinite(v) and n]
    if not vals:
        return None, False
    w = np.array([n for _, n, _ in vals], dtype=float)
    return float(np.dot([v for v, _, _ in vals], w) / w.sum()), any(s for *_, s in vals)


def params_seed(seed_dir: Path, family: str, cache: RegimeCache) -> dict[str, tuple[float | None, bool]]:
    js = _json(seed_dir / "results.json")
    ev = js.get("eval_sets") or {}
    tables, _ = _tables(seed_dir, ["A"], cache)
    t = tables.get("params", {})
    out = {"val": (js.get("best_val_loss"), False)}
    out["A"] = _cell(t.get(f"A/{family}/all"), "loss")
    out["A_fail"] = _cell(t.get(f"A/{family}/all"), "fail_rate")
    for lab in DELTA_LABELS:
        out[f"A_d{lab}"] = _cell(t.get(f"A/{family}/delta={lab}"), "loss")
    for lab in NBAR_LABELS:
        out[f"A_n{lab}"] = _cell(t.get(f"A/{family}/nbar={lab}"), "loss")
    for s in ("B", "C"):
        out[s] = ((ev.get(s) or {}).get("loss"), False)
    return out


def _detect_cols(tables: dict, thr: dict) -> dict[str, tuple[float | None, bool]]:
    out: dict[str, tuple[float | None, bool]] = {}
    sizes = list((thr.get("size_heldout") or {}).values())
    out["size"] = (float(np.mean(sizes)) if sizes else None, False)
    rungs = [r for k, r in tables.get("detect", {}).items()
             if k.startswith("C/") and k.split("/")[1] in STRUCTURED]
    bands = band_index(np.array([r["coords"].get("delta_tilde", np.nan) for r in rungs], dtype=float), DELTA_EDGES)
    for b, lab in enumerate(DELTA_LABELS):
        out[f"C_d{lab}"] = _pooled([r for r, bb in zip(rungs, bands) if bb == b], "power")
    return out


def classify_seed(seed_dir: Path, cache: RegimeCache) -> dict[str, tuple[float | None, bool]]:
    js = _json(seed_dir / "results.json")
    ev = js.get("eval_sets") or {}
    tables, thr = _tables(seed_dir, ["A", "C"], cache)
    t = tables.get("classify", {})
    out = {"val": (js.get("best_val_accuracy"), False)}
    a = t.get("A/_all")
    out["A"] = _cell(a, "accuracy")
    recall = (a or {}).get("metrics", {}).get("recall", {})
    for fam in FAMILIES:
        out[f"R_{fam}"] = (recall.get(fam), False)
    for lab in DELTA_LABELS:
        out[f"A_d{lab}"] = _pooled([t[k] for fam in STRUCTURED if (k := f"A/{fam}/delta={lab}") in t], "accuracy")
    for s in ("B", "C"):
        out[s] = ((ev.get(s) or {}).get("accuracy"), False)
    out.update(_detect_cols(tables, thr))
    return out


def detect_seed(seed_dir: Path, cache: RegimeCache) -> dict[str, tuple[float | None, bool]]:
    tables, thr = _tables(seed_dir, ["C"], cache)
    return _detect_cols(tables, thr)


def aggregate(per_seed: dict[int, dict]) -> dict[str, dict[str, Any]]:
    cols: dict[str, dict[str, Any]] = {}
    for vals in per_seed.values():
        for col, (v, small) in vals.items():
            slot = cols.setdefault(col, {"values": [], "small": False})
            if v is not None and np.isfinite(v):
                slot["values"].append(float(v))
            slot["small"] |= small
    for slot in cols.values():
        v = np.array(slot["values"])
        slot["mean"] = float(v.mean()) if len(v) else None
        slot["sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        slot["n"] = len(v)
    return cols


def fmt(agg: dict[str, dict], col: str, nd: int = 4) -> str:
    slot = agg.get(col)
    if not slot or slot["mean"] is None:
        return ""
    s = f"{slot['mean']:.{nd}f} ± {slot['sd']:.{nd}f}"
    return s + ("*" if slot["small"] else "")


# ══════════════════════════════════════════════════════════════════════════════
# Rows
# ══════════════════════════════════════════════════════════════════════════════

def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def status_of(n_done: int, expected: int, scheduled: bool) -> str:
    if n_done >= expected:
        return "complete"
    if n_done:
        return "partial seeds"
    return "ready to run" if scheduled else "not scheduled"


def _mark_val_selected(rows: list[dict], higher_is_better: bool) -> None:
    """Within each grid pair (same block key), tag the arm whose mean
    validation metric wins -- the only legitimate way to pick a grid."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("pair"):
            groups.setdefault((r.get("family"), r["pair"]), []).append(r)
    for members in groups.values():
        scored = [(r["agg"]["val"]["mean"], r) for r in members
                  if r["agg"].get("val", {}).get("mean") is not None]
        if len(scored) < 2:
            continue
        best = (max if higher_is_better else min)(scored, key=lambda x: x[0])[1]
        best["val_selected"] = True


def build_rows(results_root: Path, cache: RegimeCache) -> dict[str, list[dict]]:
    params_rows, classify_rows, detect_rows = [], [], []
    for a_idx, arm in enumerate(NEURAL_ARMS):
        for group in GROUPS:
            task = "classify" if group == "classify" else "params"
            process = "dv3_classify" if task == "classify" else f"dv3_{group}"
            subdir = arm.subdir or L_SUBDIR[task]
            method_dir = results_root / process / RAW_TAG / subdir
            dirs = seed_dirs(method_dir)
            per_seed = {s: (classify_seed(d, cache) if task == "classify" else params_seed(d, group, cache))
                        for s, d in dirs.items()}
            row = {
                "arm": arm, "family": None if task == "classify" else group, "pair": arm.pair,
                "status": status_of(len(dirs), len(SEEDS), True), "n": len(dirs), "agg": aggregate(per_seed),
                "results": _rel(method_dir),
                "by": task_ids(a_idx, group),
            }
            (classify_rows if task == "classify" else params_rows).append(row)
            if task == "classify":
                detect_rows.append({**row, "detector": f"vihrs CNN [{arm.inputs}], grid {arm.grid}",
                                    "score": "1 − P(poisson)"})

    for family, methods in MINCONTRAST.items():
        for m in methods:
            method_dir = results_root / f"dv3_{family}" / RAW_TAG / m
            dirs = seed_dirs(method_dir)
            per_seed = {s: params_seed(d, family, cache) for s, d in dirs.items()}
            params_rows.append({
                "arm": Arm(m, "minimum contrast on " + ("g" if "_g" in m else "K"), "—", m),
                "family": family, "pair": None,
                "status": status_of(len(dirs), len(SEEDS), False), "n": len(dirs), "agg": aggregate(per_seed),
                "results": _rel(method_dir),
                "by": f"python scripts/train.py configs/runs/dv3/{family}/{m}.yaml  (CPU, not in the GPU batch)",
            })

    for env in ENVELOPES:
        method_dir = results_root / "dv3_classify" / RAW_TAG / env
        dirs = seed_dirs(method_dir, deterministic=True)
        detect_rows.append({
            "arm": Arm(env, env, "—", env), "detector": f"global envelope test, {env.split('_', 1)[1]}",
            "score": "studentised S", "status": status_of(len(dirs), 1, False), "n": len(dirs),
            "agg": aggregate({s: detect_seed(d, cache) for s, d in dirs.items()}),
            "results": _rel(method_dir),
            "by": "python scripts/classical_detection.py --jobs 16  (CPU, deterministic: seed_0)",
        })

    params_rows.sort(key=lambda r: PARAM_FAMILIES.index(r["family"]))  # stable: arm order within family
    _mark_val_selected(params_rows, higher_is_better=False)
    _mark_val_selected(classify_rows, higher_is_better=True)
    return {"params": params_rows, "classify": classify_rows, "detect": detect_rows}


# ══════════════════════════════════════════════════════════════════════════════
# Sheet
# ══════════════════════════════════════════════════════════════════════════════

def _val(r: dict, nd: int = 4) -> str:
    s = fmt(r["agg"], "val", nd)
    return s + ("  ✓ val-selected" if r.get("val_selected") else "") if s else ""


def params_table(rows: list[dict]) -> tuple[list[str], list[list[Any]]]:
    head = ["Family", "Arm", "Inputs", "F/G/J grid r_max", "Status", "Seeds", "Val loss (selection)", "A: all",
            *[f"A δ̃ {x}" for x in DELTA_LABELS], *[f"A n̄ {x}" for x in NBAR_LABELS],
            "A fail rate", "B: whole set", "C: whole set", "Results", "Produced by"]
    body = [[r["family"], r["arm"].key, r["arm"].inputs, r["arm"].grid, r["status"], f"{r['n']}/{len(SEEDS)}",
             _val(r), fmt(r["agg"], "A"), *[fmt(r["agg"], f"A_d{x}") for x in DELTA_LABELS],
             *[fmt(r["agg"], f"A_n{x}") for x in NBAR_LABELS], fmt(r["agg"], "A_fail", 3),
             fmt(r["agg"], "B"), fmt(r["agg"], "C"), r["results"], r["by"]] for r in rows]
    return head, body


def classify_table(rows: list[dict]) -> tuple[list[str], list[list[Any]]]:
    head = ["Arm", "Inputs", "F/G/J grid r_max", "Status", "Seeds", "Val accuracy (selection)", "A: balanced acc",
            *[f"A recall {f}" for f in FAMILIES], *[f"A δ̃ {x} (structured)" for x in DELTA_LABELS],
            "B: whole set", "C: whole set", "Results", "Produced by"]
    body = [[r["arm"].key, r["arm"].inputs, r["arm"].grid, r["status"], f"{r['n']}/{len(SEEDS)}", _val(r),
             fmt(r["agg"], "A"), *[fmt(r["agg"], f"R_{f}") for f in FAMILIES],
             *[fmt(r["agg"], f"A_d{x}") for x in DELTA_LABELS], fmt(r["agg"], "B"), fmt(r["agg"], "C"),
             r["results"], r["by"]] for r in rows]
    return head, body


def detect_table(rows: list[dict]) -> tuple[list[str], list[list[Any]]]:
    head = ["Detector", "Score", "Status", "Seeds", "Size (held-out CSR anchors)",
            *[f"C power δ̃ {x}" for x in DELTA_LABELS], "Results", "Produced by"]
    body = [[r["detector"], r["score"], r["status"], f"{r['n']}/{1 if r['arm'].key in ENVELOPES else len(SEEDS)}",
             fmt(r["agg"], "size", 3), *[fmt(r["agg"], f"C_d{x}", 3) for x in DELTA_LABELS],
             r["results"], r["by"]] for r in rows]
    return head, body


NOTES = [
    "DV3 (generated 2026-09-14; docs/generation_procedure.tex). Families poisson, thomas, nested, matern2, lgcp. "
    "Training: data/dv3/train, 10,000 per family, train/val split fixed at generation (every method and seed trains "
    "on the same rows). Evaluation: A = 5,000 per family from the prior (risk over the prior), B = fixed-θ cells × 400 "
    "replicates, C = ladders down to exact CSR + CSR anchors. δ̃ = distance from CSR in units of the 5% critical value "
    "of the studentised L-envelope test (δ̃ ≈ 1: edge of detectability); n̄ = expected count.",
    "Values: mean ± sample SD (ddof=1) over seeds (target 10: 9371–9380) of per-seed values, from the same per-seed "
    "tables as scripts/evaluate_regimes.py. Pooled columns weight regime rows by pattern count. * = a regime row with "
    "< 20 patterns contributes. Blank = no result yet. Nothing on this sheet is comparable with the DV2 sheets that "
    "follow (different data, prior, split and test sets).",
    "Parameter estimation: normalised loss L = MSE on log-then-z-scored targets (train-split statistics), mean over "
    "the family's targets; lower is better, ≈ 1 = predicting the train mean. Classification: 5-way accuracy; A is "
    "balanced (5,000 per family) so A accuracy is the headline; on B/C the family mix is a design artefact. Detection: "
    "threshold set at 5% on even replicates of the C CSR anchors (per method and seed), size measured on the odd ones; "
    "classifiers score 1 − P(poisson).",
    "Grid pairs (0.25 vs 0.08 F/G/J radius grid): report the arm marked ✓ val-selected (lower mean validation loss / "
    "higher mean validation accuracy), never the one that wins on A/B/C. All neural arms are the vihrs 1-D CNN "
    "(Vihrs 2022) with best-validation checkpointing; only the input channels differ.",
]


def write_sheet(rows: dict[str, list[dict]], path: Path) -> None:
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise SystemExit("openpyxl is not installed in this env: pip install openpyxl "
                         "(or put a directory containing it on PYTHONPATH)") from exc

    thin = Side(style="thin", color="A6A6A6")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap = Alignment(wrap_text=True, vertical="top")
    center = Alignment(wrap_text=True, vertical="center", horizontal="center")
    hdr_fill, sub_fill, leg_fill = (PatternFill("solid", fgColor=c) for c in ("305496", "D9E1F2", "F2F2F2"))

    wb = load_workbook(path)
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    ws = wb.create_sheet(SHEET, 0)
    wb.active = 0
    width = 22

    def put(r, c, v, fill=None, bold=False, color=None, align=wrap, frame=True):
        cell = ws.cell(row=r, column=c, value=v)
        if fill:
            cell.fill = fill
        cell.font = Font(bold=bold, color=color)
        cell.alignment = align
        if frame:
            cell.border = border
        return cell

    r = 1
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=width)
    put(r, 1, "DV3 — classical baselines: status and per-regime results", bold=True, frame=False).font = Font(bold=True, size=14)
    r += 1
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=width)
    put(r, 1, f"Written {datetime.now():%Y-%m-%d %H:%M} by scripts/dv3_matrix.py from results/ on disk. "
              f"Re-run it after jobs land. Batch: bash {SUBMIT_SCRIPT} (parameter estimation first, "
              f"classification after it; {PREP_SCRIPT} builds the feature caches, {TRAIN_SCRIPT} trains).",
        frame=False)
    for note in NOTES:
        r += 1
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=width)
        put(r, 1, note, fill=leg_fill)
        ws.row_dimensions[r].height = 48
    r += 1
    put(r, 1, "Status", fill=sub_fill, bold=True, align=center)
    for j, s in enumerate(STATUS_FILL, 2):
        put(r, j, s, fill=PatternFill("solid", fgColor=STATUS_FILL[s]), color=STATUS_FONT[s], bold=True, align=center)
    counts = {s: sum(x["status"] == s for rs in rows.values() for x in rs) for s in STATUS_FILL}
    put(r, len(STATUS_FILL) + 2, "rows: " + ", ".join(f"{v} {k}" for k, v in counts.items()), frame=False)

    blocks = [
        ("Parameter estimation — normalised loss L (lower is better)", params_table(rows["params"])),
        ("Classification — 5-way accuracy (higher is better)", classify_table(rows["classify"])),
        ("Detection of departures from CSR — calibrated 5% tests on C (power: higher is better)",
         detect_table(rows["detect"])),
    ]
    for title, (head, body) in blocks:
        if len(head) < width:  # keep Results / Produced by in the same two columns in every block
            pad = width - len(head)
            head = head[:-2] + [""] * pad + head[-2:]
            body = [line[:-2] + [""] * pad + line[-2:] for line in body]
        r += 2
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=len(head))
        put(r, 1, title, fill=sub_fill, bold=True)
        r += 1
        for j, h in enumerate(head, 1):
            if h:
                put(r, j, h, fill=hdr_fill, bold=True, color="FFFFFF", align=center)
        ws.row_dimensions[r].height = 45
        status_col = head.index("Status") + 1
        for line in body:
            r += 1
            for j, v in enumerate(line, 1):
                if j == status_col:
                    put(r, j, v, fill=PatternFill("solid", fgColor=STATUS_FILL[v]), color=STATUS_FONT[v], bold=True,
                        align=center)
                elif v != "" or head[j - 1]:
                    put(r, j, v, align=wrap if isinstance(v, str) and len(v) > 24 else center)

    widths = [12, 12, 18, 10, 13, 8, 20] + [15] * 13 + [34, 46]
    for j, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = "C1"  # arm/family stay visible when scrolling right; the notes scroll away

    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    wb.save(tmp)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, default=MATRIX)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--dv3-root", type=Path, default=DEFAULT_DV3_ROOT)
    ap.add_argument("--dry-run", action="store_true", help="print the tables; do not touch the workbook")
    args = ap.parse_args(argv)

    rows = build_rows(Path(args.results_root), RegimeCache(Path(args.dv3_root)))
    for name, table in (("params", params_table), ("classify", classify_table), ("detect", detect_table)):
        head, body = table(rows[name])
        print(f"\n== {name}")
        keep = [i for i, h in enumerate(head) if h not in ("Results", "Produced by", "Inputs")]
        for line in body:
            print("  " + " | ".join(str(line[i]) for i in keep if line[i] != ""))
    if not args.dry_run:
        write_sheet(rows, args.matrix)
        print(f"\nwrote sheet {SHEET!r} -> {args.matrix}")


if __name__ == "__main__":
    main()
