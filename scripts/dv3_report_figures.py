#!/usr/bin/env python3
"""Generate figures for docs/status/dv3_classical_baselines.tex from results/ on
disk, reusing scripts/dv3_matrix.py's row builder so the figures and the xlsx
sheet cannot disagree. Read-only aggregation + plotting; no training."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dv3_matrix import build_rows, PARAM_FAMILIES, DELTA_LABELS  # noqa: E402
from evaluate_regimes import RegimeCache  # noqa: E402
from cloudforger.evaluation.dv3 import DEFAULT_DV3_ROOT  # noqa: E402

OUT = ROOT / "docs" / "status" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

ARM_ORDER = ["L", "LFGJ", "LFGJ_r080", "G", "F", "G_r080", "F_r080"]
ARM_COLOR = {
    "L": "#1b1b1b", "LFGJ": "#2166ac", "LFGJ_r080": "#67a9cf",
    "G": "#b2182b", "G_r080": "#ef8a62",
    "F": "#1a9850", "F_r080": "#91cf60",
}

rows = build_rows(ROOT / "results", RegimeCache(DEFAULT_DV3_ROOT))

# ---------------------------------------------------------------------------
# Figure 1: parameter-estimation A loss, grouped bars (family x arm)
# ---------------------------------------------------------------------------
params = {(r["family"], r["arm"].key): r for r in rows["params"]}
fig, ax = plt.subplots(figsize=(9, 3.6))
n_arms = len(ARM_ORDER)
width = 0.8 / n_arms
x = np.arange(len(PARAM_FAMILIES))
for i, arm in enumerate(ARM_ORDER):
    means, sds = [], []
    for fam in PARAM_FAMILIES:
        agg = params[(fam, arm)]["agg"].get("A", {})
        means.append(agg.get("mean", np.nan))
        sds.append(agg.get("sd", 0.0))
    ax.bar(x + (i - n_arms / 2 + 0.5) * width, means, width, yerr=sds, capsize=1.5,
           label=arm, color=ARM_COLOR[arm], linewidth=0)
ax.set_xticks(x)
ax.set_xticklabels(PARAM_FAMILIES)
ax.set_ylabel("normalised loss $L$ (set A)")
ax.set_title("Parameter estimation: set-A loss by family and arm (lower is better)")
ax.legend(ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "params_by_family.pdf")
fig.savefig(OUT / "params_by_family.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: two panels - classification accuracy vs delta band, detection
# power vs delta band, one line per arm
# ---------------------------------------------------------------------------
classify = {r["arm"].key: r for r in rows["classify"]}
detect = {r["arm"].key: r for r in rows["detect"] if r["arm"].key in ARM_ORDER}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
xb = np.arange(len(DELTA_LABELS))
for arm in ARM_ORDER:
    agg = classify[arm]["agg"]
    means = [agg.get(f"A_d{lab}", {}).get("mean", np.nan) for lab in DELTA_LABELS]
    ax1.plot(xb, means, marker="o", ms=3, label=arm, color=ARM_COLOR[arm])
ax1.set_xticks(xb)
ax1.set_xticklabels(DELTA_LABELS, rotation=40, ha="right", fontsize=7)
ax1.set_ylabel("5-way accuracy (A, structured)")
ax1.set_xlabel(r"$\tilde\delta$ band")
ax1.set_title("Classification accuracy vs. departure from CSR")
ax1.spines[["top", "right"]].set_visible(False)

for arm in ARM_ORDER:
    agg = detect[arm]["agg"]
    means = [agg.get(f"C_d{lab}", {}).get("mean", np.nan) for lab in DELTA_LABELS]
    ax2.plot(xb, means, marker="o", ms=3, label=arm, color=ARM_COLOR[arm])
ax2.axhline(0.05, color="gray", lw=0.8, ls="--")
ax2.set_xticks(xb)
ax2.set_xticklabels(DELTA_LABELS, rotation=40, ha="right", fontsize=7)
ax2.set_ylabel("power at 5% size (C)")
ax2.set_xlabel(r"$\tilde\delta$ band")
ax2.set_title("CSR-detection power vs. departure from CSR")
ax2.spines[["top", "right"]].set_visible(False)
handles, labels = ax1.get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.85))
fig.legend(handles, labels, ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.savefig(OUT / "regime_results.pdf")
fig.savefig(OUT / "regime_results.png", dpi=150)
plt.close(fig)

print("wrote", OUT / "params_by_family.pdf")
print("wrote", OUT / "regime_results.pdf")
