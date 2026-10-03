"""F2: the pipeline schematic, from a single observed cloud to the end-to-end kernel score.

Two halves, connected by a zoom-in connector:
    (top)    the abstract flow -- cloud -> ParamNet -> classifier/estimator -> fitted model
             -> K=16 simulations -> kernel score against a held-out replicate -> S(P_hat, y)
    (bottom) ParamNet itself -- the two feature branches (classical curves, persistence
             images of two filtrations), their encoders, the concat with log n, and the shared
             FCNN head forking into the classifier and estimator outputs

A pure schematic: no run data is read, so this is a drawing, not a display of results.

    python paper/scripts/f2_pipeline.py      # -> paper/figs/f2_pipeline.pdf
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

import common as C

PANEL = "#f7f7f5"
FS = 1.1                                                        # every text size, scaled for print: the smallest is ~5.4pt


def tint(color: str, a: float):
    return mcolors.to_rgba(color, a)


def rbox(ax, cx, cy, w, h, label=None, fc="white", ec=C.INK2, lw=1.1, fs=6.6 * FS, tc=C.INK,
         pad=0.3, rounding=0.4, shadow=True, zorder=3, ls="-"):
    if shadow:
        ax.add_patch(FancyBboxPatch((cx - w / 2 + 0.45, cy - h / 2 - 0.4), w, h,
                     boxstyle=f"round,pad={pad},rounding_size={rounding}",
                     fc="#000000", ec="none", alpha=0.05, zorder=zorder - 0.1))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle=f"round,pad={pad},rounding_size={rounding}",
                 fc=fc, ec=ec, lw=lw, ls=ls, zorder=zorder))
    if label:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, color=tc, zorder=zorder + 1, linespacing=1.5)


def arrow(ax, p0, p1, color=C.INK2, lw=1.1, rad=0.0, zorder=2, mutation=7, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", ls=ls,
                 mutation_scale=mutation, lw=lw, color=color, zorder=zorder, shrinkA=1, shrinkB=1))


def icon_cloud(ax, cx, cy, r=3.2, color=C.INK, n=13, seed=0, s=2.4):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = r * np.sqrt(rng.uniform(0.05, 0.95, n))
    ax.scatter(cx + rad * np.cos(ang), cy + rad * np.sin(ang) * 0.82, s=s, c=color, zorder=4, linewidths=0)


def icon_curves(ax, cx, cy, w=6.0, h=4.6):
    xs = np.linspace(cx - w / 2, cx + w / 2, 40)
    for c, k, dy in [(C.SERIES[1], 2.6, 0.5), ("#d14e13", 1.5, -0.3), ("#f2a67e", 4.0, 1.1)]:
        y = cy - h / 2 + (h * 0.78) * (1 - np.exp(-k * (xs - xs[0]) / w)) + dy
        ax.plot(xs, y, color=c, lw=1.0, zorder=4, solid_capstyle="round")


def icon_pd(ax, cx, cy, s=4.6, color=C.SERIES[0], seed=2, n=8):
    x0, y0 = cx - s / 2, cy - s / 2
    ax.plot([x0, x0 + s], [y0, y0], color=C.MUTED, lw=0.7, zorder=4)
    ax.plot([x0, x0], [y0, y0 + s], color=C.MUTED, lw=0.7, zorder=4)
    ax.plot([x0, x0 + s], [y0, y0 + s], color=C.MUTED, lw=0.6, ls=(0, (1.4, 1.3)), zorder=4)
    rng = np.random.default_rng(seed)
    bx = rng.uniform(x0 + 0.12 * s, x0 + 0.72 * s, n)
    by = np.clip(bx + rng.uniform(0.1, 0.55, n) * s, y0, y0 + s * 0.96)
    ax.scatter(bx, by, s=3.0, c=color, zorder=5, linewidths=0)


def icon_layers(ax, cx, cy, color, w=5.6, h=3.0, n=3, dx=0.9, dy=0.65):
    for i in reversed(range(n)):
        ax.add_patch(FancyBboxPatch((cx - w / 2 + i * dx, cy - h / 2 - i * dy), w, h,
                     boxstyle="round,pad=0.05,rounding_size=0.15",
                     fc=tint(color, 0.16 + 0.22 * (n - 1 - i)), ec=color, lw=0.8, zorder=4 + i))


def icon_bars(ax, cx, cy, color, n=8, w=5.6, h=4.0, seed=3, hi=5):
    rng = np.random.default_rng(seed)
    heights = rng.uniform(0.18, 0.5, n)
    heights[hi] = 0.95
    bw = w / n * 0.6
    xs = np.linspace(cx - w / 2 + bw, cx + w / 2 - bw, n)
    for i, (x, hh) in enumerate(zip(xs, heights)):
        ax.add_patch(Rectangle((x - bw / 2, cy - h / 2), bw, hh * h,
                     fc=(color if i == hi else C.MUTED), ec="none", alpha=(0.95 if i == hi else 0.4), zorder=4))
    ax.plot([cx - w / 2, cx + w / 2], [cy - h / 2, cy - h / 2], color=C.AXIS, lw=0.6, zorder=3)


def icon_scalar(ax, cx, cy, color, w=5.6):
    ax.plot([cx - w / 2, cx + w / 2], [cy, cy], color=C.AXIS, lw=0.8, zorder=4)
    for xx in np.linspace(cx - w / 2, cx + w / 2, 5):
        ax.plot([xx, xx], [cy - 0.3, cy + 0.3], color=C.AXIS, lw=0.6, zorder=4)
    ax.scatter([cx + 0.16 * w], [cy], s=15, c=color, zorder=5, linewidths=0.6, edgecolors="white")


def diamond(ax, cx, cy, w, h, label, fc="white", ec=C.INK2, lw=1.2, fs=6.4 * FS, tc=C.INK, zorder=3):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                 closed=True, fc=fc, ec=ec, lw=lw, zorder=zorder))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, color=tc, zorder=zorder + 1, linespacing=1.4)


def chip(ax, cx, cy, w, h):
    """The abstract ParamNet glyph for the main row: two lanes merging into a head."""
    rbox(ax, cx, cy, w, h, zorder=3)
    lane_w, lane_h = w * 0.3, h * 0.26
    ax.add_patch(FancyBboxPatch((cx - w / 2 + w * 0.12, cy + h * 0.06), lane_w, lane_h,
                 boxstyle="round,pad=0.03,rounding_size=0.1", fc=tint(C.SERIES[1], 0.5), ec="none", zorder=4))
    ax.add_patch(FancyBboxPatch((cx - w / 2 + w * 0.12, cy - h * 0.32), lane_w, lane_h,
                 boxstyle="round,pad=0.03,rounding_size=0.1", fc=tint(C.SERIES[0], 0.5), ec="none", zorder=4))
    hx0 = cx - w / 2 + w * 0.5
    ax.add_patch(Polygon([(hx0, cy - h * 0.32), (hx0 + w * 0.3, cy - h * 0.18), (hx0 + w * 0.3, cy + h * 0.2),
                 (hx0, cy + h * 0.32)], closed=True, fc=tint(C.SERIES[2], 0.55), ec="none", zorder=4))
    ax.text(cx, cy - h / 2 - 1.0, "ParamNet", ha="center", va="top", fontsize=6.4 * FS, color=C.INK,
            weight="bold", zorder=5)


def main() -> None:
    C.style()
    fig = plt.figure(figsize=(C.TEXT_WIDTH, C.TEXT_WIDTH * 57 / 100))   # the drawing is 100 x 57 units
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 57)
    ax.set_aspect("equal")
    ax.axis("off")

    # =================================================================== main row: abstract flow
    MY = 42.5
    X = {"cloud": 8, "chip": 22, "fork": 36, "fit": 50, "sim": 64, "score": 90}

    rbox(ax, X["cloud"], MY, 10.5, 12, zorder=3)
    icon_cloud(ax, X["cloud"], MY, r=3.0)
    ax.text(X["cloud"], MY - 7.0, r"$x_{f,\theta}$", ha="center", va="top", fontsize=6.4 * FS, color=C.INK)

    chip(ax, X["chip"], MY, 13.5, 12.5)
    arrow(ax, (X["cloud"] + 5.6, MY), (X["chip"] - 7.1, MY))

    rbox(ax, X["fork"], MY + 5.2, 13.5, 6.6, r"classifier $\to\hat f$", ec=C.SERIES[2], fs=5.6 * FS, zorder=3)
    rbox(ax, X["fork"], MY - 5.2, 13.5, 6.6, r"estimator $\to\log\hat\theta$", ec=C.SERIES[2], fs=5.6 * FS, zorder=3)
    arrow(ax, (X["chip"] + 7.1, MY + 2), (X["fork"] - 6.4, MY + 5.2), rad=-0.25)
    arrow(ax, (X["chip"] + 7.1, MY - 2), (X["fork"] - 6.4, MY - 5.2), rad=0.25)

    rbox(ax, X["fit"], MY, 14.5, 10, "fitted model\n" r"$\hat P=(\hat f,\hat\theta)$", fs=6.0 * FS, zorder=3)
    arrow(ax, (X["fork"] + 6.9, MY + 4.4), (X["fit"] - 7.0, MY + 1.8), rad=-0.2)
    arrow(ax, (X["fork"] + 6.9, MY - 4.4), (X["fit"] - 7.0, MY - 1.8), rad=0.2)

    for i, (dx, dy) in enumerate([(-2.0, 1.6), (0.3, -1.6), (2.3, 0.6)]):
        icon_cloud(ax, X["sim"] + dx, MY + dy, r=1.9, color=C.MUTED, n=8, seed=10 + i, s=1.4)
    rbox(ax, X["sim"], MY, 13, 10.5, fc=PANEL, ec=C.AXIS, ls=(0, (3, 2)), shadow=False, zorder=2)
    ax.text(X["sim"], MY - 7.5, r"$K{=}16$ simulations", ha="center", fontsize=5.1 * FS, color=C.MUTED)
    arrow(ax, (X["fit"] + 7.4, MY), (X["sim"] - 6.8, MY))

    hoy_x = X["sim"] + 15.3
    icon_cloud(ax, hoy_x, MY + 8.6, r=1.7, color=C.INK2, n=10, seed=20, s=1.7)
    ax.text(hoy_x, MY + 11.3, r"held-out $y_{f,\theta}$", ha="center", fontsize=5.1 * FS, color=C.INK2)
    arrow(ax, (hoy_x, MY + 6.7), (hoy_x, MY + 5.3), color=C.INK2)
    diamond(ax, hoy_x, MY, 13.5, 10.5, "kernel\nscore")
    arrow(ax, (X["sim"] + 6.8, MY), (hoy_x - 6.4, MY))

    rbox(ax, X["score"], MY, 13, 9, r"$S(\hat P,\, y)$", fc=tint(C.SERIES[2], 0.14), ec=C.SERIES[2], fs=7.2 * FS, zorder=3)
    arrow(ax, (hoy_x + 6.9, MY), (X["score"] - 6.2, MY))

    # ========================================================= detail inset: concrete architecture
    bx0, bx1, by0, by1 = 6, 94, 1.5, 32.0
    rbox(ax, (bx0 + bx1) / 2, (by0 + by1) / 2, bx1 - bx0, by1 - by0, fc=PANEL, ec=C.AXIS, ls=(0, (3, 2)),
         shadow=False, zorder=1, rounding=0.6)
    ax.text((bx0 + bx1) / 2, by1 - 2.2, "ParamNet — shared architecture", ha="center",
            fontsize=6.6 * FS, color=C.INK, weight="bold", zorder=2)
    ax.text((bx0 + bx1) / 2, by1 - 4.1, "trained once as the classifier, once per family as an estimator",
            ha="center", fontsize=5.1 * FS, color=C.MUTED, style="italic", zorder=2)

    # zoom connector from the chip down to the inset
    arrow(ax, (X["chip"] - 5, MY - 6.3), (X["chip"] - 9, by1 - 1), color=C.AXIS, lw=0.8, ls=(0, (2, 1.4)), mutation=1)
    arrow(ax, (X["chip"] + 5, MY - 6.3), (X["chip"] + 9, by1 - 1), color=C.AXIS, lw=0.8, ls=(0, (2, 1.4)), mutation=1)

    IY = (by0 + by1) / 2 - 2
    top, bot = IY + 7.0, IY - 7.0
    IX = {"in": 15, "feat": 29, "enc": 43, "embed": 55, "cat": 62, "head": 74, "out": 87}

    rbox(ax, IX["in"], IY, 9, 15, zorder=3)
    icon_cloud(ax, IX["in"], IY, r=2.6, s=2.0)

    rbox(ax, IX["feat"], top, 10.5, 8, fc=tint(C.SERIES[1], 0.08), ec=C.SERIES[1], zorder=3)
    icon_curves(ax, IX["feat"], top, w=5.4, h=4.0)
    rbox(ax, IX["feat"], bot, 10.5, 8, fc=tint(C.SERIES[0], 0.12), ec=C.SERIES[0], zorder=3)
    icon_pd(ax, IX["feat"], bot, s=4.2)
    arrow(ax, (IX["in"] + 4.9, IY + 1.6), (IX["feat"] - 5.6, top), rad=-0.3)
    arrow(ax, (IX["in"] + 4.9, IY - 1.6), (IX["feat"] - 5.6, bot), rad=0.3)

    rbox(ax, IX["enc"], top, 10, 8, ec=C.SERIES[1], zorder=3)
    icon_layers(ax, IX["enc"], top + 1.1, color=C.SERIES[1])
    ax.text(IX["enc"], top - 3.3, "1D CNN", ha="center", fontsize=4.9 * FS, color=C.INK2)
    rbox(ax, IX["enc"], bot, 10, 8, ec=C.SERIES[0], zorder=3)
    icon_layers(ax, IX["enc"], bot + 1.1, color=C.SERIES[0])
    ax.text(IX["enc"], bot - 3.3, "4× CNN", ha="center", fontsize=4.9 * FS, color=C.INK2)
    arrow(ax, (IX["feat"] + 5.6, top), (IX["enc"] - 5.4, top))
    arrow(ax, (IX["feat"] + 5.6, bot), (IX["enc"] - 5.4, bot))

    ax.add_patch(Circle((IX["embed"], top), 1.5, fc=C.SERIES[1], ec="none", zorder=4))
    ax.add_patch(Circle((IX["embed"], bot), 1.5, fc=C.SERIES[0], ec="none", zorder=4))
    arrow(ax, (IX["enc"] + 5.2, top), (IX["embed"] - 1.6, top))
    arrow(ax, (IX["enc"] + 5.2, bot), (IX["embed"] - 1.6, bot))

    rbox(ax, IX["cat"], IY, 5.6, 5.6, "⊕", pad=0.2, rounding=2.9, fs=9 * FS, zorder=5)
    arrow(ax, (IX["embed"] + 1.6, top), (IX["cat"] - 2.4, IY + 2), rad=-0.2)
    arrow(ax, (IX["embed"] + 1.6, bot), (IX["cat"] - 2.4, IY - 2), rad=0.2)
    rbox(ax, IX["cat"], IY - 8.6, 6, 3.3, r"$\log n$", ec=C.MUTED, fs=5.2 * FS, shadow=False, zorder=4)
    arrow(ax, (IX["cat"], IY - 6.9), (IX["cat"], IY - 2.9))

    hx = IX["head"]
    ax.add_patch(Polygon([(hx - 5.4, IY - 6.4), (hx + 5.4, IY - 4.4), (hx + 5.4, IY + 4.4), (hx - 5.4, IY + 6.4)],
                 closed=True, fc=tint(C.SERIES[2], 0.14), ec=C.SERIES[2], lw=1.1, zorder=3))
    ax.text(hx, IY + 1.0, "shared", ha="center", fontsize=5.4 * FS, color=C.INK)
    ax.text(hx, IY - 1.4, "FCNN head", ha="center", fontsize=5.4 * FS, color=C.INK)
    arrow(ax, (IX["cat"] + 2.9, IY), (hx - 5.6, IY))

    rbox(ax, IX["out"], top, 11, 7.2, ec=C.SERIES[2], zorder=3)
    icon_bars(ax, IX["out"], top + 0.5, color=C.SERIES[2], w=5.0, h=3.2)
    ax.text(IX["out"], top - 2.9, r"$\hat f$", ha="center", fontsize=5.6 * FS, color=C.INK2)
    rbox(ax, IX["out"], bot, 11, 7.2, ec=C.SERIES[2], zorder=3)
    icon_scalar(ax, IX["out"], bot + 0.6, color=C.SERIES[2], w=5.0)
    ax.text(IX["out"], bot - 2.9, r"$\log\hat\theta$", ha="center", fontsize=5.6 * FS, color=C.INK2)
    arrow(ax, (hx + 5.6, IY + 1.6), (IX["out"] - 5.3, top), rad=-0.2)
    arrow(ax, (hx + 5.6, IY - 1.6), (IX["out"] - 5.3, bot), rad=0.2)

    C.save(fig, "f2_pipeline")


if __name__ == "__main__":
    main()
