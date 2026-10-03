"""F3: confusion matrix of the frozen classifier (paper.yaml frozen_classifier) on the test split.

Rows are the true family, columns the call (argmax posterior), each row normalised to 1, so the
diagonal is recall and the poisson column is how often a family is called CSR. Families in Table 1's order,
the mechanisms (paper.yaml coarse) outlined; cells under 0.005 are left blank. Every test pattern
counts once; families are balanced in the test split, so no prior weighting is needed. With the paper
config's `confusion_tau`, only the clouds in the structured regime at that tau (compare's frozen cutoffs) count.

    python paper/scripts/f3_confusion.py     # -> paper/figs/f3_confusion.pdf
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import common as C

ORDER = ["poisson", "thomas", "nested", "ring", "lgcp", "matern2", "strauss", "cell"]


def confusion(model: str) -> np.ndarray:
    post = C.classifier(model)
    post = post[post.split == "test"]
    tau = C.cfg().get("confusion_tau")
    if tau is not None:
        from cloudforger.pipeline.regime import in_regime
        post = post[in_regime(C.cutoffs(), C.rows().loc[post.index], tau)]
    truth = post.index.str.split("-").str[0].to_numpy()
    call = np.array(C.FAMILIES)[post[C.FAMILIES].to_numpy().argmax(1)]
    fams = [f for f in ORDER if f in C.FAMILIES]
    m = np.array([[np.mean(call[truth == t] == c) for c in fams] for t in fams])
    return m


def main() -> None:
    C.style()
    model = C.cfg()["frozen_classifier"]
    m = confusion(model)
    fams = [f for f in ORDER if f in C.FAMILIES]
    fig, ax = plt.subplots(figsize=(C.COLUMN_WIDTH, 3.2))
    ax.imshow(np.ma.masked_less(m, 0.005), cmap=C.blues(), vmin=0, vmax=1)
    coarse = [C.cfg()["coarse"][f] for f in fams]
    start = 0
    for i in range(1, len(fams) + 1):                           # one outline per mechanism's block
        if i == len(fams) or coarse[i] != coarse[start]:
            ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), i - start, i - start, fill=False,
                                       edgecolor=C.INK2, lw=0.8))
            start = i
    for i in range(len(m)):
        for j in range(len(m)):
            if m[i, j] >= 0.005:
                ax.text(j, i, f"{m[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=6.5,
                        color="white" if m[i, j] > 0.55 else C.INK)
    labels = [C.PLOT_NAMES[f] for f in fams]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Called")
    ax.set_ylabel("True family")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    C.save(fig, "f3_confusion")


if __name__ == "__main__":
    main()
