"""F3: confusion matrix of the frozen classifier (paper.yaml frozen_classifier) on the test split.

Rows are the true family, columns the call (argmax posterior), each row normalised to 1, so the
diagonal is recall and the poisson column is how often a family is called CSR. Every test pattern
counts once; families are balanced in the test split, so no prior weighting is needed.

    python paper/scripts/f3_confusion.py     # -> paper/figs/f3_confusion.pdf
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import common as C


def confusion(model: str) -> np.ndarray:
    post = C.classifier(model)
    post = post[post.split == "test"]
    truth = post.index.str.split("-").str[0].to_numpy()
    call = np.array(C.FAMILIES)[post[C.FAMILIES].to_numpy().argmax(1)]
    m = np.array([[np.mean(call[truth == t] == c) for c in C.FAMILIES] for t in C.FAMILIES])
    return m


def main() -> None:
    C.style()
    model = C.cfg()["frozen_classifier"]
    m = confusion(model)
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.imshow(m, cmap=C.blues(), vmin=0, vmax=1)
    for i in range(len(m)):
        for j in range(len(m)):
            if m[i, j] >= 0.005:
                ax.text(j, i, f"{m[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.5,
                        color="white" if m[i, j] > 0.55 else C.INK)
    labels = [C.NAMES[f] for f in C.FAMILIES]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("called")
    ax.set_ylabel("true family")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    C.save(fig, "f3_confusion")


if __name__ == "__main__":
    main()
