"""Stage 1's routing curve, and the training pools it defines (`training: regime` at stages 2, 3).

The routing curve is, per true family f and group g,

    pi_fg(c) = P(stage 1 routes a cloud of family f with regime coordinate c to group g)

estimated from HELD-OUT stage-1 predictions (config regime.source), in quantile bins of the
coordinate and linearly interpolated between bin centres. It turns the per-cloud accident of
routing into a property of the regime, which every train cloud can be scored on.

Pools (config regime.threshold = tau):
  resolved    in-group clouds with pi_f,own(c) >= tau: the regime where stage 1 reliably sends
              clouds like this one to its own group. Trained on with their family label, at full
              prior weight -- ALL of them, not just the ones routed right on this draw.
  reject      everything else: in-group clouds below the boundary, poisson, the other group.
              Label `reject`, weight = prior x pi_fg(c) (reject_weight: routing_probability), i.e.
              in the proportion stage 1 actually sends them to group g. A cloud stage 1 never sends
              here gets ~0 weight; a near-CSR one it sends 30% of the time counts 0.3. This is the
              "some near-CSR clouds" for the reject output, with the amount set by stage 1 itself.
              reject_weight: uniform gives every reject candidate full prior weight instead.
The boundary itself (the coordinate where pi_f,own crosses tau) is reported, per family.

Coordinates (config regime.coordinate): any manifest column, one for all families or a mapping
family -> column (e.g. delta_tilde_hi for the clustered families, delta_tilde_lo for matern2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import CLASSES, REGIME, load_predictions

EPS = 0.01          # reject candidates below this routing probability are dropped (weight ~ 0)


def coordinate(cfg: dict, family: str) -> str:
    c = cfg["regime"]["coordinate"]
    return c if isinstance(c, str) else c.get(family, c.get("default", "delta_tilde"))


class RoutingCurve:
    def __init__(self, cfg: dict, stage1_predictions, rows: pd.DataFrame):
        rc = cfg["regime"]
        s1 = load_predictions(stage1_predictions)
        sources = {"val": ["val"], "train_oof": ["train"], "both": ["val", "train"]}[rc["source"]]
        s1 = s1[s1.split.isin(sources)]
        if s1.empty:
            raise SystemExit(f"stage 1 has no predictions on {sources}; regime.source must be held out")
        self.cfg, self.curves = cfg, {}
        for family, g in s1.join(rows[["family"]]).groupby("family"):
            c = rows.loc[g.index, coordinate(cfg, family)].to_numpy()
            edges = np.unique(np.quantile(c, np.linspace(0, 1, rc["bins"] + 1)))
            b = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, len(edges) - 2)
            centres = np.array([np.median(c[b == i]) for i in range(len(edges) - 1)])
            rates = {grp: np.array([(g.pred.to_numpy()[b == i] == grp).mean() for i in range(len(edges) - 1)])
                     for grp in CLASSES}
            self.curves[family] = (centres, rates)

    def pi(self, family: np.ndarray, coord: pd.DataFrame, group: str) -> np.ndarray:
        """Routing probability to `group` for each cloud, from its family and coordinate."""
        out = np.zeros(len(family))
        for f in np.unique(family):
            centres, rates = self.curves[f]
            m = family == f
            out[m] = np.interp(coord.loc[m, coordinate(self.cfg, f)].to_numpy(), centres, rates[group])
        return out

    def own(self, family: np.ndarray, coord: pd.DataFrame) -> np.ndarray:
        """pi to each cloud's OWN group."""
        out = np.zeros(len(family))
        for grp in CLASSES:
            m = np.array([REGIME[f] == grp for f in family])
            out[m] = self.pi(family[m], coord[m], grp)
        return out

    def boundaries(self) -> dict:
        """Per non-poisson family: the smallest coordinate at which pi_own reaches the threshold."""
        tau, out = self.cfg["regime"]["threshold"], {}
        for f, (centres, rates) in self.curves.items():
            if f == "poisson":
                continue
            above = np.flatnonzero(rates[REGIME[f]] >= tau)
            out[f] = {"coordinate": coordinate(self.cfg, f),
                      "boundary": float(centres[above[0]]) if len(above) else None,
                      "curve": dict(zip(np.round(centres, 3).tolist(), np.round(rates[REGIME[f]], 3).tolist()))}
        return out
