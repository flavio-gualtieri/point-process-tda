# src/cloudforger/vectorization/landscapes/calibrated.py
"""Build calibrated LandscapeTransformer/SilhouetteTransformer bundles from
a diagram sample, per (homology dim, DTM scale) -- the landscape/silhouette
counterpart of persistence_images/calibrated.py's build_calibrated_imager
and scalar_features/calibrated.py's build_calibrated_betti.

Same fit-on-train/apply-frozen discipline as both of those: pass only the
TRAINING split's diagrams here (build_pi_tensor/build_betti_tensor's own
train_idx callers in experiments/pi_multik/ follow this convention; the new
build_landscape_tensor/build_silhouette_tensor in
experiments/pi_multik/vectorized_multik.py do too), then apply the frozen
transformer to val/test/grid/extrapolation diagrams. Two axes ONLY (grid
range, K) are fit from data; p is an experimental axis fixed per run (see
choose_K's docstring for why K is the one thing that gets a data-driven
rule instead of a hand-picked default), never fit.

No per-diagram amplitude normalization anywhere in this module: a tent's
height is |d_i - b_i|/2 in raw filtration units, and that's what ends up in
the landscape/silhouette output -- amplitude carries lifetime scale, and
rescaling it away per diagram would throw that signal out before the model
ever sees it."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.diagram import PersistenceDiagram
from ...calibration import axis_bounds_1d
from .landscape_silhouette import (
    LandscapeTransformer,
    MultiChannelLandscape,
    MultiChannelSilhouette,
    SilhouetteTransformer,
)
from .tent import landscape_from_tents, tent


def choose_K(
    diagrams: list[PersistenceDiagram],
    homology_dim: int,
    grid: np.ndarray,
    coverage: float = 0.99,
    cap: int = 16,
) -> tuple[int, np.ndarray]:
    """Smallest K such that the (K+1)-th landscape layer is negligible
    (sup-norm <= 1% of the first layer's sup-norm) for at least `coverage`
    of `diagrams`, capped at `cap`. Returns (K, sup_norms) where sup_norms
    is the (N, cap+1) matrix of ||lambda_k||_inf per diagram -- callers log
    this even when a config overrides K with a fixed budget value (e.g. the
    "matched" landscape arm's K=8), so the coverage-rule's own answer stays
    visible regardless of what K a given run actually trained with.

    Diagrams with fewer than K+1 points saturate their unused ranks at 0
    (see landscape_from_tents), which trivially satisfies the sup-norm
    condition -- exactly the intended behavior: a diagram that never had
    K+1 simultaneously-alive features doesn't need K+1 layers to represent
    it, and shouldn't count against coverage."""
    cap = int(cap)
    sup_norms = np.zeros((len(diagrams), cap + 1), dtype=float)
    for i, d in enumerate(diagrams):
        pairs = d.finite_pairs(homology_dim)
        tents = tent(pairs, grid)
        landscape = landscape_from_tents(tents, cap + 1)  # (cap+1, G)
        sup_norms[i] = landscape.max(axis=1)

    lambda1 = sup_norms[:, 0]
    threshold = 0.01 * lambda1

    for K in range(1, cap + 1):
        next_layer_sup = sup_norms[:, K]  # row index K == layer K+1 (0-indexed)
        frac_ok = float(np.mean(next_layer_sup <= threshold))
        if frac_ok >= coverage:
            return K, sup_norms

    return cap, sup_norms


def build_calibrated_landscape(
    diagrams: list[PersistenceDiagram],
    homology_dims: tuple[int, ...] = (0, 1),
    G: int = 512,
    K: int | dict[int, int] | None = None,
    q: float = 0.99,
    pad_factor: float = 1.05,
    K_coverage: float = 0.99,
    K_cap: int = 16,
    verbose: bool = True,
) -> MultiChannelLandscape:
    """One LandscapeTransformer per dim in homology_dims, each on its own
    per-dim-calibrated [t_min, T] grid (axis_bounds_1d) -- unlike Betti
    curves (build_calibrated_betti), there's no requirement to share one
    grid across dims here, so this mirrors persistence_images/calibrated.py's
    per-dim independence instead.

    K: None -> run choose_K per dim (data-driven, capped at K_cap), THEN use
    the max across dims for every dim's transformer -- a shared K is a hard
    requirement, not a simplification: build_landscape_tensor
    (experiments/pi_multik/vectorized_multik.py) stacks every dim's (K, G)
    landscape into one (C, K, G) raster along the channel axis, which is
    only a valid rectangular array if every channel has the same K. H0
    typically has many more simultaneously-alive features than H1 under
    DTM, so their independently-chosen K's routinely differ -- confirmed by
    running choose_K on realistic H0-heavy/H1-sparse diagrams, which
    crashes np.stack without this reconciliation. The shorter dim's unused
    top layers are legitimately zero anyway (landscape_from_tents zero-pads
    any layer beyond a diagram's own point count, and the coverage rule's
    whole premise is that layers past its own chosen K are already
    negligible), so padding it up to the shared max wastes a few channels
    rather than fabricating signal.

    An int -> that K for every dim; a {dim: K} dict -> per-dim override
    (e.g. the "matched" landscape arm's fixed K=8) -- both bypass the
    max-reconciliation above (the caller has already picked one value, or
    is explicitly asking for different per-dim values and accepting that
    only makes sense when the resulting tensor doesn't need to be
    channel-stacked, e.g. K is used identically for every dim in practice).
    choose_K's own per-dim answer is still computed and logged in every
    case (see that function's docstring), even when a fixed K overrides it."""
    # Two passes: first calibrate each dim's own [t_min, T] grid and run
    # choose_K against it (independent per dim, as intended); only THEN
    # decide the K every dim's transformer actually gets, once every dim's
    # own answer is known.
    per_dim: dict[int, tuple[float, float, int, np.ndarray]] = {}
    for dim in homology_dims:
        t_min, T = axis_bounds_1d(diagrams, dim, q=q, pad_factor=pad_factor)
        grid = np.linspace(t_min, T, G)
        chosen_K, sup_norms = choose_K(diagrams, dim, grid, coverage=K_coverage, cap=K_cap)
        per_dim[dim] = (t_min, T, chosen_K, sup_norms)

    shared_K = max(chosen_K for _, _, chosen_K, _ in per_dim.values()) if K is None else None

    transformers: dict[int, LandscapeTransformer] = {}
    for dim in homology_dims:
        t_min, T, chosen_K, sup_norms = per_dim[dim]
        if K is None:
            dim_K = shared_K
        elif isinstance(K, dict):
            dim_K = int(K[dim])
        else:
            dim_K = int(K)

        transformers[dim] = LandscapeTransformer(t_min=t_min, T=T, G=G, K=dim_K)
        if verbose:
            coverage_note = "" if dim_K == chosen_K else f" (this dim's own coverage-rule answer: K={chosen_K})"
            print(
                f"  [landscape] dim={dim}: t_min={t_min:.4f} T={T:.4f} G={G} "
                f"K={dim_K}{coverage_note} ||lambda_1||_inf(median)={np.median(sup_norms[:, 0]):.4g}"
            )

    return MultiChannelLandscape(transformers)


def build_calibrated_silhouette(
    diagrams: list[PersistenceDiagram],
    homology_dims: tuple[int, ...] = (0, 1),
    G: int = 4096,
    p: float = 1.0,
    q: float = 0.99,
    pad_factor: float = 1.05,
    verbose: bool = True,
) -> MultiChannelSilhouette:
    """One SilhouetteTransformer per dim in homology_dims, each on its own
    per-dim-calibrated [t_min, T] grid (axis_bounds_1d) -- same per-dim
    independence as build_calibrated_landscape. p is fixed for every dim in
    one call (it's an experimental axis, run as separate arms -- see this
    module's docstring -- not something that varies per dim within a
    single arm)."""
    transformers: dict[int, SilhouetteTransformer] = {}
    for dim in homology_dims:
        t_min, T = axis_bounds_1d(diagrams, dim, q=q, pad_factor=pad_factor)
        transformers[dim] = SilhouetteTransformer(t_min=t_min, T=T, G=G, p=p)
        if verbose:
            print(f"  [silhouette] dim={dim}: t_min={t_min:.4f} T={T:.4f} G={G} p={p:g}")

    return MultiChannelSilhouette(transformers)
