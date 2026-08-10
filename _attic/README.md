# `_attic/`

Archived code, kept for git-blame/history and reference, deliberately not
deleted. Nothing here is imported by the active pipeline
(`src/cloudforger/`, `scripts/`, `tests/`) — if you find yourself wanting
to import from `_attic/`, that's a sign the thing you need should be moved
back out, not that `_attic/` should grow a new consumer.

## `legacy/`

The entire pre-existing top-level `legacy/` directory (dtm_experiment/,
models/, scripts/{analysis,experiments,processing,runners}/, plus its
sibling configs/ and data/), moved here as one unit.

**Why it's here:** it already didn't work before this refactor touched
anything. Every file in it that has internal imports references at least
one of `cloudforger.tda.*` (filtration/vectorizer/calibration/features
submodules that don't exist anywhere in this repo's history we could find)
or `cloudforger.nn.train_old` (also doesn't exist). This is the "before"
snapshot of an earlier restructure that replaced it with the current
`src/cloudforger/` package + `scripts/` CLI — it was never deleted, just
left behind. See `docs/refactor_inventory.md` for the file-by-file
superseded-by mapping (short version: `dtm_experiment/*` → today's
`nn`-derived `experiments/` + `scripts/{train,evaluate}.py`;
`models/mincontrast.py`/`baselines.py` → `src/cloudforger/baselines/`;
`scripts/processing/params/pipeline_lib/*` → `core/{io,records}.py` +
`data_generation/design.py` + `scripts/{generate,featurize}.py`).
`scripts/processing/classify/*` (a separate classification-task track,
point clouds of dimension 2–16) has no successor at all — it looks like an
abandoned experiment direction, not a migrated one.

## `vectorizers_retired/adaptive_persistence_image.py`

Retired per-diagram adaptive-`sigma` persistence-image design. **Superseded
by** the paper-faithful, fixed-`sigma_pixels` `PersistenceImager` in
`src/cloudforger/vectorization/persistence_images/persistence_image.py`.
The file's own header comment has the full story (the adaptive sigma sat
at its floor for ~all H0 and ~99.7% of H1 diagrams — not meaningfully
adaptive — and isn't covered by the Adams et al. 2017 stability guarantee).
Kept because `notebooks/pi_sigma_diagnostics.ipynb` still loads it
deliberately, for comparison against the current fixed-sigma design.

## `core_shims/betti.py`

Deprecated 4-line compatibility shim (`"moved to cloudforger.features.result
... Shim for not-yet-migrated callers"`). **Superseded by**
`src/cloudforger/vectorization/scalar_features/result.py`'s
`BettiCurveFeature` (the real, current home — this shim just re-exported
it). Its only callers were two files inside `legacy/` above.
