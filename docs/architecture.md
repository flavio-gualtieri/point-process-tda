# Target architecture

Phase 2 design doc for `refactor/pipeline-housekeeping`. Read
[refactor_inventory.md](refactor_inventory.md) first — this doc assumes its
findings and states the target layout, the rationale, and every place this
diverges from the brief's starting sketch.

## Design principles

1. **`src/cloudforger/` stays the installable package root.** `pyproject.toml`
   already declares `[tool.setuptools.packages.find] where = ["src"]`, and
   ~20 scripts/tests/notebooks already do
   `sys.path.insert(0, str(ROOT / "src"))` then `import cloudforger...`.
   The six pipeline-stage names the brief sketches
   (`data_generation/`, `vectorization/`, `calibration/`, `encoders/`,
   `models/`, `experiments/`) become **subpackages of `cloudforger`**, not
   new top-level repo directories. This keeps every existing import root
   (`cloudforger.*`) valid and every existing CLI invocation
   (`python scripts/train.py configs/...`) unchanged — only the dotted
   path *after* `cloudforger.` changes. `scripts/`, `tests/`, `docs/`,
   `_attic/` remain top-level repo directories, exactly as sketched.
2. **`core/` is shared vocabulary, not a seventh stage.** A file stays in
   `core/` if two or more pipeline stages depend on it (`PointCloud`,
   `Region`, `PersistenceDiagram`, `SignedMeasure`, `Registry[T]`, pickle/
   YAML IO, train/val/test splitting, metric formulas, record
   (de)serialization). A file moves into a stage package if only that one
   stage depends on it. This is a mechanical test, applied file-by-file in
   the inventory table — it's why `core/` loses only two files
   (`calibration.py` out to its own package, `betti.py` archived) instead
   of being dissolved entirely.
3. **Preserve behavior exactly.** Every move in this pass is
   import-path-only. The one content edit that isn't a pure path fix is a
   single stale-docstring correction in `pi_multik.py` (a comment pointing
   at a file that doesn't exist) — called out explicitly in the inventory,
   not silently folded in.
4. **Registries are extended, not invented.** `core/registry.py`'s
   `Registry[T]` already backs point processes, filtrations,
   bifiltrations, and scalar features. This pass relocates those
   registries with their packages and adds exactly one new one
   (`encoders/`, which currently has no registry at all — every experiment
   hardcodes `from cloudforger.nn.encoders.X import Y`). It does not touch
   the registry class itself or introduce a second, different pattern.

## Layout

```
src/cloudforger/
  __init__.py                    # unchanged (package re-exports)
  config.py                      # unchanged (RunConfig + Process/Filtration/Bifiltration/Feature/MethodConfig)
  paths.py                       # unchanged (DataPaths/ResultsPaths)
  provenance.py                  # unchanged (provenance stamp + ledger)

  core/                          # shared vocabulary (principle 2) -- unchanged except the two rows below
    base.py, cloud.py, region.py, diagram.py, signed_measure.py,
    registry.py, io.py, splits.py, metrics.py, records.py,
    features.py, utils.py
    # calibration.py  -> moved to calibration/diagram_calibration.py
    # betti.py        -> archived to _attic/ (dead shim, see inventory)

  data_generation/                       # point clouds -> persistence diagrams / signed measures
    design.py                             # <- core/design.py (design-space sampling for scripts/generate.py)
    point_processes/                       # <- processes/  (Registry[PointProcess], unchanged)
      __init__.py, poisson.py, matern.py, thomas.py, nested_thomas.py,
      neyman_scott.py, inhom_thomas.py
    filtration/                            # <- filtration/  (Registry[Filtration] + BIFILTRATION_REGISTRY, unchanged)
      __init__.py, base.py, rips.py, dtm.py, bifiltration.py

  vectorization/                          # diagrams -> persistence images + scalar features
    persistence_images/                    # <- vectorizers/  (adaptive_persistence_image.py archived, see below)
      __init__.py, persistence_image.py, multi_channel.py, calibrated.py,
      signed_measure_image.py
    scalar_features/                       # <- features/  (Registry[DiagramFeature], unchanged)
      __init__.py, base.py, betti_curve.py, persistence_entropy.py,
      calibrated.py, result.py

  calibration/                            # NEW package: axis-range + resolution + sigma_pixels calibration
    __init__.py                            # thin re-export
    diagram_calibration.py                 # <- core/calibration.py, unchanged content

  encoders/                               # <- nn/encoders/  (+ NEW ENCODER_REGISTRY, see "New scaffolding" below)
    __init__.py, base.py, coordconv_pi.py, scaleconv_pi.py, towerconv_pi.py,
    persistence_image.py, point_cloud.py, sequence_cnn.py, stats.py

  models/                                 # <- nn/models/ + nn/heads/
    __init__.py, single_modal.py, multi_modal.py
    heads/
      __init__.py, paramest.py, classifier.py

  training/                               # <- nn/data.py, nn/splits.py, nn/train.py (training mechanics, principle 2's logic applied one level down)
    __init__.py, data.py, splits.py, train.py

  experiments/                            # <- nn/experiments/  (REGISTRY unchanged, pi_multik family consolidated)
    __init__.py, base.py, common.py
    raw_pc.py, pairwise.py, persistence_image.py, mph_pi.py, mph_fusion.py,
    betti.py, betti_cnn.py, ph_combined.py, fusion.py
    pi_multik/                             # the reference k=5,10,15 pipeline
      __init__.py                           # NEW: re-exports + registers all 5 variants
      pi_multik.py                          # <- nn/experiments/pi_multik.py (PIMultiK, PIMultiKExperiment)
      pi_multik_scaleconv.py                # <- nn/experiments/pi_multik_scaleconv.py
      pi_multik_towers.py                   # <- nn/experiments/pi_multik_towers.py
      pi_multik_earlyfusion.py              # <- nn/experiments/pi_multik_earlyfusion.py
      pi_multik_fusion.py                   # <- nn/experiments/pi_multik_fusion.py

  baselines/                              # unchanged -- non-TDA comparison estimators, outside the 6 stages by design
    __init__.py, mincontrast.py, palm.py, vihrs.py

  stats/                                  # unchanged location; flagged orphan, see inventory Judgment call #1
    __init__.py, base.py, pair_dist.py

scripts/            # unchanged paths; thin CLI entry points (already matched this role)
tests/              # unchanged paths; test_registries.py imports updated, test_pipeline_e2e.py gets a pi_multik case
configs/            # unchanged -- see "Deviation" below
docs/               # unchanged existing content + this file + refactor_inventory.md
notebooks/          # unchanged paths; 4 notebooks get import-line fixes (see inventory)
_attic/             # NEW
  README.md
  legacy/                            # <- top-level legacy/ (56 files), moved as one unit
  vectorizers_retired/
    adaptive_persistence_image.py    # <- vectorizers/adaptive_persistence_image.py
  core_shims/
    betti.py                         # <- core/betti.py
```

## Requirement-by-requirement rationale

**(a) Each k in {5, 10, 15} can get its own encoder instance/config.**
Today `PIMultiK.__init__` builds exactly one `CoordConvPIEncoder` and
applies it to every k by folding k into the batch dimension
(`nn/experiments/pi_multik.py`'s `PIMultiK.forward`). That sharing is a
deliberate design choice (see the module docstring: "SHARED-WEIGHT
CoordConv branch"), not an oversight, so this pass does not change it —
doing so would change the model's parameter count and behavior, which is
exactly what "scaffolding, not the experiment itself" rules out. What this
pass *does* do: add `ENCODER_REGISTRY` (new, in `encoders/__init__.py`) so
any encoder is nameable/buildable generically (`ENCODER_REGISTRY.build("coordconv_pi", **kwargs)`),
and leave a comment at the exact line in `PIMultiK.__init__` where a future
per-k encoder list would plug in. A future change can then thread
`cfg["encoders"]: list[...]` (one entry per k, defaulting to the current
single shared config when only one is given) through that one
constructor without touching data loading, training, or saving.

**(b) Calibration params + PI resolution are sweepable; one obvious
single-combo entry point.** Already true today and left as-is:
`FeatureConfig.params` in `config.py` already carries `resolution`,
`sigma_pixels`, `pd_calibration_coverage` as plain dict entries (see
`configs/runs/nested_thomas/nested_thomas_pi_multik_k5k10k15.yaml`), and
`scripts/featurize_sigma_sweep.py` already separates `build_combo_images`
(runs exactly one `(sigma_pixels, coverage)` combination) from `main`
(the grid loop, `itertools.product` over `--sigma-pixels`/`--coverage`).
The only change here is relocating `core/calibration.py` →
`calibration/diagram_calibration.py` so calibration is its own
first-class package rather than living inside the `core/` grab-bag,
matching the brief's explicit ask for a dedicated `calibration/` home.

**(c) CNN architecture is config-driven from one place.** Already true for
the encoders pi_multik uses: `PIMultiK.__init__`'s `conv_channels`,
`dropout`, `pool_type`, `head_hidden_dims`, `head_dropout`,
`scale_fusion_hidden`, `scale_fusion_out_dim`, `scale_fusion_kernel_size`
are all threaded from `cfg["method"]["params"]` in
`PIMultiKExperiment.run()` — no hardcoded architecture constants outside
config. This pass doesn't add anything here beyond moving the files into
`encoders/`/`models/` so the architecture-definition layer is physically
separated from the experiment-orchestration layer (`experiments/`) that
reads the config and instantiates it.

**(d) Scalar features live behind a registry.** Already true:
`features/__init__.py`'s `REGISTRY: Registry[DiagramFeature]` (moving to
`vectorization/scalar_features/`). No new code needed — moved as-is.

**(e) Point processes live behind a registry.** Already true:
`processes/__init__.py`'s `REGISTRY: Registry[PointProcess]` (moving to
`data_generation/point_processes/`). No new code needed — moved as-is.

**(f) Generation and vectorization are separate, independently-runnable
stages with a disk artifact contract.** Already true, and already
enforced by `paths.py`'s `DataPaths`: `scripts/generate.py` writes
`data/<process>/clouds.pkl` (+ `adversarial_clouds.pkl`); `scripts/featurize.py`
reads that and writes `data/<process>/<filtration_tag>/{diagrams,betti_curve,
persistence_image,persistence_entropy}.pkl`, independently re-runnable
without regenerating clouds. This pass makes the split more legible by
also separating the *code*: `data_generation/` (point clouds + diagrams)
vs. `vectorization/` (diagrams + scalar features), matching the on-disk
contract exactly — `filtration/` (which produces diagrams) sits in
`data_generation/`, not `vectorization/`, because `paths.py` groups
diagrams with the generation output on disk (`DataPaths.diagrams()` lives
right next to `DataPaths.clouds()`), and the docstring in the brief itself
describes the contract as "generation writes point clouds/diagrams to
disk."

## New scaffolding added (not experiments — interfaces/config/registries only)

1. **`ENCODER_REGISTRY`** in `encoders/__init__.py` — a
   `Registry[Encoder]` (same class every other registry in this repo
   uses), registering the 8 existing encoder classes under their existing
   conventional names (`coordconv_pi`, `scaleconv_pi`, `towerconv_pi`,
   `persistence_image`, `point_cloud`, `sequence_cnn`, `stats`). Purely
   additive — no experiment is changed to use it yet, so today's direct
   `from cloudforger.encoders.coordconv_pi import CoordConvPIEncoder`
   imports keep working unchanged. This closes the one real registry gap
   found in Phase 1 (every other pluggable axis — process, filtration,
   bifiltration, scalar feature — already had one; encoders didn't).
2. **A documented (not implemented) per-k-encoder extension point** — a
   comment in `PIMultiK.__init__`, see (a) above.
3. **`calibration/` promoted to a first-class package** — see (b) above.
   No new function is added; `build_combo_images`/`main` in
   `scripts/featurize_sigma_sweep.py` already are the "one combo" /
   "the grid" split the brief asks for.

Deliberately **not** added: an `EncoderConfig` dataclass in `config.py`.
`ProcessConfig`/`FiltrationConfig`/`FeatureConfig`/`MethodConfig` all exist
because something in the current pipeline consumes them today. Nothing
consumes a per-encoder config yet (see (a)), and `MethodConfig.params` is
already a free-form dict that can carry a future `encoders: [...]` list
without a new dataclass. Adding one now would be schema nobody reads —
scaffolding that looks real but silently rots. When a future change
actually threads per-k encoders through `PIMultiK`, that's the moment to
add the matching config type next to the code that reads it.

## Deviations from the brief's sketch, and why

| brief's sketch | this design | why |
|---|---|---|
| `data_generation/`, `vectorization/`, ... as top-level repo dirs | subpackages under `src/cloudforger/` | preserves the working `setuptools` package root and every existing `cloudforger.*` import/CLI invocation (principle 1) |
| `experiments/configs/` for grid-search + experiment configs | configs stay at top-level `configs/runs/**/*.yaml` | these are YAML run configs (data), not package source; `configs/` already is the established, working convention every script/test/doc expects via `load_config(path)`. Nesting data files inside an installable `src/` package is non-standard packaging for no benefit. |
| (no explicit `training/` package) | `nn/data.py`/`splits.py`/`train.py` → new `training/` package | these are training *mechanics* (loss loop, `Dataset` wrappers), not "encoders" or "models" (architecture) or "experiments" (orchestration) — forcing them into one of those would blur exactly the separation-of-concerns the brief is asking for |
| (`baselines/`, `stats/` not mentioned) | left as their own top-level `cloudforger` subpackages, untouched | `baselines/` (mincontrast/palm/vihrs) are alternative end-to-end estimators that span generation→evaluation on their own — they aren't a pipeline *stage*, they're a comparison axis orthogonal to the six stages. `stats/` is flagged as a likely orphan (inventory Judgment call #1) rather than assigned a new home speculatively. |
| pi_multik files renamed for clarity when consolidating | filenames kept **identical**, only nested under `experiments/pi_multik/` | keeps the pi_multik consolidation a pure mechanical move (import-path-only diff per file) rather than a move+rename; the resulting `experiments/pi_multik/pi_multik.py` nesting "stutter" is a small, deliberate cosmetic cost traded for zero renaming risk |

## `_attic/` policy

Three archived items, each with a one-line reason (also in
`_attic/README.md`, written when the moves happen):

1. `_attic/legacy/` — the entire pre-existing `legacy/` tree. Reason: its
   own imports reference packages that no longer exist
   (`cloudforger.tda.*`, `cloudforger.nn.train_old`); everything in it has
   a confirmed successor in `src/cloudforger/` + `scripts/` (see the
   inventory's superseded-by table), except `legacy/scripts/processing/classify/*`
   (an abandoned classification-track experiment with no successor at
   all — kept for reference, not because anything still needs it).
2. `_attic/vectorizers_retired/adaptive_persistence_image.py` — retired
   per-diagram adaptive-`sigma` design; the file's own header comment
   already documents why it lost to the fixed-`sigma_pixels` design.
3. `_attic/core_shims/betti.py` — deprecated compatibility shim whose only
   callers were inside `legacy/`.

Nothing is hard-deleted in this pass except gitignored `__pycache__`/
`.pyc` build cruft (not tracked by git, not part of any commit either way).

## Full old → new path mapping

Every file's proposed new path is in [refactor_inventory.md](refactor_inventory.md)'s
per-directory tables (Phase 1 also required that table, so it isn't
duplicated verbatim here to avoid the two docs drifting out of sync). The
condensed version, one row per moving directory:

| old | new |
|---|---|
| `src/cloudforger/core/calibration.py` | `src/cloudforger/calibration/diagram_calibration.py` |
| `src/cloudforger/core/betti.py` | `_attic/core_shims/betti.py` |
| `src/cloudforger/core/design.py` | `src/cloudforger/data_generation/design.py` |
| `src/cloudforger/processes/` | `src/cloudforger/data_generation/point_processes/` |
| `src/cloudforger/filtration/` | `src/cloudforger/data_generation/filtration/` |
| `src/cloudforger/vectorizers/` (minus adaptive) | `src/cloudforger/vectorization/persistence_images/` |
| `src/cloudforger/vectorizers/adaptive_persistence_image.py` | `_attic/vectorizers_retired/adaptive_persistence_image.py` |
| `src/cloudforger/features/` | `src/cloudforger/vectorization/scalar_features/` |
| `src/cloudforger/nn/encoders/` | `src/cloudforger/encoders/` |
| `src/cloudforger/nn/models/` | `src/cloudforger/models/` |
| `src/cloudforger/nn/heads/` | `src/cloudforger/models/heads/` |
| `src/cloudforger/nn/data.py`, `nn/splits.py`, `nn/train.py` | `src/cloudforger/training/` |
| `src/cloudforger/nn/experiments/` (flat files) | `src/cloudforger/experiments/` (flat files) |
| `src/cloudforger/nn/experiments/pi_multik*.py` (5 files) | `src/cloudforger/experiments/pi_multik/*.py` |
| `legacy/` | `_attic/legacy/` |
| everything under `src/cloudforger/core/` not listed above, `baselines/`, `stats/`, `scripts/`, `tests/`, `configs/`, `docs/` (existing), `notebooks/`, `data/`, `results/`, `.vendor/` | unchanged path |

## How future work plugs in (scaffolding walkthrough)

**Add a new point process.** Write `data_generation/point_processes/my_process.py`
with a class implementing `PointProcess` (`core/base.py`), decorate it
`@REGISTRY.register("my_process")` in that file, add one import line to
`data_generation/point_processes/__init__.py`. Nothing in
`data_generation/filtration/`, `vectorization/`, or any script changes —
`process.name: my_process` in a RunConfig YAML is now valid.

**Add a new scalar feature.** Write `vectorization/scalar_features/my_feature.py`
with a class implementing `DiagramFeature` (`.base`), decorate it
`@REGISTRY.register("my_feature")`, add one import line to
`vectorization/scalar_features/__init__.py`. `scripts/featurize.py`'s
`_FEATURE_HANDLERS` needs one new entry mapping `"my_feature"` to a
compute function (mirroring `_compute_persistence_entropy_standalone`) —
that dispatch table is the one place feature *computation* (as opposed to
feature *definition*) is wired to the CLI, unchanged by this refactor.

**Add a new CNN variant.** Write `encoders/my_encoder.py` implementing
`Encoder` (`.base`), register it in `ENCODER_REGISTRY`
(`encoders/__init__.py`). Write `experiments/my_experiment.py`
(or `experiments/pi_multik/my_variant.py` if it's a pi_multik sibling,
following the existing pattern of `pi_multik_scaleconv.py`/`_towers.py`)
implementing `Experiment` or `MultiSourceExperiment`
(`experiments/base.py`/`common.py`), decorate `@register("my_method")`,
add one import line to `experiments/__init__.py` (or
`experiments/pi_multik/__init__.py`). `method.name: my_method` in a
RunConfig YAML is now valid; `scripts/train.py`/`evaluate.py` pick it up
with no further changes.

**Sweep calibration/resolution.** `scripts/featurize_sigma_sweep.py
configs/runs/<process>/<config>.yaml --sigma-pixels 1.0 1.5 2.0 --coverage
0.95 0.99` — already works today, unchanged by this pass.
