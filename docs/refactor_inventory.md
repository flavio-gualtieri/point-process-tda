# Refactor inventory

Phase 1 (discovery) output for `refactor/pipeline-housekeeping`. Read-only
pass over the whole repository, done before any file was moved. See
[architecture.md](architecture.md) for the target layout and rationale;
this document is the "what exists today and where it's going" ledger.

## Headline finding

The codebase is **not** the ad-hoc pile of one-off scripts the refactor
brief anticipated. `src/cloudforger/` is a real, mostly well-organized
installable package that *already* has:

- a generic `Registry[T]` (`core/registry.py`) used by point processes,
  filtrations, bifiltrations, and scalar features today (confirmed by
  `tests/test_registries.py`);
- a deterministic, two-stage data/results path contract (`paths.py`:
  `data/<process>/clouds.pkl` → `data/<process>/<filtration_tag>/{diagrams,<feature>}.pkl`
  → `results/<process>/<filtration_tag>/<method>/seed_<seed>/`);
- calibration (persistence-image axis bounds + resolution + `sigma_pixels`)
  already exposed as plain config dict params, with a dedicated grid-search
  script (`scripts/featurize_sigma_sweep.py`) that already isolates "run one
  (sigma_pixels, coverage) combo" (`build_combo_images`) from "loop over the
  grid" (`main`);
- a **pre-existing, self-contained dead tree** at top-level `legacy/`
  (56 files, 11,080 LOC — nearly as large as the active codebase) whose
  imports reference packages that no longer exist
  (`cloudforger.tda.*`, `cloudforger.nn.train_old`), i.e. a previous
  restructure already happened and this is its "before" snapshot, never
  deleted.

So this pass is overwhelmingly: (1) regroup already-modular subpackages
under the six pipeline-stage names the brief asks for, (2) consolidate the
`pi_multik` family into one subpackage, (3) archive the already-dead
`legacy/` tree plus two smaller retired/deprecated files, (4) add the two
or three pieces of registry scaffolding that don't already exist
(encoders, and documenting the calibration/resolution sweep entry point).
It is emphatically not a rewrite — see [architecture.md](architecture.md)
for what was deliberately left untouched and why.

Counts (excluding `.vendor/`, the vendored `multipers` build):

| area | files | LOC |
|---|---|---|
| `src/` + `scripts/` + `tests/` (active) | 102 | 11,815 |
| `legacy/` (dead, pre-existing archive) | 56 | 11,080 |
| `notebooks/` | 6 notebooks | — |

## How to read the table

- **used by pi_multik?** — y if the reference k=5,10,15 experiment
  (`nn/experiments/pi_multik.py`, method name `pi_multik`, driven by
  `configs/runs/{thomas,nested_thomas}/*_pi_multik_k5k10k15.yaml`) imports
  it, directly or transitively. n/a for things like tests/docs/other
  experiments that are legitimately part of the repo but outside that one
  pipeline.
- **status** — `keep` (move only, no behavior change), `merge` (folded into
  a sibling file as part of a mechanical consolidation), `move`
  (relocated, used as the anchor row for a directory-level move), `archive`
  (relocated to `_attic/`, dead or retired), `flag` (left in place;
  ambiguous enough that I did not want to guess — see Judgment calls at the
  end).

## `src/cloudforger/` — top-level modules

| current path | purpose | used by pi_multik? | status | proposed new path |
|---|---|---|---|---|
| `src/cloudforger/__init__.py` | package re-exports (`PointCloud`, `Region`, `Box`, `PointProcess`, `PoissonProcess`) | y | keep | same path; one import updated |
| `src/cloudforger/config.py` | `RunConfig`/`ProcessConfig`/`FiltrationConfig`/`BifiltrationConfig`/`FeatureConfig`/`MethodConfig` dataclasses; YAML load + `--set` dotted-path overrides | y | keep | same path, untouched |
| `src/cloudforger/paths.py` | `DataPaths`/`ResultsPaths`: the one place deriving every data/results path deterministically | y | keep | same path; one import updated |
| `src/cloudforger/provenance.py` | per-run git-commit/dirty-flag provenance stamp + `results/experiments.jsonl` ledger | y | keep | same path; comment updated |

## `src/cloudforger/core/` — shared domain vocabulary

Kept together as `core/` almost unchanged: these are the types and helpers
used by *two or more* pipeline stages (point cloud, region, diagram,
signed measure, registry base class, pickle/YAML IO, label splitting,
metrics, record (de)serialization). Only the two single-purpose/dead files
move out.

| current path | purpose | used by pi_multik? | status | proposed new path |
|---|---|---|---|---|
| `core/__init__.py` | package re-exports | y | keep | same path; stale comment fixed |
| `core/base.py` | `PointProcess` ABC | y | keep | same path |
| `core/cloud.py` | `PointCloud` dataclass | y | keep | same path |
| `core/region.py` | `Region`/`Box` | y | keep | same path |
| `core/diagram.py` | `PersistenceDiagram` dataclass | y | keep | same path |
| `core/signed_measure.py` | `SignedMeasure` dataclass (multiparameter track) | n | keep | same path |
| `core/registry.py` | generic `Registry[T]` (name→class), already backing 4 registries | y | keep | same path |
| `core/io.py` | pickle/YAML helpers, `intersect_seeds`/`align_seeds` | y | keep | same path |
| `core/splits.py` | `train_val_test_split`/`_indices` | y | keep | same path |
| `core/metrics.py` | marginal/paired metric formulas shared by `scripts/evaluate.py` | n (evaluate-stage) | keep | same path |
| `core/records.py` | domain object ⟷ pickle-dict record conversions | y | keep | same path; one import updated |
| `core/features.py` | `CorrelationFeatures` dataclass (unrelated to the top-level `features/` package despite the name) | n | keep | same path |
| `core/utils.py` | `standardize_size` | n | keep | same path |
| `core/calibration.py` | diagram percentile stats → axis bounds; bifiltration grid sizing | y | **move** | `calibration/diagram_calibration.py` |
| `core/betti.py` | 4-line deprecated shim: `"moved to cloudforger.features.result... Shim for not-yet-migrated callers"`; its only callers are inside `legacy/` | n | **archive** | `_attic/core_shims/betti.py` |

## Data generation stage (point clouds → diagrams)

| current path | purpose | used by pi_multik? | status | proposed new path |
|---|---|---|---|---|
| `processes/__init__.py` | `REGISTRY: Registry[PointProcess]` assembly | y | move | `data_generation/point_processes/__init__.py` |
| `processes/poisson.py` | homogeneous Poisson process | n (thomas/nested_thomas only) | move | `data_generation/point_processes/poisson.py` |
| `processes/matern.py` | Matérn hard-core process | n | move | `data_generation/point_processes/matern.py` |
| `processes/thomas.py` | Thomas cluster process | y | move | `data_generation/point_processes/thomas.py` |
| `processes/nested_thomas.py` | Thomas-of-Thomas nested cluster process (the other pi_multik reference process) | y | move | `data_generation/point_processes/nested_thomas.py` |
| `processes/neyman_scott.py` | general Neyman-Scott process (parent class of Thomas) | y | move | `data_generation/point_processes/neyman_scott.py` |
| `processes/inhom_thomas.py` | inhomogeneous-intensity Thomas variant | n | move | `data_generation/point_processes/inhom_thomas.py` |
| `core/design.py` | design-space sampling: YAML design spec → parameter vectors → sampled clouds; used only by `scripts/generate.py` | y | **move** | `data_generation/design.py` |
| `filtration/__init__.py` | `REGISTRY: Registry[Filtration]` + `BIFILTRATION_REGISTRY: Registry[Bifiltration]` | y | move | `data_generation/filtration/__init__.py` |
| `filtration/base.py` | `Filtration` ABC | y | move | `data_generation/filtration/base.py` |
| `filtration/rips.py` | Vietoris–Rips filtration | n (excluded from pi_multik: degenerate H0 birth axis) | move | `data_generation/filtration/rips.py` |
| `filtration/dtm.py` | Distance-to-measure filtration (the k=5,10,15 channel) | y | move | `data_generation/filtration/dtm.py` |
| `filtration/bifiltration.py` | `Bifiltration`/`DtmRipsBifiltration` (multiparameter/mph track) | n | move | `data_generation/filtration/bifiltration.py` |

## Vectorization stage (diagrams → persistence images + scalar features)

| current path | purpose | used by pi_multik? | status | proposed new path |
|---|---|---|---|---|
| `vectorizers/__init__.py` | package re-exports | y | move | `vectorization/persistence_images/__init__.py` |
| `vectorizers/persistence_image.py` | `PersistenceImager`: paper-faithful (Adams et al. 2017) fixed-`sigma_pixels` imager — the active design | y | move | `vectorization/persistence_images/persistence_image.py` |
| `vectorizers/multi_channel.py` | `MultiChannelImager`: stacks per-homology-dim imagers | y | move | `vectorization/persistence_images/multi_channel.py` |
| `vectorizers/calibrated.py` | `build_calibrated_imager`: auto-calibrates ranges from a diagram sample | y | move | `vectorization/persistence_images/calibrated.py` |
| `vectorizers/signed_measure_image.py` | signed-measure imagers (multiparameter/mph track) | n | move | `vectorization/persistence_images/signed_measure_image.py` |
| `vectorizers/adaptive_persistence_image.py` | retired per-diagram adaptive-`sigma` design; header comment: *"Superseded by the paper-faithful fixed-sigma `PersistenceImager`... kept only for comparison"*; not imported by `vectorizers/__init__.py`; only caller is `notebooks/pi_sigma_diagnostics.ipynb` (which itself labels the import "retired design") | n | **archive** | `_attic/vectorizers_retired/adaptive_persistence_image.py` |
| `features/__init__.py` | `REGISTRY: Registry[DiagramFeature]` | y (persistence_entropy) | move | `vectorization/scalar_features/__init__.py` |
| `features/base.py` | `DiagramFeature` ABC | y | move | `vectorization/scalar_features/base.py` |
| `features/betti_curve.py` | `BettiCurve` feature | n (pi_multik uses PI + entropy, not Betti curves) | move | `vectorization/scalar_features/betti_curve.py` |
| `features/persistence_entropy.py` | `PersistenceEntropy` — the per-(dim,k) scalar feature pi_multik can optionally append (`include_entropy`) | y | move | `vectorization/scalar_features/persistence_entropy.py` |
| `features/calibrated.py` | `build_calibrated_betti_curves` | n | move | `vectorization/scalar_features/calibrated.py` |
| `features/result.py` | `BettiCurveFeature` dataclass | n | move | `vectorization/scalar_features/result.py` |

Note: the `docs/pi_multik_report.tex`-referenced "log(log(death/birth))"
style diagnostics (see the `loglog_death_birth_vs_params*.ipynb` notebooks)
are exploratory notebook analyses, not a registered scalar feature in
`features/` today — see Judgment calls.

## Calibration (persistence-image axis ranges, resolution, `sigma_pixels`)

| current path | purpose | used by pi_multik? | status | proposed new path |
|---|---|---|---|---|
| `core/calibration.py` | `calibrate`/`axis_bounds`/`calibrate_report`/`bifiltration_grid`/`check_grid_resolves` | y | **move** (new package) | `calibration/diagram_calibration.py` + thin `calibration/__init__.py` |
| `scripts/featurize_sigma_sweep.py` | grid-search entry point: cartesian product of `--sigma-pixels` × `--coverage`; `build_combo_images` already isolates one combo | y (sweeps its channels) | keep | same path (already satisfies "one obvious function per combo") |

## Encoders (CNN branches, one family per input modality)

| current path | purpose | used by pi_multik? | status | proposed new path |
|---|---|---|---|---|
| `nn/encoders/__init__.py` | package re-exports (no registry today) | y | move + **new registry added** | `encoders/__init__.py` |
| `nn/encoders/base.py` | `Encoder` ABC | y | move | `encoders/base.py` |
| `nn/encoders/coordconv_pi.py` | `CoordConvPIEncoder` — the CNN branch pi_multik applies per k | y | move | `encoders/coordconv_pi.py` |
| `nn/encoders/scaleconv_pi.py` | `ScaleConvFusion` — ordered-k-axis Conv1d fusion (pi_multik_scaleconv) | y (scaleconv variant) | move | `encoders/scaleconv_pi.py` |
| `nn/encoders/towerconv_pi.py` | `TowerConv` (pi_multik_towers variant) | n | move | `encoders/towerconv_pi.py` |
| `nn/encoders/persistence_image.py` | `PIEncoder` — plain single-image CNN (used by `pi`, `fusion`, `ph_combined`) | n | move | `encoders/persistence_image.py` |
| `nn/encoders/point_cloud.py` | `PointNetEncoder` (raw_pc experiment) | n | move | `encoders/point_cloud.py` |
| `nn/encoders/sequence_cnn.py` | `SequenceCNNEncoder` (betti_cnn, fusion, ph_combined) | n | move | `encoders/sequence_cnn.py` |
| `nn/encoders/stats.py` | `StatsEncoder` (betti, pairwise) | n | move | `encoders/stats.py` |

## Models (encoder+head composition, output heads)

| current path | purpose | used by pi_multik? | status | proposed new path |
|---|---|---|---|---|
| `nn/models/__init__.py` | package re-exports | y | move | `models/__init__.py` |
| `nn/models/single_modal.py` | `SingleModalModel(encoder, head)` | y (base `Experiment` class uses it) | move | `models/single_modal.py` |
| `nn/models/multi_modal.py` | `MultiModalModel` (several encoders → shared head) | n | move | `models/multi_modal.py` |
| `nn/heads/__init__.py` | package re-exports | y | move | `models/heads/__init__.py` |
| `nn/heads/paramest.py` | `ParameterEstimator` — the regression head pi_multik's fusion output feeds | y | move | `models/heads/paramest.py` |
| `nn/heads/classifier.py` | `ClassificationHead` | n | move | `models/heads/classifier.py` |

## Training utilities (dataset wrappers, train/eval loop, splitting)

Kept as one small package rather than folded into `models/`: these are
training *mechanics* (loss computation, epoch loop, `Dataset` wrappers),
orthogonal to architecture definition, and are shared by nearly every
experiment regardless of which encoder/model it uses.

| current path | purpose | used by pi_multik? | status | proposed new path |
|---|---|---|---|---|
| `nn/data.py` | `PersistenceImageDataset`/`BettiCurveDataset`/`PointCloudDataset`/`CorrelationFeatureDataset` | n (pi_multik builds tensors directly, not via these) | move | `training/data.py` |
| `nn/splits.py` | thin re-export of `core.splits.train_val_test_split` | y (via `experiments/base.py`) | move | `training/splits.py` |
| `nn/train.py` | `train_one_epoch`/`evaluate`/`evaluate_per_target` | y | move | `training/train.py` |

## Experiments (CNN method registry; pi_multik is the reference pipeline)

`nn/experiments/__init__.py` registers 16 method names today. Every one
imports cleanly and is reachable from `scripts/train.py`'s dispatcher, so
none is dead in the "broken import" sense `legacy/` is — see Judgment
calls for the two (`pairwise`, plain `betti`) that look unreachable in
practice for a different reason (no `configs/runs/*.yaml` selects them and
no `scripts/featurize.py` handler produces their input file).

| current path | purpose | used by pi_multik? | status | proposed new path |
|---|---|---|---|---|
| `nn/experiments/__init__.py` | registers all 16 experiment classes | y | move | `experiments/__init__.py` |
| `nn/experiments/base.py` | `Experiment` ABC, `REGISTRY`, `build_experiment`, shared `run()` orchestration, classification helpers | y | move | `experiments/base.py` |
| `nn/experiments/common.py` | `MultiSourceExperiment` ABC, `save_results` (the one place every method writes `results.pt`/`results.json`), label-norm fit/apply helpers | y | move | `experiments/common.py` |
| `nn/experiments/raw_pc.py` | `RawPointCloudExperiment` (`raw_pc`) — PointNet directly on coordinates, no TDA | n | move | `experiments/raw_pc.py` |
| `nn/experiments/pairwise.py` | `PairwiseExperiment` (`pairwise`) | n — **see Judgment calls** | move | `experiments/pairwise.py` |
| `nn/experiments/persistence_image.py` | `PersistenceImageExperiment` (`pi`) — single-k PI baseline pi_multik is compared against | n | move | `experiments/persistence_image.py` |
| `nn/experiments/mph_pi.py` | `MPHImageExperiment` (`mph_pi`) — multiparameter/bifiltration image CNN (active sibling pipeline, see recent git history) | n | move | `experiments/mph_pi.py` |
| `nn/experiments/mph_fusion.py` | `MPHFusionExperiment` (`mph_fusion`) — vihrs ⊕ mph_pi fusion | n | move | `experiments/mph_fusion.py` |
| `nn/experiments/betti.py` | `BettiCurveExperiment` (`betti`) — stats-encoder baseline over Betti curves | n — **see Judgment calls** | move | `experiments/betti.py` |
| `nn/experiments/betti_cnn.py` | `BettiCurveCNNExperiment` (`betti_cnn`, `betti_cnn_weighted`) | n | move | `experiments/betti_cnn.py` |
| `nn/experiments/ph_combined.py` | `PHCombinedExperiment` (`ph_combined`) — fuses PI + Betti-curve encoders | n | move | `experiments/ph_combined.py` |
| `nn/experiments/fusion.py` | `FusionExperiment` (`fusion`) — vihrs L(r)−r ⊕ single-k PI/betti | n | move | `experiments/fusion.py` |
| `nn/experiments/pi_multik.py` | **reference pipeline**: `PIMultiK` model + `PIMultiKExperiment` (`pi_multik`), `load_multik_split`/`build_pi_tensor`/`build_extra` | **y — this is it** | move (consolidate) | `experiments/pi_multik/pi_multik.py` |
| `nn/experiments/pi_multik_scaleconv.py` | `PIMultiKScaleConvExperiment` (`pi_multik_scaleconv`) — swaps in `ScaleConvFusion`, reuses `pi_multik.py`'s `run()` verbatim | y (sibling variant) | move (consolidate) | `experiments/pi_multik/pi_multik_scaleconv.py` |
| `nn/experiments/pi_multik_towers.py` | `PIMultiKTowersExperiment` (`pi_multik_towers`) | y (sibling variant) | move (consolidate) | `experiments/pi_multik/pi_multik_towers.py` |
| `nn/experiments/pi_multik_earlyfusion.py` | `PIMultiKEarlyFusionExperiment` (`pi_multik_earlyfusion`) — the *old* (pre-late-fusion) pi_multik design, re-implemented as an explicit comparison sibling | y (comparison sibling) | move (consolidate) | `experiments/pi_multik/pi_multik_earlyfusion.py` |
| `nn/experiments/pi_multik_fusion.py` | `PIMultiKFusionExperiment` (`pi_multik_fusion`) — vihrs ⊕ pi_multik fusion | y (sibling variant) | move (consolidate) | `experiments/pi_multik/pi_multik_fusion.py` |

`pi_multik.py`'s own module docstring says the pre-late-fusion design "is
preserved verbatim in `experiments/delete.py`" — that file does not exist
anywhere in the repo (grepped; only self-reference found). It appears the
old design was actually kept as `pi_multik_earlyfusion.py` instead (same
role: explicit early-fusion comparison sibling) and the docstring was never
updated. I'll correct that one stale sentence while relocating the file
(mechanical accuracy fix, not a behavior change) — flagged here for
visibility rather than silently changed.

## `src/cloudforger/baselines/` and `src/cloudforger/stats/` — outside the six pipeline stages

| current path | purpose | used by pi_multik? | status | proposed new path |
|---|---|---|---|---|
| `baselines/__init__.py` | package re-exports | n (pi_multik borrows only `vihrs.fit_log_zscore`/`apply_log_zscore`) | keep | same path |
| `baselines/mincontrast.py` | classical minimum-contrast Thomas-process estimator; also a standalone CLI | n | keep | same path |
| `baselines/palm.py` | classical Palm-likelihood Thomas estimator; also a standalone CLI | n | keep | same path |
| `baselines/vihrs.py` | self-contained replication of the Vihrs (2022) neural baseline; also a standalone CLI | y (utility functions only) | keep | same path; comments updated |
| `stats/__init__.py` | package re-exports | n | **flag** | unchanged — see Judgment calls |
| `stats/base.py` | `CloudStatistic` ABC | n | **flag** | unchanged |
| `stats/pair_dist.py` | `PairDistanceCDF` | n | **flag** | unchanged |

## `scripts/` — thin CLI entry points (already matches the target role)

| current path | purpose | used by pi_multik? | status | proposed new path |
|---|---|---|---|---|
| `scripts/generate.py` | RunConfig → sampled clouds → `data/<process>/clouds.pkl` | y | keep | same path; imports updated |
| `scripts/featurize.py` | diagrams (per filtration) → betti_curve/persistence_image/persistence_entropy | y | keep | same path; imports updated |
| `scripts/featurize_bifiltration.py` | signed measures + images from bifiltrations (mph track) | n | keep | same path; imports updated |
| `scripts/featurize_sigma_sweep.py` | calibration/resolution grid sweep over already-computed diagrams | y | keep | same path; imports updated |
| `scripts/featurize_topo_superset.py` | persistence images for a chosen `topo_superset` mass-fraction subset | n | keep | same path; imports updated |
| `scripts/precompute_topo_superset.py` | precomputes the full `topo_superset` feature superset | n | keep | same path; imports updated |
| `scripts/train.py` | RunConfig → trained method for N seeds → `results.pt`/`.json`/`model.pt` | y | keep | same path; imports updated |
| `scripts/evaluate.py` | aggregates/compares `results.json` across methods and seeds | y (compares pi_multik to siblings) | keep | same path; imports updated |
| `scripts/archive_run.py` | moves a method's current results into a named `_runs/<tag>/` slot | n | keep | same path; imports updated |
| `scripts/bifiltration_smoke.py` | pre-flight check before a full bifiltration featurization run | n | keep | same path; imports updated |
| `scripts/compare_training_curves.py` | ad-hoc train/val curve overlay plot | n | keep | same path, no import change needed |
| `scripts/plot_scaleconv_vihrs_curves.py` | docstring: *"One-off training-curve comparison"* | n | keep (flagged as one-off in its own docstring; left in `scripts/` as-is) | same path, no import change needed |

## `tests/`

| current path | purpose | status | proposed new path |
|---|---|---|---|
| `tests/test_baselines.py` | smoke tests for mincontrast/palm/vihrs | keep | same path, no import change needed |
| `tests/test_pipeline_e2e.py` | subprocess-driven generate→featurize→train→evaluate smoke test on a tiny synthetic `betti_cnn_01` config | keep + **extended** | same path; add a second small `pi_multik` case (currently the *reference* pipeline has no dedicated regression test) |
| `tests/test_registries.py` | registry build/run smoke tests for processes/filtrations/features/calibrated imager | keep | same path; imports updated |

## `notebooks/` — left in place, four import cells need mechanical fixes

Not part of the directory restructure (the brief's target layout doesn't
ask for notebook reorganization), but four notebooks import modules that
are moving and would silently break next time someone runs them:

| notebook | fix needed |
|---|---|
| `compare_pi_adaptive_sigma.ipynb` | update `cloudforger.core.calibration` / `cloudforger.vectorizers.*` import lines |
| `pi_low_persistence_diagnostic.ipynb` | update `cloudforger.vectorizers.persistence_image` import line |
| `pi_sigma_diagnostics.ipynb` | update `cloudforger.core.calibration`/`cloudforger.filtration`/`cloudforger.vectorizers.multi_channel`; its `adaptive_persistence_image` import points at a file moving to `_attic/` — path updated to load it from there explicitly |
| `thomas_diffusion.ipynb` | update `cloudforger.processes.poisson` import line |
| `loglog_death_birth_vs_params.ipynb`, `loglog_death_birth_vs_params_nested_thomas.ipynb` | no change (only import `cloudforger.core.records`, which isn't moving) |

`notebooks/out/*` (generated figures/CSVs) are pipeline output, left alone.

## `legacy/` — pre-existing dead tree, archived wholesale

All 56 files under `legacy/` (`dtm_experiment/`, `models/`,
`scripts/{analysis,experiments,processing,runners}/`) plus their sibling
`legacy/configs/` and `legacy/data/`. Every file that has internal imports
references at least one of: `cloudforger.tda.*` (filtration/vectorizer/
calibration/features submodules that no longer exist), `cloudforger.nn.train_old`
(doesn't exist), or duplicates functionality now in `src/cloudforger/` +
`scripts/` under a different, generalized shape. Confirmed superseded
one-for-one, e.g.:

| legacy file | superseded by |
|---|---|
| `legacy/dtm_experiment/fusion_model.py` | `src/cloudforger/nn/experiments/fusion.py` |
| `legacy/dtm_experiment/pi_multik_model.py` (early-fusion, stacked-channel design) | `nn/experiments/pi_multik_earlyfusion.py` (clean reimplementation) + current `pi_multik.py` (late-fusion) |
| `legacy/dtm_experiment/pi_multik_fusion_model.py` | `nn/experiments/pi_multik_fusion.py` |
| `legacy/dtm_experiment/compute_features.py` | `scripts/featurize.py` + `features/`/`vectorizers/` |
| `legacy/dtm_experiment/train*.py`, `compare*.py` | `scripts/train.py` + `scripts/evaluate.py` (generalized, config-driven) |
| `legacy/dtm_experiment/vihrs_checkpointed.py`, `legacy/scripts/runners/params/run_vihrs.py` | `src/cloudforger/baselines/vihrs.py` (identical docstring header, superset of functionality) |
| `legacy/models/mincontrast.py`, `legacy/models/baselines.py` | `src/cloudforger/baselines/mincontrast.py`/`palm.py` (byte-similar, already migrated) |
| `legacy/models/evaluate.py`, `evaluate_seeds.py` | `scripts/evaluate.py` (renamed "feature"→"method" throughout, generalized) |
| `legacy/scripts/processing/classify/*` (whole separate classification track, dims 2d–16d) | no successor — appears to be an abandoned experiment direction, not migrated at all |
| `legacy/scripts/processing/params/pipeline_lib/*` | `core/io.py`, `core/records.py`, `core/design.py`, `scripts/{generate,featurize}.py` (evolved successors — `pipeline_lib/records.py`'s functions are a strict subset of current `core/records.py`'s) |

Proposed new path for all of it: `_attic/legacy/` (single directory move,
internal structure preserved). Status: **archive**.

## Out of scope (not code, or not part of this restructure)

| path | why out of scope |
|---|---|
| `data/` | gitignored, untracked, pure generated pipeline output (point clouds/diagrams/features). `git mv` doesn't apply; left untouched. |
| `results/` | tracked but is pipeline *output* (checkpoints, `results.json/.pt`), not code; already has its own organic archive convention (`results/legacy/`, `results/*/old_newer/`); reorganizing it risks large binary diffs for zero behavioral benefit. Left untouched. |
| `.vendor/multipers-2.6.1-nocgal/` | vendored third-party C++/Python build (`multipers`, CGAL-free build), not code this project authored. Left untouched. |
| `configs/runs/**/*.yaml` | already at the top level, already the convention every script/test expects (`load_config(path)`); moving these under `src/cloudforger/experiments/configs/` (as the brief's sketch literally shows) would be wrong packaging (config data inside an installable `src/` package) for zero benefit — see architecture.md. Left untouched. |
| `docs/*` (existing figures/reports) | pre-existing research output/reports (`.tex`/`.pdf`/`.md`), not code. Untouched; this pass only adds `refactor_inventory.md` and `architecture.md`. |
| `.claude/settings.local.json`, `.vscode/` | tooling config, unrelated to pipeline structure. |

## Judgment calls to confirm

1. **`nn/experiments/pairwise.py` (`pairwise` method) and `cloudforger/stats/`
   (`CloudStatistic`, `PairDistanceCDF`)** — `pairwise`'s `file_key` is
   `"pairwise"`, but `scripts/featurize.py`'s `_FEATURE_HANDLERS` only
   produces `betti_curve`/`persistence_image`/`persistence_entropy` — there
   is no current code path that writes a `pairwise.pkl`. No
   `configs/runs/*.yaml` selects `method: pairwise` either. `cloudforger/stats/`
   has zero callers outside `legacy/`. This *looks* dead, but everything
   still imports cleanly and it's fully wired into the registry/CLI method
   sets (`scripts/archive_run.py`, `scripts/train.py`,
   `scripts/evaluate.py`), unlike `legacy/`'s outright broken imports — so
   I left both in place rather than archiving them. Please confirm whether
   `pairwise`/`stats/` should move to `_attic/` too, or whether there's an
   external/manual data-prep step that still feeds it that I didn't find.
2. **`nn/experiments/betti.py` (plain `betti` method, not `betti_cnn`)** —
   same situation: registered, imports fine, but no `configs/runs/*.yaml`
   selects it. Left in place; flagging in case it's also intentionally
   retired in favor of `betti_cnn`.
3. **Stale docstring in `pi_multik.py`** pointing at a nonexistent
   `experiments/delete.py` — I'm fixing this one-line comment while moving
   the file (see the Experiments table above) since I'm confident about
   what it should say instead (`pi_multik_earlyfusion.py`), but flagging
   the correction explicitly since it's a content change, not a pure move.
4. **`core/betti.py` shim** — self-documented as a compatibility shim for
   callers that no longer exist post-archive. I archived it rather than
   deleting it outright, per "prefer archive over delete", but it's the
   closest thing in this repo to the "trivially safe to hard-delete"
   exception the brief allows. Left the more conservative choice; easy to
   actually delete later if preferred.
5. **The "log(log(death/birth))" scalar diagnostic** the task brief names
   as an example scalar feature lives only in exploratory notebooks
   (`loglog_death_birth_vs_params*.ipynb`) today, not as a registered
   `DiagramFeature`. I did not promote it into
   `vectorization/scalar_features/` myself — doing so would mean writing
   new feature code (guessing at grid/binning choices the notebooks made
   ad hoc), which is out of this pass's scope per the golden rule (this
   repo's convention is to treat those choices as diagnostic/ablation
   territory that needs a deliberate decision, not a refactor-time guess).
   The registry is ready to receive it whenever that promotion is done
   deliberately.
