# point-process-tda

Persistence-homology / topological-data-analysis pipeline for estimating
spatial point-process parameters (Thomas, nested Thomas, Matérn, ...) from
persistence images and related topological summaries, compared against
classical (mincontrast, Palm-likelihood) and neural (Vihrs 2022) baselines.

The reference pipeline is **pi_multik**: for a spatial point cloud, compute
DTM-filtration persistence diagrams at several neighborhood sizes
k ∈ {5, 10, 15}, vectorize each into a persistence image, and feed all k's
images through a shared-weight CNN whose per-k embeddings are late-fused
into a parameter-estimation head. Everything else in the codebase — other
point processes, other vectorizations, other fusion architectures — is
organized around being a comparable variant of, or component reusable by,
that pipeline.

This layout is the result of a housekeeping/restructuring pass
(`refactor/pipeline-housekeeping`); see [docs/architecture.md](docs/architecture.md)
for the full design rationale and [docs/refactor_inventory.md](docs/refactor_inventory.md)
for a file-by-file account of what moved from where.

## Quickstart

```bash
pip install -e .
python scripts/generate.py   configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml
python scripts/featurize.py  configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml
python scripts/train.py      configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml
python scripts/evaluate.py   configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml --methods pi_multik
```

Each stage reads the previous stage's disk output and writes its own
(`data/<process>/clouds.pkl` → `data/<process>/dtm_k<k>/{diagrams,persistence_image}.pkl`
→ `results/<process>/dtm_k5+10+15/pi_multik/seed_<seed>/results.json`), so
any stage can be re-run in isolation without recomputing the others. See
`configs/runs/**/*.yaml` for the full set of process/method combinations,
and each script's own docstring (`python scripts/train.py --help`) for
flags like `--set path.to.field=value`, `--seed`, `--force`, `--run-tag`.

Run the test suite with `pytest tests/` (add `-m slow` to include the
full-CLI end-to-end tests, which take ~15s and need no external data).

## Layout: one package per pipeline stage

```
src/cloudforger/
  data_generation/        point clouds -> persistence diagrams
    point_processes/        one module per process, behind a registry
    filtration/              Rips / DTM / bifiltration, behind a registry
    design.py                 design-space sampling (YAML spec -> clouds)
  vectorization/            diagrams -> persistence images + scalar features
    persistence_images/       PersistenceImager, resolution/sigma as config
    scalar_features/           Betti curves, persistence entropy, behind a registry
  calibration/              diagram-sample -> axis bounds / resolution / sigma_pixels
  encoders/                 CNN branches (one per input modality), behind a registry
  models/                   encoder+head composition, output heads
  training/                  Dataset wrappers, train/eval loop, splitting
  experiments/               the method registry scripts/train.py dispatches on
    pi_multik/                 the reference pipeline + its 4 comparison siblings
  baselines/                classical (mincontrast, Palm) + neural (Vihrs) comparisons
  stats/                    pre-topology point-cloud statistics (see note below)
  core/                     shared vocabulary: PointCloud, Region, PersistenceDiagram,
                             Registry[T], pickle/YAML IO, splits, metrics, records
  config.py / paths.py / provenance.py   cross-cutting: RunConfig schema,
                             deterministic data/results paths, run provenance

scripts/     thin CLI entry points wiring the above together (generate,
             featurize, featurize_bifiltration, featurize_sigma_sweep,
             featurize_topo_superset, train, evaluate, archive_run, ...)
configs/     RunConfig YAML files, one per process/method combination
tests/       pytest suite (registry smoke tests + full-CLI e2e tests)
notebooks/   exploratory/diagnostic analysis (calibration sweeps, sigma
             diagnostics, ...), not part of the pipeline proper
docs/        this restructure's design docs, plus pre-existing research
             reports/figures
_attic/      archived/superseded code, kept for history -- see its own
             README.md for what's there and what replaced it
```

A file lives in `core/` if two or more stages depend on it; it lives in a
stage package if only that one stage does. `baselines/` sits outside the
six stage packages by design — each baseline is an alternative end-to-end
estimator (its own generation-through-evaluation path via `vihrs.py`'s own
feature extraction, or classical M-estimators), not a stage of the CNN
pipeline. `stats/` (`CloudStatistic`, `PairDistanceCDF`) has no callers in
the current pipeline (flagged, not removed, in `docs/refactor_inventory.md`
— it may be an intentionally-kept-for-later utility or a genuine orphan;
unresolved judgment call).

## Adding things

**A new point process.** Write `data_generation/point_processes/my_process.py`
with a class implementing `PointProcess` (`core/base.py`); decorate it
`@REGISTRY.register("my_process")` in that same file; add one import line
to `data_generation/point_processes/__init__.py`. `process: {name: my_process}`
in a RunConfig YAML is now valid — nothing else changes.

**A new scalar feature.** Write `vectorization/scalar_features/my_feature.py`
with a class implementing `DiagramFeature` (`.base`); decorate it
`@REGISTRY.register("my_feature")`; add one import line to
`vectorization/scalar_features/__init__.py`. Then add one entry to
`scripts/featurize.py`'s `_FEATURE_HANDLERS` dict mapping `"my_feature"` to
a compute function (mirroring the existing `persistence_entropy` handler)
— that's the one place feature *computation* (as opposed to feature
*definition*) is wired into the CLI.

**A new encoder / CNN variant.** Write `encoders/my_encoder.py` implementing
`Encoder` (`.base`); register it in `encoders/__init__.py`'s
`REGISTRY: Registry[Encoder]`. Write `experiments/my_experiment.py` (or
`experiments/pi_multik/my_variant.py` if it's a pi_multik sibling — see
`pi_multik_scaleconv.py`/`pi_multik_towers.py` for the pattern) implementing
`Experiment` or `MultiSourceExperiment` (`experiments/base.py`/`common.py`),
decorated `@register("my_method")`; add one import line to
`experiments/__init__.py`. `method: {name: my_method}` in a RunConfig YAML
is now valid, and `scripts/train.py`/`evaluate.py` pick it up automatically.

**Sweep calibration (sigma_pixels) or resolution.**
`resolution`/`sigma_pixels`/`pd_calibration_coverage` are already plain
config values under a `features:` entry's `params:` — see
`configs/runs/nested_thomas/nested_thomas_pi_multik_k5k10k15.yaml`. For an
actual grid sweep (not just picking one value), use
`scripts/featurize_sigma_sweep.py <config> --sigma-pixels 1.0 1.5 2.0
--coverage 0.95 0.99` — `build_combo_images` there is the "run one combo"
function; `main`'s `itertools.product` is the grid.

**k=5,10,15 with a separate encoder per k.** Not implemented (this would
change `PIMultiK`'s parameter count/behavior, out of scope for a
housekeeping pass) — but `encoders/__init__.py`'s `REGISTRY` and the
comment at `PIMultiK.__init__`'s single `self.encoder =` line document
where this would hook in. See `docs/architecture.md`'s "New scaffolding"
section for the full reasoning.

## Where things went (if you're looking for something that moved)

| used to be at | now at |
|---|---|
| `processes/` | `data_generation/point_processes/` |
| `filtration/` | `data_generation/filtration/` |
| `core/design.py` | `data_generation/design.py` |
| `vectorizers/` | `vectorization/persistence_images/` |
| `features/` | `vectorization/scalar_features/` |
| `core/calibration.py` | `calibration/diagram_calibration.py` |
| `nn/encoders/` | `encoders/` |
| `nn/models/`, `nn/heads/` | `models/`, `models/heads/` |
| `nn/data.py`, `nn/splits.py`, `nn/train.py` | `training/` |
| `nn/experiments/` | `experiments/` |
| `nn/experiments/pi_multik*.py` (5 files) | `experiments/pi_multik/*.py` |
| top-level `legacy/` | `_attic/legacy/` |

Full file-by-file mapping: [docs/refactor_inventory.md](docs/refactor_inventory.md).
Design rationale and every deviation from a plain literal restructure:
[docs/architecture.md](docs/architecture.md).
