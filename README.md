# Topological Feature Learning for Parameter Estimation in Neyman–Scott Processes

First-year progression report — Flavio Gualtieri.
The report is [`writeup/short_report.tex`](writeup/short_report.tex)
([compiled PDF](writeup/short_report.pdf)); this repository is the code and
results behind it.

## Summary

Spatial point processes model clustered natural phenomena — galaxy
clustering, cell positions, tree stands — and their parameters (parent
intensity `κ`, offspring intensity `μ`, cluster scale `σ`) are interpretable
scientific quantities. Classical estimators (minimum contrast, Palm
likelihood, Bayesian MCMC) need a closed-form summary statistic, are
sensitive to hand-chosen hyperparameters, and become unstable as the
process approaches complete spatial randomness.

**ParamNet** replaces the closed form with a learned topological feature:
a point cloud is turned into DTM-filtration persistence diagrams at several
neighbourhood sizes `k ∈ {5, 10, 15}`, each vectorized into a calibrated
`64×64` persistence image, encoded by a small CoordConv CNN, late-fused
across `k`, and mapped to `θ̂` by a feed-forward head. It is trained purely
on synthetic clouds from a reparametrized design distribution. The report's
contributions are the per-diagram coverage-quantile calibration of the
persistence images, application to the two-level nested Thomas process, an
explicit ablation of the design choices, a comparison of TDA vectorizations
(image / landscape / silhouette / Betti curve), and benchmarking against
classical and neural (Vihrs 2022) baselines on shared splits. Headline
finding: persistence images carry real signal (about an order of magnitude
below label-shuffle chance) but do not beat a well-chosen radial summary on
single-level Thomas; the topological pipeline's advantage is confined to
nested Thomas and to its density/count targets.

## Layout

```
writeup/short_report.{tex,pdf}   the report (source of truth) + its 2 figures in figs/
scripts/                         the pipeline entry points (below)
src/cloudforger/                 the package the scripts wire together (unchanged)
configs/runs/{thomas,nested_thomas}/   one YAML per process/method
results/                         per-seed results.json behind the report's tables
param_sweeps/                    calibration + encoder sweeps (scripts/ + results/), separate on purpose
figs/                            calibration_comparison.pdf, vectorization_schematic.pdf
tests/                           pytest suite (registry smoke + full-CLI e2e)
```

Anything not needed to read the report lives under an `archive/` subfolder
(`writeup/archive/`, `scripts/archive/`, `figs/archive/`, `configs/archive/`).
Whole superseded areas — older write-ups, exploratory notebooks, cluster job
scripts, the vendored `multipers` C++ source — are not on this branch; they
live in full on **`backend/full-history`** (`git show backend/full-history:_attic/...`).

## How to run

```bash
pip install -e .

# one process/method, four stages, each reads the previous stage's disk output
python scripts/generate.py  configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml
python scripts/featurize.py configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml
python scripts/train.py     configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml
python scripts/evaluate.py  configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml --methods pi_multik
```

Nested Thomas uses `configs/runs/nested_thomas/pi_multik.yaml` (fused
DTM `k = 5,10,15`). Baselines are ordinary configs in the same folders:
`*_vihrs.yaml`, `mincontrast.yaml`, `mincontrast_g.yaml`.

The two report figures:

```bash
python scripts/make_calibration_comparison_figure.py   # -> figs/calibration_comparison.pdf
python scripts/make_vectorization_schematic.py          # -> figs/vectorization_schematic.pdf
```

Parameter sweeps (report §3.2–3.3) are driven from `param_sweeps/scripts/`
(`run_pi_calibration_sweep.py`, `featurize_sigma_sweep.py`) and land in
`param_sweeps/results/`.

Tests: `pytest tests/` (add `-m slow` for the ~1 min end-to-end CLI test).

## Where the report's numbers come from

| Report element | Code / results |
|---|---|
| Simulator (§3.1) | `scripts/generate.py`, `src/cloudforger/data_generation/` |
| Features | `scripts/featurize.py`, `src/cloudforger/vectorization/`, `.../calibration/diagram_calibration.py` |
| ParamNet + ablations (Tables 1–3) | `scripts/train.py`, `scripts/evaluate.py`, `src/cloudforger/experiments/pi_multik/`, `results/{thomas,nested_thomas}/` |
| Vectorization comparison (Table 3) | `scripts/collect_vectorization_results.py`, `results/**/vec_multik_*`, `.../betti_multik*` |
| Baselines | `src/cloudforger/baselines/{vihrs,mincontrast,mincontrast_g}.py` |
| Encoder ablation (Table 3, "shared / independent, concat") | `method: pi_multik` with `method.params.encoder_mode ∈ {shared, independent}`, `fusion_mode: concat` |
| Calibration "Naive (q=1.00)" row | `param_sweeps/results/**/pi_multik/_runs/calib_q100/` |

`src/cloudforger/` is left exactly as on `backend/full-history`; modules for
bifiltration / multiparameter-persistence experiments
(`experiments/mph_*`, `data_generation/filtration/bifiltration.py`,
`core/signed_measure.py`, `vectorization/persistence_images/signed_measure_image.py`)
are present but are not part of this report.
