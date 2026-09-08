# ParamNet — topological feature learning for spatial point processes

Learned topological features (persistence images of DTM-filtration diagrams)
for **likelihood-free inference on spatial point processes**: estimating the
latent cluster parameters of a Neyman–Scott process, and discriminating
between process families, from a *single* bounded-window realization.

**Headline results.** The topological features beat the neural
radial-summary baseline (Vihrs 2022) on the two-level **nested Thomas**
process and on **4-way process classification**, and outperform
minimum-contrast estimation on Ripley's `K` by ~7×. On single-level Thomas
they match, but do not beat, a well-chosen radial summary. Details, failure
modes and exact provenance below.

## Background

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
classical and neural (Vihrs 2022) baselines on shared splits.

## Results

### 1. Parameter estimation

Normalised loss `L` (lower is better) on the shared test split; `n = 10`
seeds. `L ≈ 1` is the label-shuffle chance level, so it doubles as an error
ceiling. Numbers are Table 1 of `writeup/short_report.pdf`.

| Method | Thomas `L` | Nested Thomas `L` |
|---|---|---|
| **ParamNet** (topological) | 0.125 ± 0.007 | **0.192 ± 0.010** |
| vihrs (`L(r)−r` CNN) | **0.115 ± 0.007** | 0.217 ± 0.006 |
| Minimum contrast, `K` | 0.855 ± 0.682 | — |
| Minimum contrast, `g` | 6.199 ± 0.339 | — |
| *chance (label shuffle)* | *1.003 ± 0.019* | *1.006 ± 0.020* |

- **Nested Thomas: ParamNet wins**, 0.192 vs 0.217 (−12%), and wins on 3 of
  the 5 individual parameters — parent intensity (0.283 vs 0.320),
  meta-offspring (0.402 vs 0.493), mean offspring (0.086 vs 0.110) — losing
  on the two cluster-scale targets. This is the regime where a closed-form
  summary statistic does not exist at all, so minimum contrast cannot be run.
- **Single-level Thomas: parity, not a win.** 0.125 vs 0.115 is within seed
  noise. Both neural methods beat minimum contrast on `K` by ~7× on
  average, and by ~3× on the 6/10 seeds where it converges (0.365–0.399);
  the remaining 4 seeds land at 0.903–2.056, which is the documented
  near-Poisson instability of contrast estimation (Waagepetersen & Guan,
  *JRSS-B* 71(3), 2009), not a tuning failure. **The `g` row is not a fair comparison and should not
  be read as one:** every seed is worse than chance, traceable to a contrast
  exponent `c = 1` set by rule of thumb where `c = 1/4` is the value
  validated for aggregated patterns. It is reported for completeness; the
  report declines to read it as a statement about contrast estimation on `g`.
- ParamNet is process-specific here: DTM`₅` for Thomas, fused
  DTM`₅,₁₀,₁₅` for nested Thomas. Nested-Thomas ParamNet is `n = 3` seeds.

### 2. Point-process model classification

4-way classification of the generating process family (Matérn cluster /
nested Thomas / Strauss / Thomas) from one realization. Identical clouds,
identical 5 seeds, identical train/val/test split for both methods — only
the feature set and the network differ. `adversarial` is a held-out split
drawn off the training design grid.

| Method | test acc | adversarial acc |
|---|---|---|
| **pi_multik** (H0+H1, DTM `k = 5,10,15`) | **0.877 ± 0.005** | **0.872 ± 0.003** |
| vihrs (`L(r)−r` + `n(x)`, 1-D CNN) | 0.840 ± 0.005 | 0.836 ± 0.006 |

Per-class recall (mean over seeds) — the topological features win or tie on
every class, and the margin is concentrated in exactly the two families a
second-order radial summary confuses:

| | Matérn cluster | nested Thomas | Strauss | Thomas |
|---|---|---|---|---|
| pi_multik | **0.637** | **0.899** | 0.997 | **0.926** |
| vihrs | 0.577 | 0.893 | **0.999** | 0.867 |

Read per-class recall rather than the aggregate: `aniso_thomas` is folded
into `thomas`, so that class carries 2× the clouds, the split is
unstratified and the loss unweighted (majority-class accuracy is 0.40).
Both methods are affected identically.

Reproduce:

```bash
python scripts/train.py configs/runs/classification/pi_multik_h0h1_k5k10k15.yaml --seed 9371
python scripts/train.py configs/runs/classification/vihrs_lr_nx_k5k10k15.yaml    --seed 9371
# -> results/classification/{dtm_k5+10+15/pi_multik,raw/vihrs}/seed_9371/results.json
```

### 3. Ablations

`writeup/short_report.pdf` Table 4 ablates filtration (Rips vs DTM vs fused
multi-`k`), homology degree (`H0` / `H1`), encoder weight sharing,
persistence-image calibration, and four TDA vectorizations (image /
landscape / silhouette / Betti curve), plus the `param_sweeps/` grids. Two
load-bearing findings: removing topology collapses the estimator to chance
(`L` 0.125 → 0.893 on Thomas), and `H1` alone is near-chance — the signal is
in `H0`. 500+ per-seed runs sit under `results/` and `param_sweeps/`, each
row of `results/experiments.jsonl` carrying its producing commit hash via
`provenance.py`; `FROZEN.md` records a bit-reproducible frozen release of
the state behind the report.

## Layout

```
writeup/short_report.{tex,pdf}   the report (source of truth) + its 2 figures in figs/
scripts/                         the pipeline entry points (below)
src/cloudforger/                 the package the scripts wire together
configs/runs/{thomas,nested_thomas,classification}/   one YAML per process/method
results/                         per-seed results.json behind the tables above
param_sweeps/                    calibration + encoder sweeps (scripts/ + results/), separate on purpose
slurm/                           the sbatch scripts that fan the seed grids out on the cluster
figs/                            calibration_comparison.pdf, vectorization_schematic.pdf
tests/                           pytest suite (registry smoke + full-CLI e2e)
```

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
`*_vihrs.yaml`, `mincontrast.yaml`, `mincontrast_g.yaml`. Classification
configs live in `configs/runs/classification/` and read the per-`k` diagram
bundles directly (no `featurize.py` step).

The two report figures:

```bash
python scripts/make_calibration_comparison_figure.py   # -> figs/calibration_comparison.pdf
python scripts/make_vectorization_schematic.py          # -> figs/vectorization_schematic.pdf
```

Parameter sweeps (report §3.2–3.3) are driven from `param_sweeps/scripts/`
(`run_pi_calibration_sweep.py`, `featurize_sigma_sweep.py`) and land in
`param_sweeps/results/`.

Tests: `pytest tests/` (add `-m slow` for the ~1 min end-to-end CLI test).
