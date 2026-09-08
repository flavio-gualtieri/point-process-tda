# ParamNet — topological feature learning for spatial point processes

Learned topological features (persistence images of DTM-filtration diagrams)
for **likelihood-free inference on spatial point processes**: estimating the
latent cluster parameters of a Neyman–Scott process, and discriminating
between process families, from a *single* bounded-window realization.

**Headline results** (all paired seed-by-seed against the baselines on
identical clouds and splits, Wilcoxon signed-rank):

- **4-way process classification: 0.880 ± 0.002 vs 0.837 ± 0.005** for the
  neural radial-summary baseline (Vihrs 2022) — +4.2 points, `p = 0.002`,
  `n = 10` seeds, and only −0.3 points on an out-of-design adversarial split.
- **Two-level nested Thomas: 0.198 ± 0.008 vs 0.207 ± 0.006** normalised
  loss, winning on 5/5 shared seeds — the regime where no closed-form summary
  statistic exists, so classical estimators cannot be run at all.
- **Single-level Thomas: 0.111 ± 0.005 vs 0.112 ± 0.004**, a statistical tie
  (`p = 0.92`) once the topological features are fused with 8 columns of the
  cloud's own `L(r) − r` curve; persistence images alone lose (0.131).
- **~7× better than minimum contrast on Ripley's `K`**, which additionally
  fails to converge on 4/10 seeds near the Poisson limit.

Details, failure modes, negative results and exact provenance below.

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
on synthetic clouds from a reparametrized design distribution. The
contributions are the per-diagram coverage-quantile calibration of the
persistence images, application to the two-level nested Thomas process, the
`H0`-image ⊕ `L(r)` fusion below, an explicit ablation of the design choices,
a comparison of TDA vectorizations (image / landscape / silhouette / Betti
curve), and benchmarking against classical and neural (Vihrs 2022) baselines
on shared splits.

## Results

### 1. Parameter estimation

Normalised loss `L` (lower is better) on the shared test split. `L ≈ 1` is
the label-shuffle chance level, so it doubles as an error ceiling. The
`vihrs` baseline is the best-validation-checkpointed re-run at the same 10
seeds as ParamNet, i.e. a *stronger* baseline than the one tabulated in
`writeup/short_report.pdf` (0.115 / 0.217).

| Method | Thomas `L` | Nested Thomas `L` | `n` |
|---|---|---|---|
| ParamNet, PH only (fused DTM `k=5,10,15`) | 0.124 ± 0.007 | **0.198 ± 0.008** | 5 |
| **ParamNet + `L(r)` fusion** (DTM `k=5`) | **0.111 ± 0.005** | 0.207 ± 0.013 | 10 |
| vihrs (`L(r)−r` CNN, checkpointed) | 0.112 ± 0.004 | 0.209 ± 0.008 | 10 |
| Minimum contrast, `K` | 0.855 ± 0.682 | — | 10 |
| Minimum contrast, `g` | 6.199 ± 0.339 | — | 10 |
| *chance (label shuffle)* | *1.003 ± 0.019* | *1.006 ± 0.020* | — |

- **Nested Thomas: ParamNet wins.** 0.198 vs 0.207 on the 5 shared seeds
  (5/5 seed-wise wins, −4.7%), and wins on 3 of the 5 individual parameters —
  parent intensity (0.285 vs 0.304), meta-offspring (0.439 vs 0.478), mean
  offspring (0.091 vs 0.103) — losing on the two cluster-scale targets. No
  closed-form summary statistic exists for this process, so minimum contrast
  cannot be run at all. The margin against the report's non-checkpointed
  baseline is larger (0.192 vs 0.217, −12%); the number above is against the
  strengthened one.
- **Single-level Thomas: parity, reached by fusion.** Persistence images
  alone lose to the radial summary (0.131 vs 0.112, `p = 0.002`). Adding 8
  log-spaced samples of the cloud's own `L(r) − r` curve to the scalar side
  vector recovers it: 0.111 vs 0.112, `p = 0.92` — a tie, and a significant
  gain over PH-only (`p = 0.002`, `n = 10`). The gain is concentrated in the
  density/count targets (parent intensity 0.213 → 0.185, mean offspring
  0.140 → 0.118), exactly where a second-order statistic should help a
  density-based topological feature.
- **Both neural methods beat minimum contrast on `K` by ~7×** on average,
  and by ~3× on the 6/10 seeds where it converges (0.365–0.399); the
  remaining 4 seeds land at 0.903–2.056, which is the documented near-Poisson
  instability of contrast estimation (Waagepetersen & Guan, *JRSS-B* 71(3),
  2009), not a tuning failure. **The `g` row is not a fair comparison and
  should not be read as one:** every seed is worse than chance, traceable to
  a contrast exponent `c = 1` set by rule of thumb where `c = 1/4` is the
  value validated for aggregated patterns. It is reported for completeness.
- **Where it loses.** Fusion buys nothing on nested Thomas (0.218 vs 0.209,
  `p = 0.16`; PCA encodings of the curve break it outright). On Strauss —
  a Gibbs process with no closed-form summary — ParamNet is 0.558 ± 0.18 and
  fails to converge on 3/5 seeds against vihrs at 0.192; fusion has not yet
  been tested there. That is the weakest result in the project and is stated
  as such in the report's limitations.

### 2. Point-process model classification

4-way classification of the generating process family (Matérn cluster /
nested Thomas / Strauss / Thomas) from one realization. Identical clouds,
identical seeds, identical train/val/test split for every row — only the
feature set and the network differ. `adversarial` is a held-out split drawn
off the training design grid.

| Method | test acc | adversarial acc | `n` |
|---|---|---|---|
| **ParamNet + `L(r)` (PCA-2), DTM `k=5`** | **0.880 ± 0.002** | **0.877 ± 0.002** | 10 |
| ParamNet, PH only (fused DTM `k=5,10,15`) | 0.877 ± 0.004 | 0.872 ± 0.002 | 5 |
| ParamNet, PH only (DTM `k=5`) | 0.871 ± 0.003 | 0.869 ± 0.002 | 10 |
| vihrs (`L(r)−r` + `n(x)`, 1-D CNN) | 0.837 ± 0.005 | 0.835 ± 0.005 | 10 |

The fused row is the best number in the project, beats vihrs by 4.2 points
(`p = 0.002`) and PH-only by 0.8 points (`p = 0.002`), has the tightest
spread of any arm, and costs **1/3 of the persistence computation** of the
previous best configuration (one `k` instead of three). It loses only 0.3
points on the adversarial split.

Per-class recall (mean over seeds) — the topological features win or tie on
every class, and the margin is concentrated in exactly the two families a
second-order radial summary confuses:

| | Matérn cluster | nested Thomas | Strauss | Thomas |
|---|---|---|---|---|
| ParamNet + `L(r)` | **0.637** | **0.908** | 0.999 | **0.927** |
| vihrs | 0.573 | 0.889 | 0.999 | 0.863 |

Read per-class recall rather than the aggregate: `aniso_thomas` is folded
into `thomas`, so that class carries 2× the clouds, the split is
unstratified and the loss unweighted (majority-class accuracy is 0.40).
Both methods are affected identically.

Reproduce (fused-`k` PH-only and the vihrs baseline; the fusion arms are
`--set` overrides driven by `slurm/fusion_classification.sh`):

```bash
python scripts/train.py configs/runs/classification/pi_multik_h0h1_k5k10k15.yaml --seed 9371
python scripts/train.py configs/runs/classification/vihrs_lr_nx_k5k10k15.yaml    --seed 9371
python scripts/train.py configs/runs/classification/fusion_k5.yaml --seed 9371 \
    --run-tag fusion__pca2 --set method.params.include_lfunc=16 --set method.params.lfunc_pca=2
```

### 3. The fusion finding: the right `L` encoding is task-dependent

Sweeping how the `L(r)` curve enters the scalar side vector (5 arms × 10
seeds × 3 processes) gives a clean, interpretable split:

| arm | Thomas `L` | classification acc |
|---|---|---|
| PH only | 0.131 ± 0.007 | 0.871 ± 0.003 |
| 8 raw `L` columns | **0.111 ± 0.005** | 0.828 ± 0.039 (collapses on 3 seeds) |
| PCA-2 of the curve | 0.176 ± 0.003 | **0.880 ± 0.002** |
| PCA-4 of the curve | 0.119 ± 0.004 | 0.868 ± 0.040 |

Classification needs only the 2 leading directions (clustering amplitude and
scale); the remaining directions of a severely collinear curve (correlation
condition number ~2.8 × 10⁴) are estimation noise the cross-entropy head
overfits. Continuous parameter estimation needs the finer curve shape, so
truncating to 2 components discards signal and the raw columns win. Same
features, opposite prescriptions — evidence that the topological and radial
features carry non-redundant information rather than a tuning artifact.

### 4. Ablations and negative results

`writeup/short_report.pdf` Table 4 ablates filtration (Rips vs DTM vs fused
multi-`k`), homology degree (`H0` / `H1`), encoder weight sharing,
persistence-image calibration, and four TDA vectorizations (image /
landscape / silhouette / Betti curve), plus the `param_sweeps/` grids. Two
load-bearing findings: removing topology collapses the estimator to chance
(`L` 0.125 → 0.893 on Thomas), and `H1` alone is near-chance — the signal is
in `H0`.

Two design ideas were tested at `n = 10` and **rejected on their own
evidence**, which is recorded rather than dropped:

- *Separate encoders per homology dimension* is consistently harmful
  (nested Thomas 0.207 → 0.215, `p = 0.002`, degrading monotonically with
  capacity). Early mixing of the `H0`/`H1` channels is doing real work.
- *Reparameterizing the filtration by the cloud's own `L`* confirms its
  predicted mechanism (the transform alone is strongly harmful, `p = 0.002`,
  because the second-order trend is divided out) and is significant
  conditional on `L` being fed back (`p = 0.049` on nested Thomas), but
  yields no net gain over plain DTM. Demoted to an ablation row.

500+ per-seed runs sit under `results/` and `param_sweeps/`, each row of
`results/experiments.jsonl` carrying its producing commit hash via
`provenance.py`; `FROZEN.md` records a bit-reproducible frozen release of the
state behind the report.

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
bundles directly (no `featurize.py` step). The `L(r)` fusion, dimension-split
and `L`-reparameterization sweeps are `{fusion,dimsplit,lfunc}_k5.yaml` in
each process folder, driven by the matching `slurm/*.sh`; aggregate them with

```bash
python scripts/collect_fusion_results.py   # paired Wilcoxon + seed-collapse detector
```

The two report figures:

```bash
python scripts/make_calibration_comparison_figure.py   # -> figs/calibration_comparison.pdf
python scripts/make_vectorization_schematic.py          # -> figs/vectorization_schematic.pdf
```

Parameter sweeps (report §3.2–3.3) are driven from `param_sweeps/scripts/`
(`run_pi_calibration_sweep.py`, `featurize_sigma_sweep.py`) and land in
`param_sweeps/results/`.

Tests: `pytest tests/` (add `-m slow` for the ~1 min end-to-end CLI test).

## License

MIT — see [LICENSE](LICENSE).
