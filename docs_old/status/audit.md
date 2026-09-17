# Status audit — 2026-09-11

Read-only audit of `point-process-tda` at HEAD `6d5e0f3` (branch `frontend`, working tree clean).
Nothing was trained. Companion workbook: `docs/status/experiment_matrix.xlsx` (Overview, Params
detail, Classification detail, To run, Runs, Run groups, Keys).

Audience: me and future Claude sessions. Advisors should read the workbook, not this file.

---

## 0. TL;DR

1. **The ablation matrix is essentially empty on the current data.** Of 54 cells (24 persistence
   plus 3 classical, for each of the two tasks): **1 complete, 1 partial, 16 fused-only, 34 missing,
   2 not meaningful**. No persistence-homology cell has a single-component run on the current
   (2026-09-02 regenerated) data. Every current PH run fuses H0+H1, and most also fuse k=5,10,15.
2. **Every Betti-curve and landscape result predates the data regeneration.** All of them ran on
   Aug 11–21 clouds. The regen replaced those clouds on 2026-09-02, and commit `6799111` deleted the
   results. What survives: 600 `results.json` in git tag `legacy-pre-kernel`, plus SLURM logs for
   arms that were never committed. That includes every single-component Betti-curve run
   (`betti_cnn` sweep). None of it is comparable with current runs (different clouds, different
   split sizes).
3. **Betti curves and landscapes cannot run classification at all.** `betti_cnn.py:260`,
   `betti_multik.py:345` and `vectorized_multik.py:553` hard-code `MSELoss` plus
   `ParameterEstimator`. 16 of the 34 missing cells need code before they can run.
4. **PI on Rips H0 is degenerate by construction.** All births are 0, so `axis_bounds` raises
   (`calibration/diagram_calibration.py:115-116`). All 20 legacy attempts crashed. This is
   "not meaningful", not "missing".
5. **The "adversarial" split is a second i.i.d. test set, not out-of-distribution.** It is 12.5% of
   the sampled parameter vectors, held out uniformly at random from the same prior
   (`data_generation/design.py:127-152`; configs use `mode: random, reps: 1, fraction: 0.125`).
   README calls it "out-of-design". That is wrong.
6. **Several on-disk results cannot be reproduced from the current code.**
   - The `fgp__PH*` and `fgfus__*` runs used per-column F/G z-scoring, since fixed at
     `pi_multik.py:520-546`.
   - 1,089 of 1,090 current results were written from dirty trees.
   - `slurm/` is gitignored.
   - Training is not deterministic across jobs. Strauss `restart__r0` was advertised as reproducing
     the default run, but differs on 4/5 seeds, by up to 0.136.
7. **README Table 1 mixes dataset versions.** The min-contrast (0.855 / 6.199) and label-shuffle
   (1.003) rows are pre-regen. The neural rows are post-regen. The writeup's ablation table
   (`tab:branches`) is entirely pre-regen.

---

## 1. Codebase map (task 1)

### 1.1 Data generators

The pipeline is `scripts/generate.py`, which calls `CloudDesign.build(...).generate()`
(`data_generation/design.py:188-261`). Parameters are drawn in random mode: 8,000 vectors, one cloud
each (`reps: 1`), on the unit square. Of these, 1,000 are held out as "adversarial" and 7,000 are
train/val/test. Cloud seeds are `base_seed + i`; adversarial clouds use `base_seed + 100000 + i`.
All processes use `process.seed: 0`.

| process | class / file | sampled axes (scale) | derived targets | constraint | target set (loss) |
|---|---|---|---|---|---|
| thomas | `point_processes/thomas.py` (GaussianKernel) | K∈[15,120] log; EN∈[150,800] log; c∈[0.10,0.90] log | κ=K, μ=EN/K, σ=c/(2√K) | μ≥2.5 | parent_intensity, mean_offspring, cluster_scale |
| matern_cluster | `matern_cluster.py` (BallKernel) | same as thomas | κ, μ, R=c/√K | μ≥2.5 | parent_intensity, mean_offspring, cluster_radius |
| nested_thomas | `nested_thomas.py` (2-level NS) | K∈[15,120]; μ1∈[1.5,7]; EN∈[150,800]; c1∈[.10,.90]; c2∈[.03,.35], all log | κ, meta_offspring=μ1, μ=EN/(Kμ1), meta_σ=c1/(2√K), σ=c2/(2√(Kμ1)) | μ≥2.5; 3 ≤ meta_σ/σ ≤ 12 | 5 targets |
| aniso_thomas | `aniso_thomas.py` (AnisotropicGaussianKernel) | thomas axes + aspect∈[1,4] lin + θ∈[0,π) lin | + cluster_aspect, cluster_theta | μ≥2.5 | 5 targets incl. θ |
| strauss | `gibbs.py` StraussProcess (birth–death MH) | β∈[200,900] log; γ∈[0.05,0.95] lin; R∈[0.005,0.05] lin | — | — | beta, gamma, radius |
| lgcp / lgcp_strauss | `cox.py`, `gibbs.py` | μ, σ², s (+γ, R) | — | μ+σ²/2 ≤ 7 | clouds only: no diagrams, no PH configs, no results |

Notes:

- **Edge buffers.** Neyman–Scott edge buffers now come from `Kernel.support_radius`
  (`kernels.py:73-78`). For the Gaussian kernel with ε=1e-4 that is ≈3.72σ, replacing the old
  hard-coded 4σ. This is why pre- and post-regen clouds differ.
- **Realized point counts.** They run 51–1,291 per cloud (thomas/matern manifests), even though the
  design bounds are on *expected* counts [150, 800].
- **thomas and matern_cluster are seed-matched twins.** Same seed, same (K, EN, c) ranges, same
  constraint, so the accept/reject sequence is identical. I checked the manifests: the adversarial
  indices are identical and 1000/1000 adversarial vectors match exactly. Cloud seeds are also
  identical. Parents come from the same RNG stream in buffers of different size (`neyman_scott.py:94-97`),
  so each twin pair has the same parameters and nearly aligned parent layouts. aniso_thomas shares
  the same first draw and the same adversarial indices, but diverges after that (two extra axes).
- **Strauss mixing is unverified.** Strauss is simulated by birth–death MH from a Poisson(β) start
  (`gibbs.py:57-61`). Step count is `max(15000, 40·β·|W|)`, about 40 sweeps (`gibbs.py:96-100`). The
  docstring says this is "ample for the weak Strauss interaction here" and suggests Vihrs' 100k steps
  "for strong interaction / publication". But the design includes strong inhibition
  (γ down to 0.05, R up to 0.05, β up to 900). No convergence diagnostic is saved anywhere.

### 1.2 Filtrations and diagrams

- `rips.py:31-36` uses ripser. H0 has all births 0 and one death=∞ bar.
- `dtm.py:62-82` uses gudhi `DTMRipsComplex(k, q=2)`. Vertex value = DTM, so H0 births are > 0.
- Diagrams are computed shard-parallel by `scripts/processing/diagrams_shard.py` and
  `slurm/diagrams_{compute,merge}*.sh`.
- Bundles on disk: `data/<process>/{rips,dtm_k5,dtm_k10,dtm_k15}/{,adversarial_}diagrams.pkl`, for
  thomas, matern, nested, aniso and strauss, plus `data/classification/…`. Also
  `l_dtm_k5` (L-reparameterized) exists for thomas, nested and classification.
- **Rips diagrams exist for every process, but nothing on the current data has ever been trained on
  them.**
- On disk I checked 500 thomas clouds. Rips H0 finite births are exactly 0, with exactly one ∞ bar
  per diagram. DTM-k5 H0 births lie in [0.0022, 0.90], also with one ∞ bar. Both have n−1 finite H0
  points (121–1,057 per diagram; median 356).

### 1.3 Classical summary statistics (vihrs baseline and PH side-channels)

| stat | estimator | edge correction | r-grid | normalization | file:line |
|---|---|---|---|---|---|
| L(r)−r | Ripley K, area/(n(n−1)) normalization | Ripley isotropic (closed-form rectangle arcs) | `linspace(0, 0.25, 513)` | one pooled z-score over all train clouds × all r, per channel (`fit_zscore_per_channel`) | `baselines/vihrs.py:149-226`, `:381-407` |
| G | nearest-neighbour distance CDF | reduced-sample (border) | `linspace(0, fg_r_max=0.25, 513)`; 0.08 tested once | pooled per channel (as L) in vihrs | `baselines/summstats.py:78-132` |
| F | empty-space CDF on a 64×64 interior test grid | reduced-sample | same as G | same | `summstats.py:135-155` |
| J | (1−G)/(1−F), held flat once 1−F<0.05, clipped at 5 | inherits | same | same | `summstats.py:158-184` |
| n(x) | point count | — | — | log then z-score (train) | `vihrs.py:690` |

Both F and G enforce monotonicity with `np.maximum.accumulate` (`summstats.py:117`).

Inside `pi_multik`, L/F/G enter in one of two ways:

- **Side-vector columns.**
  - L: `include_lfunc` log-spaced radii from 0.01 to 0.25, interpolated from the cached curve
    (`pi_multik.py:104-143`). Each column is z-scored separately (`:514-518`), or projected with a
    train-fit PCA (`:252-282`).
  - F/G: `include_fgfunc` radii each (`:146-199`). Pooled z-score per function (`:520-546`). That
    pooling is the fix. The on-disk `fgp__PH*` / `fgfus__*` runs predate it.
- **Full curves.** Full curves go through a `CurveEncoder` that copies the vihrs trunk
  (`:576-603`, `curve_channels`).

Caches: `data/<proc>/{,adversarial_}clouds.lfunc_cache.npz`, `*.summ_fg0250_cache.npz`, and the
`*_classify_` variants.

### 1.4 PH vectorizations

| vectorization | construction | hyperparameters (current defaults) | calibration | file:line |
|---|---|---|---|---|
| Persistence image | birth × persistence box; Gaussian per point with persistence weight; exact per-pixel mass via erf; no clipping | 64×64, σ=0.5 px per axis, coverage q=0.95, pad α=1.05 | per seed, per k, per dim, on train rows only | `vectorization/persistence_images/persistence_image.py:34-109`; `calibrated.py:16-54`; `pi_multik.py:401-439`, `:870-873` |
| Betti curve | β_d(t)=#{b ≤ t < d}; ∞ bars dropped; one shared grid for all dims (needed for χ) | grid 128 on [0, 1.1·q99 of pooled finite deaths]; `include_euler: false` in configs | per seed, train rows | `scalar_features/betti_curve.py:59-99`; `scalar_features/calibrated.py:27-70`; `experiments/betti_cnn.py`, `pi_multik/betti_multik.py:87-158` |
| Landscape | tents Λ_i(t)=max(0, min(t−b, d−t)); λ_k = k-th largest per grid point | G=128; K chosen by coverage rule (λ_{K+1} ≤ 1% λ_1 for 99% of diagrams), capped at 16, reconciled by max over dims; q=0.95 in configs; grid [t_min, 1.05·q-quantile of max death] | per seed, per k, per dim | `vectorization/landscapes/{tent,landscape_silhouette,calibrated}.py`; `pi_multik/vectorized_multik.py:92-135` |
| (Silhouette, persistence statistics, EC) | legacy only; outside the matrix | p∈{0,1,2,3}; 2 channels per dim | | `vectorized_multik.py:138-240` |

### 1.5 Encoders and models

- **PI, `pi_multik`.** `PIMultiK` (`pi_multik.py:606-757`). Pipeline:
  - `EncoderBank` (`encoders/encoder_bank.py:14-91`): shared or independent per-k
    `CoordConvPIEncoder` (`encoders/coordconv_pi.py:11-55`). That encoder is conv 32/64/128, 3×3,
    2×2 pooling (avg in configs), dropout 0.2, global-average-pool, linear to 64.
  - Concat over k, or `ConvFusion` (`encoders/scaleconv_pi.py:30`).
  - Concat with the side vector.
  - MLP head [64, 32]: `ParameterEstimator` (`models/heads/paramest.py:4`) or `ClassificationHead`
    (`models/heads/classifier.py:7`).
- **Variants.** `pi_multik_dimsplit` (one encoder per homology dim); two-branch
  (`use_pi`, `curve_channels`).
- **Betti curves.** `betti_cnn` is 1 filtration × 1 dim, 1-D conv, dropout 0.3. `betti_multik` is
  per-k `SequenceEncoderBank` (`encoders/sequence_bank.py`: Conv1d 64 k7 pool2 ×2 + conv) followed by
  concat. Both are regression-only.
- **Landscapes.** `vec_multik`: the CoordConv encoder applied to the (K, G) raster (`vectorized_multik.py:324-334`).
  Regression-only.
- **vihrs.** `VihrsCNN` (`baselines/vihrs.py:414-448`): Conv1d(C,64,7)-pool5 ×2 + Conv1d, ⊕ n(x),
  Dense 64-32-out.

### 1.6 Training entry points, configs, budgets

- **Entry point.** `scripts/train.py <config.yaml> --seed S [--run-tag T] [--set k=v] [--force]`
  (`main` at `:436`). It dispatches to `run_experiment_method` (PH),
  `run_vihrs_method` / `run_vihrs_classify_method` (`:206-358`), or classical baselines.
- **Other scripts.** `generate.py`, `featurize.py` (legacy PI precompute; `pi_multik` does not use
  it), `build_classification_data.py`, `collect_{fusion_results,restarts,twobranch}.py`.
- **Configs.** `configs/runs/<process>/*.yaml`. Many thomas/nested configs are legacy-era
  (`betti_cnn_*`, `vec_multik_*`, `pi_multik_k{5,10,15}`, `pi_multik_rips`). The current sweeps are
  driven by `slurm/*.sh` through `--set` overrides. **`slurm/` is gitignored** (`.gitignore:470`), so
  the exact commands behind current results live only on purgeable scratch.
- **Budgets.** Checkpoint selection is best validation loss throughout.

| family | epochs / patience | batch | optimizer | notes |
|---|---|---|---|---|
| PH params | 500 / 50 | 32 | AdamW lr 1e-3, wd 3e-4 | |
| vihrs params | 500 / 30 | 32 | Adam lr 1e-3, no wd | patience comes from the config; `results.json` does **not** record it (`vihrs.py:772-788`) |
| classification (older arms) | 200 / 30 | 128 | | `fusion__*`, `dimsplit__*`, `lfunc__*`, default k5+10+15, default vihrs |
| classification (newer arms) | 600 / 60 | 128 | | `cap600__*`, `summ__*`, `fgfus__*`, `tb__*` |

- **Split.** `train_val_test_indices(n, seed)` is `randperm(n, Generator(seed))` sliced 70/15/15
  (`core/splits.py:29-43`). The permutation depends on n.

### 1.7 The "adversarial" held-out set

`split_adversarial_vectors` (`design.py:127-152`) picks `round(0.125·8000) = 1000` indices
uniformly at random, without replacement, from the 8,000 sampled vectors. The RNG is
`default_rng(seed + 100000)`. Those vectors get new cloud seeds (`design.py:245-248`).

Because `reps: 1`, ordinary test clouds already have parameter vectors never seen in training. So
the adversarial set is a second i.i.d. test set with a different seed range. It is not off-grid and
not out-of-design.

For classification, `build_classification_data.py` concatenates each source directory's adversarial
file. That gives 1000/1000/1000/2000 (thomas + aniso).

`README.md:12-13` and `:103-104` ("out-of-design adversarial split", "drawn off the training design
grid") overstate this.

### 1.8 Where results live

| location | what | notes |
|---|---|---|
| `results/<proc>/<ftag>/<method>/[_runs/<tag>/]seed_<s>/{results.json,results.pt,model.pt}` | 1,090 current seed-runs (2026-09-02 → 09-10) | layout from `paths.py:121-141`; tracked in git |
| `results/experiments.jsonl` | 1,100-row append-only ledger, post-regen only | includes rows for overwritten runs (5-class classification) |
| `param_sweeps/results/**/results.json` | 155 legacy calibration/encoder-sweep JSONs, no `.pt` | pre-regen |
| git tag `legacy-pre-kernel` | 755 legacy `results.json` (600 under `results/`) | deleted from the tree by `6799111`; recover with `git archive legacy-pre-kernel <paths>` |
| `logs/*.out, *.err` | 5,010 SLURM logs | the only record of `betti_cnn`, `betti_multik` single-scale and untagged runs, landscape single-scale/untagged runs, silhouette-native runs, and 5-class classification |
| SLURM accounting (`sacct`) | per-task wall times | the logs contain no timings |
| external `legacy-pre-kernel-data.tar.zst` (QMUL OneDrive, per `FROZEN.md`) | exact legacy `data/` + `results/` | not accessible from here; `FROZEN.md` still has `<PASTE SHARE LINK>` |
| wandb / mlflow / tensorboard | none | grep finds no usage |

---

## 2. H0 under Rips vs DTM (task 2)

### Common to all vectorizations: the infinite bar is dropped

`PersistenceDiagram.finite_pairs` removes any row with a non-finite entry
(`core/diagram.py:24-30`). The PI, landscape and silhouette transforms call it
(`persistence_image.py:94`, `landscape_silhouette.py:43,80`). So do all calibrators:
`diagram_stats` (`diagram_calibration.py:57-63`), `axis_bounds_1d` (`:161-166`), `choose_K`
(`landscapes/calibrated.py:71`), and `_shared_grid_hi` (`scalar_features/calibrated.py:40`).

`BettiCurve` reads raw pairs but uses `drop_infinite=True` (`betti_curve.py:73-76`), forced at
`scalar_features/calibrated.py:67`.

So the essential H0 class is **dropped, never truncated or capped**. Under Rips it carries nothing
(birth 0, death ∞). Under DTM it would carry the global minimum DTM value. That information is
dropped, but it is redundant with the other births.

### Persistence image

- Calibration (`axis_bounds`, `diagram_calibration.py:106-117`) summarizes each train diagram by
  (min birth, max birth, max persistence) (`:53-65`). It then sets:
  - `birth_lo` = the (100−q)-th percentile of per-diagram min births;
  - `birth_hi` = α × the q-th percentile of per-diagram max births;
  - `pers_hi` = α × the q-th percentile of per-diagram max persistence.

  Current values are q=0.95 and α=1.05 (`pi_multik.py:872-873`).
- **Rips H0.** Every birth is 0, so `birth_lo = birth_hi = 0`, and `birth_hi <= birth_lo` raises
  `ValueError("H0 birth axis is degenerate; use a 1-D vectorizer")` (`:115-116`). Any config with
  Rips and `homology_dims` containing 0 crashes at the first seed.
  - Evidence: the legacy `pi_multik_rips_{thomas,nested_thomas}` jobs failed 20/20
    (`logs/pi_multik_rips_*_2376908[9|90]_*.err`, traceback ending at line 116).
  - The old fallback that fabricated a birth range (`degenerate_birth_frac`) is dead code inside a
    string literal (`:93-103`).
  - The writeup describes a "fallback to 1-D vectorization"; that fallback does not exist.
- **Rips H1.** Births are > 0 (edge lengths), so it is well-defined. It has never been run.
- **DTM H0.** Births are DTM values, so the birth axis is well-defined. Points outside the calibrated
  box are **not clipped or clamped**: their Gaussian mass outside the pixel edges is lost
  (`_box_mass`, `persistence_image.py:25-31,102-106`). With q=0.95, by construction about 5% of
  train diagrams have their minimum birth below `birth_lo`. Up to about 5% exceed the upper birth or
  persistence edge; fewer in practice, because of the 1.05 pad.
- **Padding.** Pad is multiplicative on the *upper* edge only (`:113-114`), not on the span, so the
  effective margin depends on distance from 0.

### Betti curve

- The grid is `linspace(0, α·q99(finite deaths pooled across H0 and H1 and diagrams), 128)`, with
  α=1.1 (`scalar_features/calibrated.py:27-45, 62-70`). A feature counts as alive when
  `b ≤ t < d` (`betti_curve.py:86-88`).
- **Rips H0.** β₀(0) = n−1 (the ∞ bar is dropped). It then decreases as MST edges appear:
  β₀(t) = (n−1) − #{MST edges ≤ t}. So it is the MST edge-length survival function scaled by n.
  It encodes n(x) directly, and n(x) is also fed as a side input. Well-defined; the degeneracy is
  harmless here.
- **DTM H0.** Births are > 0, so β₀ rises from 0 as vertices enter, then falls as components merge.
  The grid still starts at 0, so the segment [0, min DTM) is always zero. That wastes resolution but
  is harmless.
- **One grid shared by H0 and H1.** Whichever dimension has the larger deaths sets the upper end.

### Landscape (and silhouette)

- The grid is `[t_min, T]` from `axis_bounds_1d` (`diagram_calibration.py:136-179`):
  - `t_min` = the (100−q)-th percentile of per-diagram min birth;
  - `T` = α × the q-th percentile of per-diagram max finite death.

  There is no birth axis, so nothing can be degenerate.
- **Rips H0.** `t_min` = 0 exactly. Every tent starts at t=0 and peaks at d_i/2.
  λ_k(t) = t while at least k deaths exceed 2t. So λ₁…λ_K encode only the **K largest** MST edges.
  - Saved legacy runs record `t_min=0.0, K=16` for Rips H0
    (`git:legacy-pre-kernel:results/thomas/rips/vec_multik_landscape_native/_runs/calib_q100/*`).
  - K is capped at 16 (`landscapes/calibrated.py:40-85`), but the H0 diagram has about 350 points.
    So the landscape discards the bulk of the H0 distribution **under both Rips and DTM**. Every
    saved data-driven landscape run chose K=16, the cap, for both dims (490 per-k/per-dim entries in
    the legacy JSONs). K=4 or 8 appears only in the explicit K-override sweep arms.
- **DTM H0.** `t_min` > 0 is fitted. Otherwise the same top-16 truncation applies.
- **Silhouettes.** Weighted mean of all tents (`tent.py:60-92`); no truncation.

### Summary

| | PI | Betti curve | Landscape |
|---|---|---|---|
| Rips H0 | **raises**: birth axis has zero width; not meaningful | β₀ = MST-edge survival × n; valid | t_min=0; top-16 MST edges only; valid |
| DTM H0 | valid; births > 0; mass outside the q=0.95 box is lost | valid; [0, min DTM) wasted | valid; top-16 envelope |
| ∞ bar | dropped | dropped | dropped |

---

## 3. Run inventory (task 3)

### Sources

2,645 seed-runs, grouped into 343 groups (one results directory or log arm each) and 54
comparability keys.

| era | source | seed-runs | notes |
|---|---|---|---|
| current | `results/**/results.json` | 1,090 | includes 5 one-epoch cache-warm runs (`summ__cachewarm_*`); not results |
| legacy | git `legacy-pre-kernel` | 600 | full config and commit |
| legacy | `param_sweeps/results` (working tree) | 155 | JSON copies of legacy sweeps |
| legacy | SLURM logs only | 790 | includes failed runs and earlier-job duplicates (latest job kept); metric at 4 d.p.; no commit recorded |
| superseded | SLURM logs (5-class classification, overwritten by `--force`) | 10 | |

Recorded per run in the Runs sheet: task, family, filtration(s), dims, vectorization or statistic,
encoding, fused flag, process, dataset version, split, seed, metric, adversarial metric, path, key,
commit, and timestamp.

### Comparability key

Key = (dataset version, split, architecture including the side vector, training budget, metric
definition). The Keys sheet lists all 54.

- **Dataset versions.**
  - DV1: params processes, regenerated 2026-09-02.
  - DV2: classification bundle 2026-09-03 (4 classes, aniso folded into thomas).
  - DV2a: the first 5-class bundle, overwritten.
  - DV0: pre-regen, Aug 2026.
- **Split.** n=7000 or 35000 for current runs. For legacy DTM arms, n=6995–6998, because clouds with
  failed diagrams were dropped. n differs by process and by k-set: nested fused-k 6995 vs single-k
  6996. Legacy Rips and vihrs used n=7000. Since the permutation depends on n, **legacy
  PH-vs-vihrs "paired" seeds never shared a test set.**
- **Pooling rule.** Pool seeds only within one results directory. When two directories hold the same
  config (for example `fusion__Loff`, `fgp__Loff` and `dimsplit__shared_e64`), show one and list the
  others as alternates. Never average them: they reuse the same seeds.

### Headline current-data groups (all from the workbook; mean ± SD, n)

**Parameter estimation**

| config | thomas | nested | notes |
|---|---|---|---|
| PI k5 H0+H1 + log n | 0.1300±0.0075 (10) | 0.2081±0.0072 (10) | matern 0.1454±0.0091 (10) |
| PI k5+10+15 H0+H1 + log n | 0.1242±0.0073 (5) | 0.1977±0.0089 (5) | 5 processes, n=5 each; strauss 0.5581±0.2043 (bimodal), aniso 0.4500 |
| PI k5+10+15 H0 + log n | 0.1340±0.0129 (5) | 0.2018±0.0045 (5) | |
| PI + 8 raw L cols | 0.1110±0.0049 (10) | 0.2183±0.0136 (10) | |
| vihrs L | 0.1120±0.0037 (10) | 0.2091±0.0079 (10) | strauss 0.1931 (10), matern 0.1246 (10, `fgp__vL`), aniso 0.4381 (5) |
| vihrs L+F+G | 0.1065±0.0038 (10) | 0.1983±0.0078 (10) | matern 0.1142 |
| two-branch curves vs PH+curves, r0 | 0.1089 vs 0.1082 | 0.1983 vs 0.2022 | val-selected over 4 restarts: 0.1092 vs 0.1074 (thomas); 0.1995 vs 0.1993 (nested) |

**Classification (4-class accuracy)**

| config | accuracy | notes |
|---|---|---|
| vihrs L (600 ep) | 0.8371±0.0044 (`summ__L`) | |
| vihrs L+F+G | 0.8838±0.0036 | |
| PI k5 (600 ep, no n(x)) | 0.8729±0.0030 | |
| PI + L PCA-2 | 0.8806±0.0029 | |
| PI + L + F/G (pre-fix) | 0.8896±0.0114 | |
| two-branch, val-selected | curves 0.8841±0.0046 vs PH+curves 0.8836±0.0036 | |

All of these are fused configs. None fills a matrix cell.

---

## 4. Matrix status

| task | complete | partial | fused-only | missing | not meaningful |
|---|---|---|---|---|---|
| Parameter estimation (27) | 0 | 1 (L: aniso 5/10) | 8 (PI DTM ×6, F, G) | 17 (PI Rips H1, 8 BC, 8 LS) | 1 (PI Rips H0) |
| Classification (27) | 1 (L) | 0 | 8 (PI DTM ×6, F, G) | 17 (PI Rips H1, 8 BC, 8 LS; the 16 BC/LS need code) | 1 (PI Rips H0) |
| **Total (54)** | **1** | **1** | **16** | **34** | **2** |

**What it costs to close.** The To-run sheet sums median wall time per run from sacct for the closest
existing job family:

- Params: about 1,255 runs, ≈17 GPU-h.
- Classification: 250 runs, ≈4 GPU-h for the 9 runnable cells. The 16 BC/LS classification cells
  need a `task=classify` path first.

Every per-seed run is 1–4 minutes, so compute is not the bottleneck; code and config are.

---

## 5. Inconsistencies, suspicious choices and likely bugs

Ordered by how much they affect claims.

1. **No single-component ablation exists on current data.** This matters for the paper. The only
   current homology-dimension ablation is fused-k H0-only (n=5); there is no H1-only run at all.
   The only filtration ablation is k5 vs k5+10+15 (both H0+H1). The vectorization ablation (BC, LS)
   exists only on DV0.
2. **`log_label_names` is dead config.** `pi_multik.py:939-940` and `vihrs.py:689` call
   `fit_log_zscore` on **every** target, and results store `label_transforms = ["log"] × n`. Two
   consequences:
   - Strauss γ ∈ [0.05, 0.95] and R are log-transformed, although `strauss_*.yaml:76` lists only
     `beta`.
   - aniso `cluster_theta` ∈ [0, π) is log-transformed and z-scored. It is a circular variable
     (θ≈0 and θ≈π are the same orientation), and log(θ→0) is heavy-tailed. Its per-target MSE is
     near 1 for every method, and it is 1 of the 5 terms averaged into aniso's headline loss.
   - Only `experiments/base.py:464` honours the config.

   This is consistent between PH and vihrs, so it is fair, but the configs misstate it.
3. **Pre-fix F/G scaling results are on disk.** `fgp__PHfg`, `fgp__PHLfg`, `fgfus__PHfg`,
   `fgfus__PHLfg` and `fgfus__PHLpca2fg` were produced before the pooled z-score fix
   (`pi_multik.py:520-546`). The comment there says per-column scaling "made PH (+) F/G train worse
   than PH alone on every process". The current code cannot reproduce these numbers. The
   0.8896 "best number in the project" is one of them. The workbook flags each row.
4. **PH and vihrs classification comparisons are not input-matched.**
   - PH classification arms default to `include_log_n=false` (`pi_multik.py:839`;
     `configs/runs/classification/fusion_k5.yaml`). vihrs always feeds n(x).
   - So `cap600__*` and `fgfus__*` vs `summ__*` differ in n(x) as well as features. Only the
     `tb__*` two-branch runs turn n(x) on for both.
   - Optimizer and budget also differ: AdamW with wd 3e-4 and patience 50 vs Adam with no wd and
     patience 30 (params).
5. **"Adversarial" is i.i.d.** See §1.7. Any robustness claim built on it is unsupported.
6. **Training is not deterministic across jobs.**
   - Identical configs rerun under different tags differ per seed:
     - thomas PH k5 (`fusion__Loff` vs `fgp__Loff`): up to 0.0059, 5/10 identical.
     - nested: up to 0.0086.
     - vihrs thomas (default vs `fgp__vL`): 0/10 identical, up to 0.0048.
     - classification `summ__L` vs `cap600__vihrs`: up to 0.0091.
   - Strauss `restart__r0` was advertised in `slurm/strauss_restarts.sh` as reproducing the default
     run exactly. It differs on 4/5 seeds, by up to 0.136.

   Paired Wilcoxon tests across arms run in different jobs therefore carry this noise. The README
   and TODAY.MD treat it as zero.
7. **Classification epoch cap.** Arms at 200/30 include the default k5+10+15, `fusion__*`,
   `dimsplit__*`, `lfunc__*` and default vihrs. None of them record `n_epochs_run` in `results.json`
   (the field came later), so truncation cannot be checked from saved results. Even at 600/60,
   `cap600__raw8` hits the cap on 2/10 seeds (`n_epochs_run` = 600).
8. **Landscape details.**
   - K is capped at 16 and reconciled across dims but not across k. Multi-k H1-only landscapes
     crashed 10/10 (`ValueError: all input arrays must have the same shape`;
     `vectorized_multik.py:133-135` stacks per-k tensors with different K). This is a likely cause,
     consistent with `calibrated.py:143` reconciling only over dims.
   - The top-16 truncation discards most H0 features (§2).
   - Legacy Rips landscape runs used coverage q=1.00 (`calib_q100`); the DTM rows used 0.95.
     `tab:branches` compares them as a filtration ablation, but calibration is confounded.
9. **Unstable single-component legacy PI.** DTM k5 H0-only gives thomas 0.2038±0.2350 and nested
   0.3513±0.3043, n=10 (`git:legacy-pre-kernel:results/*/dtm_k5/pi_multik/_runs/h0only`). These are
   bimodal collapses. Single-component PI runs will likely need restarts and val-selection, as
   Strauss did.
10. **Stale-cache hazard.** `vihrs.get_features` (`:296-337`) validates caches only by r-grid and
    keys, not by cloud content. After any regeneration with the same cloud count, a stale cache is
    silently reused; `vihrs_classify_rerun.sh` had to force a recompute. Current caches post-date the
    regen (checked by mtime), so nothing is contaminated today. lgcp and new processes are exposed.
11. **Latent fallback bug.** The `_classification_n_points` fallback decodes `seed // seed_offset`
    as the *class* index (`pi_multik.py:72-80`), but the builder offsets by *source-directory* index
    (`build_classification_data.py:143-150`). With aniso merged into thomas these differ. Not
    triggered today, because current bundles have unique seeds.
12. **Provenance gaps.**
    - 1,089 of 1,090 current `results.json` have `git_dirty: true`, at commits 6799111, 61354c5,
      9cc6c18 or 5d91f03. The exception is `results/thomas/dtm_k5+10+15/pi_multik/seed_9371`.
    - vihrs JSONs omit `summary_channels` and early-stopping patience; the feature set is inferred
      from the run tag.
    - `slurm/` is untracked.
    - Several run groups mix commits within one directory: default vihrs for
      thomas/nested/strauss/classification (09-02/03 plus the 09-07/09 top-ups), and thomas
      k5+10+15 PI (clean plus dirty).
    - Everything, including `cloud-env`, lives on purgeable `/gpfs/scratch`.
13. **Strauss MH convergence is unverified.** See §1.1. The Strauss "failure" story (restarts,
    γ-gap) presumes equilibrium samples.
14. **thomas and matern_cluster are seed-matched twins** (§1.1). The classification split is random
    over all 35k clouds, so twin clouds (same κ, μ, c; same seed; nearly aligned parents; different
    kernel) land in different splits. That is not label leakage. But it is a dependence between
    train and test worth disclosing, because the matern-vs-thomas confusion is the whole difficulty
    of the task.
15. **Imbalance.** Classes are imbalanced (thomas 40% after the merge) and the split is
    unstratified. The loss is unweighted. Overall accuracy partly measures the prior; per-class
    recall is in the workbook.
16. **Docs vs code.**
    - README Table 1 mixes DV0 (min-contrast, label-shuffle) with DV1 (neural).
    - README says "each row of `results/experiments.jsonl` carrying its producing commit"; the
      pre-regen ledger was replaced, and every current row is dirty.
    - `tab:branches` is all DV0, with its own split-size issue.
    - The writeup's claimed Rips-H0 "1-D fallback" does not exist.
    - `FROZEN.md` has no share link.
    - The two-level-calibration prose inconsistency (writeup `ssec:calibration`, q=0.99) is noted in
      memory as still unfixed.

---

## 6. What a reviewer would attack

- *"Where is the ablation?"* Every PH number fuses dims (and usually k). The claim "the signal is in
  H0" rests on DV0 runs (H1-only 0.90/0.93) and on a bimodal DV0 H0-only single-k run.
- *"The strongest baseline wins."* On current data, vihrs L+F+G has the lowest parameter-estimation
  loss of any configuration on thomas (0.1065) and matern (0.1142). On nested it ties the 5-seed
  fused-k PH run (0.1983±0.0078, n=10 vs 0.1977±0.0089, n=5). On classification, the clean
  two-branch ablation shows no PH gain: 0.8836 vs 0.8841 (val-selected). The one higher number
  (0.8896) comes from pre-fix code and a coarser model.
- *"Is 'adversarial' out-of-distribution?"* No (§1.7).
- *"Seeds are paired, but are runs deterministic?"* No (§5.6).
- *"Unequal inputs."* PH classification arms lack n(x); vihrs has it. Budgets and optimizers differ.
- *"Rips was 'ablated' at a different calibration."* Legacy Rips landscape at q=1.00 vs DTM at 0.95.
- *"Strauss samples may not be at equilibrium."* ~40 MH sweeps from a Poisson(β) start, with
  γ down to 0.05.
- *"aniso θ is unlearnable by construction."* The DTM filtration is rotation-invariant, and θ is
  log-z-scored as a linear quantity. It drags the aniso average.
- *"Numbers you can't regenerate."* Dirty trees, untracked sbatch files, pre-fix F/G results, a
  mixed-version README table.

---

## 7. What the repo cannot tell me

- Which exact code produced any current result: every stamp is dirty, and the diffs were not saved.
- Whether the legacy log-only arms used `include_euler` true or false. Only today's config files say
  false; the logs don't record it. The legacy `betti_multik` fused-k run appears in two job arrays;
  I keep the later one.
- Whether Strauss MH chains reached equilibrium.
- Whether 200-epoch classification arms were truncated (no `n_epochs_run`). Memory says yes, from
  logs; I did not re-derive it.
- The external legacy archive contents (not accessible).
- The seed list and settings for the 1-epoch `summ__cachewarm_*` runs, beyond what their JSON shows.
  They are excluded as non-results.

---

## 8. How the workbook was built

The workflow was read-only and ran in the session scratchpad. None of it is in the repo.

1. Parse every `results.json` (working tree and `param_sweeps/`).
2. `git archive legacy-pre-kernel` the legacy JSONs into scratch.
3. Parse all `logs/*.out`: per-seed test and adversarial loss, failures, `Command:` lines.
4. Query `sacct` for wall times. This is SLURM accounting, not compute.
5. Normalize every run into semantic fields and a comparability key. Dedupe legacy log arms that ran
   twice (latest job per seed). Group by results directory.
6. Aggregate: mean and sample SD (ddof=1) over seeds within a group.
7. Compute derived val-selected restart numbers only where noted: per seed, keep the restart with the
   lowest saved `best_val_loss`.

openpyxl came from a scratch-local `pip --target` install. No environment or repo file was modified.
Re-running needs the same steps. Ask for the scripts if they should be committed under
`docs/status/`.
