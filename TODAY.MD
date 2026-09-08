# Project status — 2026-09-07

Branch `frontend`. Target: AISTATS submission, ~3 weeks out. All work below is
uncommitted.

---

## TL;DR

Three ideas were tested today. Two were rejected with clean negative evidence;
one produced **two publishable wins** that address the paper's two most-cited
weaknesses:

| finding | before | after | paired p |
|---|---|---|---|
| **Thomas parameter estimation** — PH ⊕ raw L(r) columns | PH 0.1309, loses to vihrs 0.1120 | **PH+L 0.1110, statistically tied with vihrs** | 0.922 (tie), 0.002 vs PH-only |
| **Family classification** — PH ⊕ 2 PCA components of L(r) | PH 0.8714, vihrs 0.8374 | **PH+L 0.8797**, best in project | 0.002 vs both |

Everything is at n=10 seeds, paired seed-by-seed, Wilcoxon signed-rank.

---

## 1. Where the project stood this morning

The writeup (`writeup/short_report.tex`) is a first-year progression report:
*Topological Feature Learning for Parameter Estimation in Neyman-Scott
Processes*. ParamNet = DTM-filtration persistence images → CoordConv CNN →
MLP head, trained purely on synthetic clouds, benchmarked against `vihrs`
(Vihrs 2022: 1-D CNN over edge-corrected `L(r)−r` + `log N`) and minimum
contrast.

**The honest state of the results (full budget, DTM k=5/10/15):**

| process | PH (ParamNet) | vihrs | winner |
|---|---|---|---|
| thomas | 0.1242 | **0.1120** | vihrs |
| matern_cluster | 0.1406 | **0.1242** | vihrs |
| nested_thomas | **0.1977** | 0.2091 | PH |
| aniso_thomas | 0.4500 | **0.4381** | vihrs |
| strauss | 0.5581 ± 0.18 (unstable) | **0.1921** | vihrs, 3× |
| **classification** (4-way) | **0.8766** | 0.8374 | **PH** |

The problem: **vihrs wins parameter estimation 4/5**, and on Strauss — the
no-closed-form Gibbs process that most justifies a simulation-trained
estimator — PH is 3× worse *and* fails to converge on 3/5 seeds. PH's only
parameter win (nested Thomas) is inside seed noise. PH's one clear advantage
is classification.

**Standing conceded limitations:** "no advantage on Thomas"; the single-k vs
fused-k choice is a per-process hack that "undercuts the process-agnostic
framing"; three alternative vectorizations all lost to persistence images.

**Framing discussion (morning).** The proposed pitch was
classification + parameter inference as one "process-agnostic" pipeline.
Assessment: the direction is right but rests on two overstated legs — vihrs is
not "only slightly" better at parameter inference (4/5 plus a 3× Strauss
blowup), and the classification win is one 4-class benchmark that is mostly
*within* the Neyman-Scott cluster family (Strauss is trivially separable at
99.7%; matern-vs-thomas is the hard part at ~64%). The missing keystone is an
**end-to-end two-stage evaluation** (classify → dispatch to family head vs.
oracle family vs. a single pooled head). That experiment still does not exist.

---

## 2. What was attempted today

### 2.0 Infrastructure crisis (unplanned, ~1.5 h)

`/gpfs/scratch/.../envs/cloud-env` had been **partially deleted by a GPFS
access-time purge** — the pure-Python stdlib `.py` files were reaped
(`encodings/`, `os.py`, `codecs.py` gone; `.so` files and `torch` survived),
so Python could not bootstrap at all:
`Fatal Python error: init_fs_encoding ... no codec search functions`.

Symptom: every SLURM array task failed in <1 s with ExitCode 1:0 before any
repo code ran. It killed the first dim-split submission entirely (jobs
25811867/75/76/77 — all 76 tasks dead).

Rebuilt in place from the recipe recovered from the purged env's surviving
`conda-meta/history`, using the module's mamba
(`/share/apps/rocky9/general/apps/miniforge/24.7.1/bin/mamba`; note
`module load miniforge` does *not* put conda/mamba on PATH in a
non-interactive shell). 203 packages, exact versions preserved
(python 3.11.16, torch 2.5.1/cu12.4, gudhi 3.13.0, ripser 0.6.14), plus
`pip install -e`. Old tree kept at `envs/cloud-env.broken-20260907`.

> **Still at risk.** The env is back on scratch, so the purge will recur.
> Relocate it to home/project space or add a weekly
> `find $ENV -exec touch {} +`. `grok-env` is *not* a fallback — it has torch
> but no gudhi/ripser.

### 2.1 Idea: separate encoders per homology dimension — **REJECTED**

Hypothesis: H0 and H1 persistence images are calibrated to *different*
birth/persistence ranges, so a pixel means different things in each channel.
Feeding them as 2 channels into one CoordConv stack (mixed at conv-1) is
under-constrained; give each dimension its own encoder.

Design: 5 arms bracketing the capacity confound (`shared_e64` control,
`split_e32` head-width-matched, `split_e64` unconstrained, plus `shared_e128`
and a narrowed-conv arm as capacity controls), k5-only, 3 datasets.
Core arms run at n=10 / n=5.

**Result: consistently and significantly *harmful*.**

| process | shared (control) | split_e32 | split_e64 |
|---|---|---|---|
| thomas | **0.1300** | 0.1327 (p=0.049) | 0.1319 (p=0.193) |
| nested_thomas | **0.2072** | 0.2119 (p=0.020) | 0.2150 (**p=0.002**) |
| classification | 0.8729 | 0.8712 | 0.8734 |

Never better on any process at either width. On nested Thomas — where the
hypothesis predicted the *largest* gain (H1 voids from nesting) — it is the
worst, degrades monotonically with capacity, and the H1-relevant scale
parameters are exactly what regressed. Reading: the shared encoder's ability
to mix H0/H1 features in early conv layers is doing useful work; separating
them removes it and adds parameters to overfit.

**Decision: H0+H1 stay as stacked channels. One design question closed.**

### 2.2 Idea: L-reparameterized filtration — **mechanism confirmed, no net gain**

Transform the diagram `(b, d) → (L(b), L(d))`, where L is the cloud's own
empirical Ripley L. This is *not* a coordinate hack: the empirical K is a
cumulative sum of positive weights, so L is monotone, and applying a monotone
map to a filtration's values yields exactly the diagram of the reparameterized
filtration. L-units are "equivalent-Poisson radius" (`L(r) = r` under CSR).

The motivating idea was a **factorization**: the L curve carries the
second-order trend, the transformed diagram carries topology *with that trend
divided out* — orthogonal by construction rather than by luck. Prediction: the
transform alone should look *worse* (the trend is moved into L and never fed
back), so the test is D vs C in a 2×2, not A vs B.

**Result (n=10):**

| arm | thomas | nested_thomas |
|---|---|---|
| A `dtm`, L off | 0.1300 | 0.2072 |
| B `l_dtm`, L off | 0.1287 | 0.2622 (**+0.055, p=0.002 — worse**) |
| C `dtm`, L on | 0.1110 | 0.2183 |
| D `l_dtm`, L on | 0.1113 | **0.2065** |
| **D vs C** | +0.0003, p=0.375 | **−0.0118, p=0.049** |

The predicted mechanism is confirmed (B ≫ A on nested, p=0.002 — the transform
alone is strongly harmful, exactly as it should be). The conditional effect is
real and significant on nested (D vs C, p=0.049). **But there is no net gain:**
nested D (0.2065) ≈ A (0.2072) ≈ vihrs (0.2091). The transform mostly *rescues
the damage that adding L causes*, rather than beating plain DTM. On Thomas it
does nothing (p=0.375).

**Decision: demote to an ablation row.** The 2×2 is a clean interaction result
worth one paragraph, not a headline contribution.

### 2.3 Idea: PH ⊕ L(r) fusion — **TWO WINS**

Chased incidentally from 2.2: the C-vs-A comparison showed that simply feeding
`L(r)−r` samples into the scalar side-vector took Thomas 0.1300 → 0.1110
(p=0.002). That is the Idea-1/Idea-3 fusion claim, obtained from 8 scalar
columns rather than a three-branch architecture.

Swept how the L curve is encoded, 5 arms × 10 seeds × 3 processes, plus
extending the vihrs baseline from 5 to 10 seeds so every comparison is paired.

---

## 3. Fusion results (the main output of the day)

### Thomas — parameter estimation

| arm | loss (n=10) | vs PH-only | vs vihrs |
|---|---|---|---|
| `Loff` PH only | 0.1309 ± 0.0070 | — | +0.0189, **p=0.002 (worse)** |
| **`raw8`** 8 raw L cols | **0.1110 ± 0.0046** | **−0.0199, p=0.002** | **−0.0011, p=0.922 → TIE** |
| `pca2` | 0.1760 ± 0.0028 | +0.0451, p=0.002 | +0.0640, p=0.002 |
| `pca3` | 0.1359 ± 0.0052 | +0.0050, p=0.020 | +0.0238, p=0.002 |
| `pca4` | 0.1193 ± 0.0041 | −0.0116, p=0.002 | +0.0072, p=0.004 |
| `vihrs` | 0.1120 ± 0.0035 | — | — |

**PH alone is significantly worse than vihrs (p=0.002). PH+L is
statistically indistinguishable from it (p=0.922).** Stable: per-seed range
[0.1007, 0.1176], the tightest arm in the sweep. Per-target, the gain is
concentrated in parent intensity (0.213 → 0.185) and mean offspring
(0.140 → 0.118) — the density/count targets, exactly where a second-order
statistic should help a density-based topological feature.

This directly retires the writeup's "**No advantage on Thomas**" limitation.

### Classification — 4-way family ID

| arm | accuracy (n=10) | vs PH-only | vs vihrs |
|---|---|---|---|
| `Loff` PH only | 0.8714 ± 0.0026 | — | +0.0340, p=0.002 |
| `raw8` | 0.8275 ± 0.0385 | −0.0440, p=0.027 **(worse)** | −0.0099, p=0.695 |
| **`pca2`** | **0.8797 ± 0.0024** | **+0.0082, p=0.002** | **+0.0423, p=0.002** |
| `pca3` | 0.8202 ± 0.0841 | −0.0510 (3/9 seeds collapsed) | — |
| `pca4` | 0.8683 ± 0.0402 | −0.0031, p=0.084 | +0.0309, p=0.084 |
| `vihrs` | 0.8374 ± 0.0046 | — | — |

`pca2` = **0.8797**, the best classification number in the project (above the
full-budget k5+10+15 PH result of 0.8766, at 1/3 the persistence cost), with
the tightest spread of any arm and no collapsed seeds.

### Nested Thomas — honest null

| arm | loss (n=10) |
|---|---|
| `Loff` PH only | **0.2091 ± 0.0059** |
| `raw8` | 0.2180 ± 0.0127 (+0.0088, p=0.160) |
| `pca2/3/4` | 0.266 / 0.315 / 0.288 — catastrophic, 2–4 collapsed seeds each |
| `vihrs` | 0.2091 ± 0.0075 |

PH alone already ties vihrs exactly (Δ=0.0001, p=0.846). Fusion adds nothing;
PCA breaks it. Least to gain here, and nothing gained.

### The unifying finding

**The right L encoding is task-dependent, and the pattern is interpretable:**

- **Classification needs only the 2 leading directions** of the L curve
  (essentially clustering *amplitude* and clustering *scale*). Those separate
  the families cleanly. The remaining raw directions are estimation noise that
  the cross-entropy head overfits — which is why `raw8` **collapses** on
  classification (3 bimodal seeds; this reproduced the earlier observation, so
  it is a real effect and not the bug in §4).
- **Continuous parameter estimation needs finer curve detail.** Pinning three
  real-valued parameters uses more of the curve's shape, so truncating to 2
  components discards signal (`pca2` = 0.176, far worse than PH-only) and the
  raw curve wins. The MSE head does not overfit the noise directions the way
  the CE head does.

An earlier working hypothesis — that collinearity in the raw L columns
(correlation-matrix condition number ~2.8×10⁴) would harm *everything* — was
**wrong as a generalization**. It harms the classification head only. The PCA
fix is correct for classification and actively damaging for regression.

---

## 4. Bug found and fixed

`build_extra` inferred "fitting vs applying frozen" from `n_norm is None`. With
`include_log_n: false` no `n_norm` is ever produced, so **the adversarial /
extrapolation call silently re-fit its normalization on the evaluation
population** instead of applying the frozen train-split statistics — a leak.

Pre-existing and latent since `include_log_n` was added; harmless only while no
other scalar column group was ever used without `log N`. That is exactly the
classification configuration, so it bit as soon as L columns were added. Now
keyed off `train_idx is not None` with explicit guards.

**Scope of impact:** `adversarial_*` numbers in the earlier classification
lfunc runs are invalid. Main-split `test_loss` / `test_accuracy` were never
affected (that path always fit correctly), so no reported headline number
changes.

---

## 5. Code shipped today (all uncommitted)

**New modules**
- `src/cloudforger/data_generation/filtration/lfunc.py` — `LFuncFiltration`
  wrapping any base filtration; registry presets `l_rips` / `l_dtm`
  (path tags `l_rips`, `l_dtm_k5`). Handles monotonicity enforcement, the
  slope-1 tail above `r_max`, `inf` passthrough for the essential H0 bar.
- `src/cloudforger/experiments/pi_multik/pi_multik_dimsplit.py` —
  `pi_multik_dimsplit` method (per-homology-dim encoders).
- `scripts/processing/transform_diagrams_lfunc.py` — builds `l_dtm_k5` bundles
  from existing `dtm_k5` ones + cached L, **without recomputing persistence**
  (seconds, not hours).
- `scripts/collect_fusion_results.py` — paired-Wilcoxon aggregator with a
  MAD-based seed-collapse detector.

**Modified**
- `pi_multik.py` — `include_lfunc: N` (N log-spaced `L(r)−r` columns),
  `lfunc_pca: k` (whitened PCA, fit on train, applied frozen), plus the §4 fix.
  Defaults are off, so every pre-existing run is bit-identical.
- `scripts/train.py`, three `__init__.py` — registration.

**Configs** (9): `{thomas,nested_thomas,classification}/{dimsplit,lfunc,fusion}_k5.yaml`
**SLURM** (8): `dimsplit_*`, `lfunc_*`, `fusion_*`, `vihrs_topup`, `lfunc_transform`

**Compute:** ~343 array tasks submitted; 267 completed successfully, 76 lost to
the env purge and re-run.

---

## 6. What this changes about the paper

The pitch is materially stronger than this morning:

1. **"No advantage on Thomas" is retired.** PH+L ties the radial baseline on
   its home ground, with a coherent per-parameter mechanism.
2. **The classification lead widens** to 0.880 vs 0.837 (+4.2 pts, p=0.002) at
   1/3 the persistence cost of the current headline configuration.
3. **The complementarity claim now has direct evidence**: topological and
   radial features carry non-redundant information, and *how much* of the
   radial curve helps depends on whether the task is discrete or continuous.
   That is a more interesting result than a single improved number.

**Still unresolved and still the biggest risks:**
- **Strauss.** PH is 0.558 (unstable, 3/5 seeds fail to converge) vs vihrs
  0.192. Fusion was not tested here. This is the case the "handles
  no-closed-form processes" pitch most needs, and it is the weakest result in
  the project.
- **The end-to-end two-stage experiment does not exist** (classify → dispatch
  vs. oracle family vs. pooled head). Without it, the process-agnostic framing
  is an argument, not a result.
- Nested Thomas: fusion adds nothing.
- No real data. No inhomogeneous processes.

---

## 7. Suggested next steps

1. **Run the fusion arms on Strauss, matern_cluster, aniso_thomas** (`raw8` for
   params). Cheap — the L caches and DTM diagrams already exist; it is one more
   sbatch per process. If `raw8` also rescues Strauss, the fusion result
   becomes the paper's spine rather than a Thomas-specific fix.
2. **Build the end-to-end two-stage evaluation.** This is the keystone for the
   headline framing and remains the single highest-value missing experiment.
3. **Lock the encoding per task** in the writeup: `raw8` for parameter
   estimation, `pca2` for classification, with the task-dependence explained as
   a finding rather than a tuning artifact.
4. **Move `cloud-env` off scratch** before the purge recurs.
5. Commit today's work — the tree is entirely uncommitted.

---

## 8. Caveats on today's numbers

- All fusion/dimsplit/lfunc numbers are **k5-only**, chosen to isolate the
  variable under test. They are *not* directly comparable to the writeup's
  fused DTM k=5/10/15 table; compare within-experiment against the `Loff` /
  `shared_e64` control.
- Classification `pca3` has 3/9 collapsed seeds and `pca4` 1/10; only `pca2`
  is clean. The sweet spot is narrow — worth one more confirmation run before
  the number goes in a table.
- vihrs and PH derive their splits independently from
  `train_val_test_indices(n, seed)`; identical `n` and seed means identical
  splits, but this was asserted rather than byte-verified.
- One `fusion_classification` task was still running on a slow node at the time
  of writing (`pca3` seed, ~47 min vs the ~3 min norm), hence n=9 for that arm.
