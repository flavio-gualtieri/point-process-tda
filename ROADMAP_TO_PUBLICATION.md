# Roadmap to a publishable result

Analysis of the current state of `point-process-tda` / `cloudforger` and a
concrete plan for closing the gap to publication. Written 2026-09-01 against
commit `270eb88` (branch `frontend`).

Contents:

1. [Where the project actually stands](#1-where-the-project-actually-stands)
2. [The three strategic options, assessed](#2-the-three-strategic-options-assessed)
3. [Recommended way forward (synthesis)](#3-recommended-way-forward-synthesis)
4. [Missing canonical point processes](#4-missing-canonical-point-processes)
5. [Inhomogeneous processes: what to estimate and how to simulate](#5-inhomogeneous-processes-what-to-estimate-and-how-to-simulate)
6. [Application candidates](#6-application-candidates)
7. [Filtrations to try](#7-filtrations-to-try)
8. [Codebase: how it works today and what to change](#8-codebase-how-it-works-today-and-what-to-change)
9. [Sequenced task list](#9-sequenced-task-list)

---

## 1. Where the project actually stands

**What works.** ParamNet (DTM-filtration persistence images at `k ∈ {5,10,15}`,
per-diagram coverage-quantile calibration, CoordConv CNN, late fusion, FCNN head)
is trained purely on synthetic clouds and benchmarked on shared splits against
`vihrs` (Vihrs 2022: 1-D CNN over the edge-corrected `L(r)−r` plus `log N`) and
minimum contrast. Headline numbers from `writeup/short_report.tex`:

| | Thomas `L` | Nested-Thomas `L` |
|---|---|---|
| ParamNet | 0.125 | **0.192** |
| vihrs | **0.115** | 0.217 |
| min-contrast (K) | 0.855 (unstable) | — |
| chance (label shuffle) | ~1.0 | ~1.0 |

The honest reading, which the report already gives: **persistence images carry
real signal (an order of magnitude below chance) but do not beat a well-chosen
radial summary on single-level Thomas.** ParamNet's advantage is confined to
nested Thomas and to its density/count targets; `vihrs` still wins both *scale*
parameters even there.

**Why Option 1 is hard.** For a stationary isotropic Neyman-Scott process the
pair `(intensity, K-function)` is very close to a sufficient statistic for
`(κ, μ, σ)` — the K-function *is* the closed form these parameters were
historically estimated from. `vihrs` feeds the network almost exactly that
statistic. A density-based topological feature is structurally disadvantaged on
precisely the family where a sufficient radial statistic exists. Beating it there
means beating it on its home ground.

**Assets that already exist but are not wired up or not trusted** (these matter
for the plan):

- `InhomThomas` (`src/cloudforger/data_generation/point_processes/inhom_thomas.py`)
  — a full inhomogeneous Thomas with RFF Gaussian-random-field covariates and
  location-dependent thinning; labels `(parent_intensity, cluster_scale, beta)`;
  `expected_points` decouples count from `beta`. **No config, no run.** This is
  most of Option 2 sitting on the shelf.
- `uniform_ball_displacements` (`neyman_scott.py:27`) — the offspring kernel for
  the *Matérn cluster* process. Defined, never used. A `MaternClusterProcess` is
  ~15 lines.
- `MaternHardCoreProcess` — a *repulsive* Type-II process (not Neyman-Scott).
  Data-only report, never trained. Note the name collision: the report's
  "Matérn" means the *cluster* process, which is not this class.
- `NeymanScottProcess.parent_sampler` already composes to arbitrary depth — a
  3-level nested process needs no new class, just a loop.
- `bifiltration/` + `experiments/mph_*` (multiparameter persistent homology via
  `multipers`). An unreported 3-seed side experiment had `mph_fusion` *beating*
  `vihrs` on Thomas (0.113 vs 0.115). Archived, not in the report.
- `mincontrast_nested` / `mincontrast_g_nested` — implemented and run (5 seeds)
  but **explicitly unvalidated**; the numbers (179×, 15× worse than chance) are
  not trustworthy per the module's own docstring.
- `scripts/archive/estimate_error_floor.py` + `configs/runs/*/error_floor.yaml`
  — a Bayes-risk reference. Never produced output (NaN bug: per-target loss
  indexed against a label list that wrongly included `edge_buffer`/`c1`).

**The computational ceiling.** Persistence computation scales badly in `n`, so
the design distribution is capped at `E[N] ∈ [150, 800]`. The DTM/PH C++ backend
also leaks memory, forcing `featurize.py` down to **one worker** on a 16 GB
machine. Any plan that wants denser or larger clouds has to address this first
(see §7 alpha complexes, §8 item 7).

---

## 2. The three strategic options, assessed

### Option 1 — Improve the model so it demolishes `vihrs`

**Viability: low as stated; moderate if reframed.** "Demolish `vihrs` on
Thomas" fights on the baseline's home ground (see §1). What *is* achievable and
still publishable:

- **Orthogonality / fusion.** Show `vihrs` features and topological features
  carry *complementary* signal: `topo ⊕ L(r)−r` beats either alone, on Thomas
  and beyond. The archived `mph_fusion` result is a hint this is real. This is a
  legitimate contribution ("topology adds information a sufficient-for-stationary
  statistic misses") without the impossible "demolish" bar.
- **Beat `vihrs` where `L(r)−r` is not near-sufficient**: anisotropy,
  non-stationarity, superposition, interaction — i.e. Option 2. There the
  advantage is structural, not a matter of tuning.
- **Equal accuracy at lower assumption cost.** `vihrs` bakes in stationarity,
  isotropy, a fixed `r_max`, and an edge correction specific to a rectangular
  window. If ParamNet matches it on Thomas *and* keeps working when those break,
  "same accuracy, fewer assumptions" is a paper.

**Way forward for Option 1:** fold it into Option 2 as the "fusion + orthogonal
signal" sub-claim. Do not pursue a standalone architecture bake-off aimed at the
Thomas leaderboard — the report already concedes that fight and reviewers will
see it.

### Option 2 — Pathological cases where existing methods fail

**Viability: high. This is the strongest scientific pitch and matches the
report's own "Planned directions".** The failure modes of K-function / minimum
contrast / `vihrs` are well understood:

| Regime | Why classical / `vihrs` fails | Why topology can win |
|---|---|---|
| **Inhomogeneous / non-stationary** | `L(r)−r` assumes stationarity; Waagepetersen–Guan two-step needs covariates *observed* and the intensity model *correctly specified* | simulation-trained estimator needs neither; local density is exactly what DTM sees |
| **No closed-form K** (Strauss, LGCP-Strauss, Gibbs/interaction, determinantal) | minimum contrast has no curve to fit; MLE needs MCMC / intractable normalizing constant | PH needs no closed form and no summary-statistic choice |
| **K exists but fails regularity** (Matérn *cluster*: `K_ψ` not twice-differentiable in `ψ` — the report already cites this) | Waagepetersen–Guan estimator's theory explicitly excludes it | no smoothness assumption on any summary |
| **Heavy-tailed cluster kernel** (Cauchy, variance-gamma) | minimum contrast unstable; `r_max` choice dominates | multi-scale by construction |
| **Anisotropic clustering** (elliptical kernel) | radially-averaged K discards the anisotropy entirely | a 2-D persistence image / directional filtration sees it |
| **Superposition / contamination** (two scales; cluster + CSR background) | K-fitting notoriously unstable under mixing | multi-scale persistence separates the components |
| **Short-range inhibition + long-range clustering** (hybrid) | non-monotone `g`; `L(r)−r` blurs the exclusion zone | `H1` loops from exclusion zones carry real signal |

**Cleanest targets** (a real "we win where they fail" result, minimal new
machinery): **inhomogeneous Thomas** (flagship — code exists), **Matérn cluster**
(regularity failure — kernel exists), **anisotropic/elliptical cluster**,
**cluster + background contamination**, and **Strauss** (no closed-form density —
the exact case Vihrs motivates neural estimation with).

**Way forward for Option 2:** §4 (add the processes) → §5 (wire inhomogeneity) →
run the frozen ParamNet + `vihrs` + min-contrast comparison across all of them →
the deliverable is a single degradation table: *each row a process, each column a
method, cells are `L` relative to chance and to the error floor.* The story
writes itself from the shape of that table.

### Option 3 — Real applications with physical observables

**Viability: medium-high, but only compelling when paired with Option 2.** A
real-data fit that merely reproduces a literature minimum-contrast fit is not a
paper. The exciting version is *"real data where the process is inhomogeneous or
has no closed form, and we recover a physical observable the standard method
cannot"*.

**Way forward for Option 3:** pick one anchor dataset that is *simultaneously*
Option 2 and Option 3 — **Barro Colorado Island tree data** (Waagepetersen–Guan
2009 already fit a Thomas process; `σ` reads directly as seed-dispersal range;
the pattern is genuinely inhomogeneous in elevation and soil, which is the whole
point of their two-step method). Add one secondary dataset from a different
field (spatial single-cell / tumour microenvironment, or a `spatstat` classic
like `redwood`/`longleaf`). Report the observable with an uncertainty, next to
the literature fit and next to what happens when the covariates are withheld.

See §6 for the full candidate list.

---

## 3. Recommended way forward (synthesis)

**Pursue Option 2 as the spine; use Option 3 (BCI trees) as the concrete
real-data anchor; downgrade Option 1 to a "fusion / orthogonal signal"
sub-claim.**

Target claim:

> A simulation-trained topological estimator recovers interpretable
> point-process parameters — including covariate effects — in inhomogeneous and
> no-closed-form regimes where minimum contrast and stationarity-based neural
> estimators are biased or inapplicable, validated on real ecological (and
> cell-biology) data.

Four phases, each a self-contained deliverable:

- **Phase A — freeze + round out the homogeneous family.** Add a `Kernel`
  abstraction (§8.1); add Matérn cluster, Cauchy cluster, LGCP, Strauss (§4);
  re-run frozen ParamNet + `vihrs` + min-contrast on all of them. *Deliverable:
  the degradation table.* Expected: `vihrs` stays competitive where closed-form K
  exists and is smooth; min-contrast fails on Matérn cluster (regularity) and
  Cauchy (tail); Strauss breaks minimum contrast entirely; ParamNet degrades most
  gracefully.
- **Phase B — inhomogeneity.** Fix the design spec to carry a vector `beta`
  (§8.2); wire `InhomThomas`; add inhomogeneous LGCP and a non-stationary-`σ`
  Thomas. Comparator: Waagepetersen–Guan two-step. *Deliverable: ParamNet recovers
  `beta` and `(κ, σ)` with covariates unobserved / misspecified; two-step cannot
  run or is badly biased.*
- **Phase C — new filtration.** Diffusion-distance / commute-time Rips filtration
  and a KDE-sublevel cubical filtration (§7). *Deliverable: at least one gives a
  measurable gain on the anisotropic / inhomogeneous cases, reported as an
  ablation axis next to DTM and Rips.*
- **Phase D — real data.** BCI trees primary, a cell-biology or `spatstat`
  dataset secondary. *Deliverable: physical observable + uncertainty vs. the
  literature fit, and a covariates-withheld robustness result.*

Also fix the **error floor** (§1) early — every one of these phases wants a
Bayes-risk reference, and reviewers will ask "how close to optimal is this".

---

## 4. Missing canonical point processes

Current registry (`data_generation/point_processes/__init__.py`): `poisson`,
`matern` (hard-core Type II), `neyman_scott`, `thomas`, `nested_thomas`,
`inhom_thomas`.

Priority additions (all are small once §8.1's `Kernel` abstraction exists):

| Process | Family | Why it matters | Closed-form K? | Effort |
|---|---|---|---|---|
| **Matérn cluster** | NS, uniform-in-ball kernel | the report's own example; `K_ψ` not twice-differentiable → excludes Waagepetersen–Guan | yes (but non-smooth) | ~15 lines; kernel exists (`uniform_ball_displacements`), `edge_buffer = R` **not** `4R` |
| **Cauchy cluster** | NS, bivariate-`t`/Cauchy kernel | heavy-tailed displacement; minimum contrast unstable; tests kernel-shape recovery | yes (Ghorbani 2013) | one `Kernel` subclass |
| **Variance-Gamma (Bessel) cluster** | NS, VG kernel with shape `ν` | interpolates Thomas ↔ Cauchy; "does the model recover kernel *shape*?" | yes (Jalilian et al. 2013) | one `Kernel` subclass |
| **Log-Gaussian Cox (LGCP)** | Cox / doubly stochastic | canonical, extremely widely used, *different mechanism* from NS; params = GRF variance `σ²` + scale `β` | yes | new class; reuse the RFF field from `inhom_thomas.py` or use a grid GRF |
| **Strauss** | Gibbs / interaction (repulsive) | canonical **no-tractable-normalizing-constant** case — exactly what Vihrs motivates neural estimation with; MLE needs MCMC | no | birth–death / MH sampler, ~40 lines |
| **Matérn hard-core Type I** | repulsive | completes the repulsive pair (Type II already implemented) | — | ~10 lines |
| **Thomas with elliptical kernel** | NS, anisotropic Gaussian | radial K discards the anisotropy; direct Option-2 win | K exists for the isotropised version only | `Kernel` subclass with `(σ_major, σ_minor, θ)` |

Secondary / stretch: determinantal point process (exact likelihood, repulsive,
different regime); Simple Sequential Inhibition (Matérn III-like); Neyman-Scott
with random cluster sizes drawn from a non-Poisson law (negative binomial) to
break the `Var[N] = κμ(1+μ)` identity the audit relies on.

**"Freezing" the model** just means: tag the commit, and snapshot the config set
(`configs/runs/{thomas,nested_thomas}/pi_multik*.yaml`,
`*_vihrs.yaml`, `mincontrast*.yaml`) into `configs/frozen/`. No model-code change
is needed to run it against new processes — the pipeline keys everything off
`process.name` and the registry.

---

## 5. Inhomogeneous processes: what to estimate and how to simulate

### What the estimator should output

`(κ, μ, σ, β)` where `β` is the covariate-effect vector for a log-linear
intensity/retention model `ρ(u) ∝ exp(β·Z(u))`. `InhomThomas` already produces
exactly `(parent_intensity, cluster_scale, beta)` and uses `expected_points` to
hold the retained count roughly constant across the `β` sweep, so count does not
trivially leak `β` (there is no identifiable intercept).

### Simulation mechanisms (in rough order of value)

1. **Inhomogeneous offspring / retention** (already in `InhomThomas`): homogeneous
   parents, offspring kept with probability `∝ exp(β·Z(u))`. This is
   Waagepetersen's second-order intensity-reweighted stationary construction —
   the exact setting the two-step estimator targets. **Start here.**
2. **Inhomogeneous parents**: parent intensity `κ(u) = κ₀ exp(β·Z(u))` via
   thinning a homogeneous Poisson at rate `κ_max`. Clusters concentrate where `κ`
   is high. Combine with (1) for a fully inhomogeneous NS process.
3. **Non-stationary cluster scale** `σ(u)`: clusters tighter in some regions.
   This breaks *reweighted* K too, not just plain K — a stronger pathological
   case, and one no classical estimator handles.
4. **Inhomogeneous LGCP**: mean function `m(u) = β·Z(u)` plus a stationary GRF.
   Canonical inhomogeneous Cox process; well-studied comparator literature.
5. **Anisotropy with spatially varying orientation** `θ(u)`: elliptical kernel
   whose axis follows a covariate (e.g. a slope gradient). Breaks isotropy, not
   just stationarity.

### Covariate fields for simulation

- **Smooth Gaussian random fields** via random Fourier features — already
  implemented (`_RFFField` in `inhom_thomas.py`), controllable length scale,
  standardised to mean 0 / unit variance over the domain. Use `field_seed=None`
  for a fresh field per realisation (model learns to be field-agnostic) or a
  fixed seed to reuse one field.
- **Deterministic trends** (linear / quadratic gradient) — interpretable sanity
  checks; `β` has an exact meaning you can read off a scatter plot.
- **Real covariate rasters** for Phase D — BCI elevation + slope; population
  density for an epidemiological dataset. Add a small raster-covariate `Kernel`/
  field type that bilinearly interpolates a supplied grid.

### The one code gap

`data_generation/design.py::build_param_vectors` samples **scalar** parameters
from per-key ranges and applies **scalar** `derived` expressions (`apply_derived`
does `float(eval(...))`). It cannot sample or emit a vector `beta`. Fix options,
smallest first:

- Sample `beta_0, beta_1, …` as separate ranges, and teach
  `generate_clouds_for_design` to pack any `beta_*` keys into a vector before
  calling the constructor. ~15 lines, no schema change.
- Add an explicit `vector_ranges:` block to the design spec and a `pack:` step.
  Cleaner, slightly more work.
- Allow a `derived` expression to return a list (drop the `float(...)` cast, add
  a type check). Fragile — the `eval` sandbox would need list literals.

Recommend the first for Phase B, the second if inhomogeneity becomes central.

### Comparator

Waagepetersen–Guan two-step (`spatstat::kppm` with a trend formula). Show:
(a) ParamNet matches it when covariates are observed and the model is correct;
(b) ParamNet still works when covariates are **withheld** (marginalised) or
**misspecified** (wrong basis), where the two-step is biased or cannot be fit.

---

## 6. Application candidates

Ranked by "exciting × low-friction × plays to Option 2":

1. **Barro Colorado Island tropical tree census** (ecology). Public; ~50 ha
   mapped plot, many species, elevation + soil covariates. Waagepetersen–Guan
   2009 already fit Thomas processes to three species and read `σ` as
   seed-dispersal range, finding that capsule- / wind- / animal-dispersed species
   separate along it. **This is the anchor**: inhomogeneous, literature fit to
   compare against, physically meaningful observable, and it is *the* dataset the
   two-step method was built for.
2. **Spatial single-cell / tumour microenvironment** (cell biology). Multiplexed
   imaging or spatial transcriptomics; positions of cytotoxic T cells, tumour
   cells, etc. The degree and scale of immune-cell clustering is prognostic. The
   tissue is strongly inhomogeneous (tumour vs. stroma), so stationarity-based
   fits are suspect. Public data exists (imaging-mass-cytometry cohorts, spatial
   challenge datasets). Observable: infiltration cluster scale / occupancy.
3. **Star-forming regions / young stellar objects** (astronomy). YSO positions
   from infrared surveys; cluster scale ≈ core radius / local Jeans length.
   Inhomogeneous (molecular cloud structure). Ties to the report's cosmology
   framing without needing a full 3-D halo catalogue.
4. **Galaxy / halo catalogues** (cosmology). The report's running example:
   `κ` → halo number density, `μ` → mean halo occupancy, `σ` → satellite spatial
   extent. Higher friction (3-D, needs the region/dimension wiring in §8.8, large
   `n`), and Yip et al. 2025 already occupy the "PH → cosmological params" niche —
   but "PH → *point-process* params → HOD" is a distinct angle.
5. **Nanoparticle / colloid aggregation** (materials, SEM/TEM/AFM). Particle
   centroids from micrographs; cluster scale as a QC / process observable.
   Abundant images, but data assembly is per-paper.
6. **Disease-case clustering** (epidemiology). Case locations; the confound is
   population density (inhomogeneity) — which is precisely the Option-2 selling
   point. Data-access and ethics friction is real.

For a first submission: **BCI trees + one of {cell biology, a `spatstat`
classic}**. Two datasets from different fields is enough; more dilutes.

---

## 7. Filtrations to try

The `Filtration` ABC (`data_generation/filtration/base.py`) is a clean plug
point: implement `_compute_diagrams(cloud) -> {dim: (m,2) ndarray}`, `name`,
`params`, `path_tag()`, and register. Calibration is already re-fit per run / per
seed on the training split, so a filtration living on a different value scale
needs no special handling.

Candidates, grouped:

### Cheap, already supported

- **DTM sweeps**: more `k` values, `q ≠ 2`. One-line config changes.

### Attack the `n ≤ 800` ceiling

- **Alpha complex** (`gudhi.AlphaComplex`). In 2-D the alpha complex is
  `O(n)`-sized vs. Rips's `O(n²)`, with the same homological information below the
  threshold. This alone could raise the point budget 3–5×. **High practical
  value** — do this before scaling up any process.
- **Alpha-DTM (`αDTMℓ`)** — density-weighted alpha; the Yip et al. filtration.
  Best of both: density sensitivity *and* alpha's efficiency.

### Density-image filtrations (natural fit for inhomogeneity)

- **Sublevel-set filtration of a KDE** on a grid → cubical complex, filtered by
  `−density`. Cubical persistence is near-linear; `H1` captures voids / rings;
  handles non-stationarity gracefully because it filters an *estimated intensity
  surface*. Strong Phase-B / Phase-C candidate.
- **DTM-on-a-grid** (cubical) — same idea, DTM instead of KDE.

### The user's random-walk idea — publishable novelty

- **Diffusion-distance / commute-time filtration.** Build a `k`-NN or `ε` graph
  on the cloud, form the graph Laplacian `L = D − W`, and use a random-walk
  metric as the pairwise distance fed to a Rips-style filtration:
  - *Commute-time distance* `d_CT(i,j)² = Σ_{λ>0} (1/λ)(φ_λ(i) − φ_λ(j))²` (sum
    over non-trivial Laplacian eigenpairs) — "expected round-trip time for a
    random walk". Equivalent to effective resistance.
  - *Diffusion distance at scale `t`*: `d_t(i,j)² = Σ_λ e^{−2λt}(φ_λ(i) − φ_λ(j))²`
    — a tunable scale parameter that is itself an ablation axis.
  - Both are genuine metrics for fixed `t`, so they drop straight into
    `RipsFiltration` via `ripser(..., distance_matrix=True)`. Add a
    `FiltrationFromDistanceMatrix` mixin (§8.4) and each is one small file.
  - *Why it could win*: the metric contracts densely-connected regions and
    stretches sparse bridges, so it is intrinsically aware of the *geometry of
    the data* — clustering shows up as short commute times, inhibition as long
    ones — in a way raw Euclidean distance (Rips) and even DTM (which only sees
    `k`-NN radius) do not. It also re-scales "closeness" in a way that may be
    more stable across the near-Poisson boundary.
  - *Cost*: `O(n²)`–`O(n³)` for the eigendecomposition; fine at `n ≤ 800`, and a
    reason alpha/cubical filtrations matter for scaling.
- **`k`-NN graph geodesic (Isomap-style) distance** — cheaper cousin:
  shortest-path distance on the neighbourhood graph. Captures filament / manifold
  structure; no eigendecomposition.

### Multiparameter / bridges to `vihrs`

- Push the existing **DTM × Rips bifiltration** (`mph_*`) — it is implemented and
  hinted at beating `vihrs` on Thomas in a 3-seed run.
- **Density × anisotropy** bifiltration for the elliptical-kernel process.
- **Persistence of the empirical directional-`K` surface** — an explicit bridge
  that gives the CNN both the radial summary and its topology.

Recommended order: alpha complex (scaling) → KDE-sublevel cubical (inhomogeneity)
→ diffusion-distance Rips (novelty) → revisit bifiltration.

---

## 8. Codebase: how it works today and what to change

**Overall: the codebase is in good shape.** The refactor described in
`_attic/docs/architecture.md` was carried out. Registries are consistent across
processes / filtrations / features / bifiltrations. The 4-stage
generate → featurize → train → evaluate contract is clean, disk-artifact-based,
and each stage is independently re-runnable. `paths.py` derives every path
deterministically. `provenance.py` stamps every result with git commit + dirty
flag + timestamp and appends a ledger row. Tests cover the registries and a full
CLI end-to-end run. **You do not need a rewrite** — you need a handful of
targeted seams opened up so the Phase A–D work is additive.

### How a new experiment flows today (so the changes below make sense)

```
RunConfig YAML  ──load_config──▶  RunConfig dataclass
     │                               (process / filtration / features / method / seeds / targets)
     ▼
scripts/generate.py   CloudDesign.build(process.name, design, adversarial).generate()
     │                 └─ design.py samples param vectors, PROCESS_REGISTRY builds the process
     ▼  data/<process>/clouds.pkl  (+ adversarial_clouds.pkl, manifest.yaml)
scripts/featurize.py  FILTRATION_REGISTRY.build(...) per filtration → diagrams.pkl
     │                 → (optional) build_calibrated_imager → persistence_image.pkl
     ▼  data/<process>/<filtration_tag>/{diagrams,persistence_image,...}.pkl
scripts/train.py      build_experiment(cfg_dict) → Experiment | MultiSourceExperiment
     │                 .run(dataset_path(s), output_dir, adversarial_path(s))
     │                 → save_results()  (one schema for CNN / vihrs / classical)
     ▼  results/<process>/<filtration_tag>/<method>/seed_<seed>/{results.pt,results.json,model.pt}
scripts/evaluate.py   collect_method_results → paired Wilcoxon vs. best, plots, summary.json
```

### The changes, in priority order

#### 8.1 — A `Kernel` abstraction for Neyman-Scott *(highest leverage for §4)*

The audit (`audit/report.md`, Step 4) already spells this out. Today `4.0 * scale`
is duplicated at three edge-buffer sites (`thomas.py:20`, `nested_thomas.py:27`,
`inhom_thomas.py:162`) and `NeymanScottProcess` takes `displacement_sampler` +
a bare `edge_buffer` number with no link between them.

Introduce:

```python
class Kernel(Protocol):
    def sample(self, n: int, dim: int, rng) -> np.ndarray: ...      # what DisplacementSampler already is
    def support_radius(self, eps: float = 1e-4) -> float: ...        # NEW

class GaussianKernel(Kernel):   support_radius = 4σ (or eps-driven)
class BallKernel(Kernel):       support_radius = R          # exact, eps-independent — Matérn cluster
class CauchyKernel(Kernel):     support_radius from the eps quantile
class EllipticalGaussianKernel(Kernel): (σ_major, σ_minor, θ)
```

`NeymanScottProcess` takes `kernel: Kernel` and computes
`edge_buffer = kernel.support_radius(eps)` once, internally. This:

- removes the three duplicated buffer sites;
- makes Matérn cluster's `R`-not-`4R` distinction **structural**, not a thing a
  future author has to remember;
- turns Cauchy / variance-gamma / elliptical into one `Kernel` subclass each.

Also add `compose_neyman_scott(levels: list[LevelSpec]) -> NeymanScottProcess`
that folds a root→leaf list of `(kernel, count_sampler)` pairs into nested
`parent_sampler`s — the thing `NestedThomasProcess.__init__` does by hand for two
levels, generalised to `k`.

#### 8.2 — Design spec: vector-valued and non-derived parameters *(needed for Phase B)*

See §5's "one code gap". Minimal version: pack `beta_*` keys into a vector in
`generate_clouds_for_design`. Do this when Phase B starts; it is ~15 lines in
`data_generation/design.py`.

#### 8.3 — Consolidate the training loop *(reduces per-experiment surface area)*

There are currently **three** near-duplicate loops that have drifted apart:

| | optimiser | weight decay | early stop | checkpoint-best | per-target eval |
|---|---|---|---|---|---|
| `training/train.py::train_and_eval` | Adam | — | no | yes | no |
| `experiments/base.py::_train_and_eval` | Adam | 1e-4 | no | yes | via caller |
| `experiments/pi_multik/pi_multik.py` (inline) | AdamW | cfg | yes | yes | yes |
| `vec_multik.py`, `betti_multik.py` (inline) | inline copies of the above |

Extract one `fit(model, loaders, cfg) -> (history, best_state, test_loss)` with
early stopping, checkpoint-best, and optional per-target MSE, and have every
experiment call it. A new experiment then becomes "build the tensor, build the
model, call `fit`" — ~30 lines. This is the single biggest guard against "one
seed / one arm silently trained under different settings" bugs, which the report
already shows biting (the `†` collapsed-seed footnotes).

#### 8.4 — Two small filtration base classes *(unlocks most of §7)*

- `FiltrationFromDistanceMatrix`: subclass supplies `distance_matrix(cloud) ->
  (n,n) ndarray`; base hands it to `ripser(..., distance_matrix=True)` and
  normalises the output. Diffusion-distance, commute-time, geodesic filtrations
  are then one method each.
- `CubicalFiltration`: subclass supplies `scalar_field(cloud) -> (G,G) ndarray`
  (KDE, DTM-on-grid, `−intensity`); base runs `gudhi.CubicalComplex`. KDE-sublevel
  and DTM-grid filtrations are one method each.

#### 8.5 — Move input-resolution into the experiment classes *(de-clutters `train.py`)*

`scripts/train.py` has accreted `MULTI_K_METHODS`, `CLASSICAL_BASELINE_NAMES`,
`FILE_KEY_TO_FEATURE_NAME`, `FILTRATION_INDEPENDENT_FILE_KEYS`, the `topo_superset`
`m_values` pathway, and `_multi_source_dataset_paths`'s branching. Give each
`Experiment` / `MultiSourceExperiment` a `resolve_inputs(cfg, data_paths,
filtrations, adversarial) -> dict[str, Path | list[Path]]` method and let
`train.py` just call it. New methods become self-describing; `train.py` shrinks to
dispatch + the seed loop.

#### 8.6 — Make the "one model, all processes" claim real *(addresses the report's headline limitation)*

The report concedes that using single-`k` for Thomas and fused `k` for nested
"undercuts the process-agnostic framing". The fix is a **scale-attention** or
set-transformer fusion over `k` in `PIMultiK` — the seam is already there
(`EncoderBank` → `ConvFusion` in `experiments/pi_multik/pi_multik.py`). One config
covering every process is a concrete, reviewable improvement and removes a
standing limitation.

#### 8.7 — Point-budget relief *(prerequisite for denser / larger processes)*

- Wire the alpha-complex filtration (§7) and raise the `E[N]` ceiling in the
  design constraints.
- Address the DTM/PH memory leak that currently forces `featurize.py` to one
  worker: run each diagram in a short-lived subprocess with a hard RSS cap, or
  batch-and-recycle more aggressively than `max_tasks_per_child=10`. Right now a
  single 16 GB machine is the bottleneck for every dataset.

#### 8.8 — Thread `region` / `dimension` through `RunConfig` *(prerequisite for 3-D / real windows)*

The generator is already dimension-generic (audit Step 2, check 8). Only
`config.py` (`ProcessConfig` has no `region`/`dimension` field) and
`scripts/generate.py` (never passes `region=` to `CloudDesign.build`, so it always
falls back to `default_region(2)`) need the passthrough. Needed for galaxy
catalogues and for any real dataset whose window is not the unit square.

#### 8.9 — Fix or retire the error-floor script *(reviewers will ask)*

`scripts/archive/estimate_error_floor.py` + `configs/runs/*/error_floor.yaml`
exist but never produced output — the per-target loss array is indexed against a
label list that wrongly includes `edge_buffer`/`c1`. Either fix the indexing and
produce the Bayes-risk number, or delete it and compute the floor a different way
(replicate dispersion of a fixed per-realisation estimator on a `θ`-grid — a
one-line change to the eval path). Every phase wants this reference.

#### 8.10 — Validate or remove `mincontrast_nested` / `mincontrast_g_nested`

They run but produce untrustworthy numbers (module docstring: "NOT yet validated
end-to-end", "the derivation needs independent checking"). Spot-check the recovered
parameters against known simulator inputs, or drop them from the comparison set so
they cannot leak into a table.

#### 8.11 — Housekeeping (low urgency)

- `stats/` (`PairDistanceCDF`, `CloudStatistic`) is a flagged orphan — fold into
  `vectorization/` as a non-topological baseline feature, or archive.
- `_normalize_labels` exists in both `experiments/base.py` and (as
  `fit_label_norm`/`apply_label_norm`) in `experiments/common.py`; converge on the
  `common.py` pair.
- The `data/` tree is 3.7 GB of pickles in the working copy — confirm `.gitignore`
  / `.gitattributes` keep it out of the repo (it looks like they do) before it
  grows with new processes.

---

## 9. Sequenced task list

**Phase 0 — unblock (1–2 days)**

- [ ] Fix the error-floor indexing bug (§8.9); produce the Thomas + nested floor.
- [ ] Snapshot the frozen model: tag the commit, copy configs to `configs/frozen/`.
- [ ] Decide `mincontrast_nested` fate (§8.10).

**Phase A — round out the homogeneous family (1–2 weeks)**

- [ ] `Kernel` abstraction + `compose_neyman_scott` (§8.1).
- [ ] Add `MaternClusterProcess`, `CauchyClusterProcess`, `StraussProcess`,
      `LGCP`; register; extend `tests/test_registries.py`.
- [ ] `configs/runs/<process>/{pi_multik,vihrs,mincontrast}.yaml` for each.
- [ ] Consolidate the training loop (§8.3) — do this before the sweep so every arm
      is trained identically.
- [ ] Run the sweep; produce the degradation table.

**Phase B — inhomogeneity (2–3 weeks)**

- [ ] Design-spec vector `beta` (§8.2).
- [ ] Wire `InhomThomas`; add inhomogeneous LGCP and non-stationary-`σ` Thomas.
- [ ] Waagepetersen–Guan two-step comparator (via `spatstat`/`rpy2`, or port the
      two-step estimating equations).
- [ ] Covariates-observed / withheld / misspecified experiment.

**Phase C — new filtration (1–2 weeks, overlaps B)**

- [ ] `FiltrationFromDistanceMatrix` + `CubicalFiltration` base classes (§8.4).
- [ ] Alpha-complex filtration (§8.7) — also raises the point budget.
- [ ] KDE-sublevel cubical filtration.
- [ ] Diffusion-distance / commute-time Rips filtration.
- [ ] Report as an ablation axis on the Phase-A/B processes.

**Phase D — real data (2–3 weeks)**

- [ ] `region`/`dimension` passthrough (§8.8) if any target is 3-D.
- [ ] Real-data loader: `(points, window, covariate raster) -> PointCloud`.
- [ ] BCI trees: fit, report `σ` (dispersal range) + uncertainty vs.
      Waagepetersen–Guan 2009, plus covariates-withheld robustness.
- [ ] Secondary dataset (cell biology or `spatstat` classic).

**Cross-cutting (fit in where convenient)**

- [ ] Scale-attention fusion for a single process-agnostic model (§8.6).
- [ ] Move input resolution into experiment classes (§8.5).
- [ ] `topo ⊕ L(r)−r` fusion arm for the orthogonality sub-claim (§2, Option 1).
