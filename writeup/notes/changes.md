# Changes — 2026-09-02

Reference for how synthetic training clouds are generated, plus today's additions
(`aniso_thomas`, `trend_thomas`, `AnisotropicGaussianKernel`).

---

## 1. How training clouds are generated (recovery reference)

### Pipeline

- 4 stages, disk-artifact-based, each independently re-runnable:
  `generate.py` → `featurize.py` → `train.py` → `evaluate.py`.
- Cloud generation entrypoint: `scripts/generate.py <config.yaml> [--set k=v] [--force]`.
- Reads only the `process:` block of the `RunConfig`. Writes:
  - `data/<process.name>/clouds.pkl`            — train/test records
  - `data/<process.name>/adversarial_clouds.pkl` — held-out records
  - `data/<process.name>/cloud_generation_manifest.yaml`
- `data/` is gitignored; regeneration is deterministic from the config.

### Config → parameter vectors (`data_generation/design.py`)

- `process.design.mode: random` (all current configs). `grid` mode also exists.
- `design.random.n_param_vectors`: N distinct parameter draws (8000 in every
  current config — 7000 train/test + 1000 adversarial after the split).
- `design.random.ranges.<key>: {low, high, scale}` — `scale: log` draws
  log-uniform, `scale: linear` (default) draws uniform. One scalar per key.
- `design.random.derived.<name>: "<expr>"` — evaluated in order, each sees the
  sampled keys plus earlier derived values. Sandbox globals: `sqrt log exp pi`
  only. Cast to `float` (scalars only — no vectors).
- `design.random.constraints: ["<expr>", ...]` — all must be truthy; rejected
  draws are resampled (rejection loop, aborts after 200·N tries).
- `design.reps`: each parameter vector realised `reps` times (all configs: 1).
- `adversarial: {enabled, fraction, seed_offset}` — `fraction` of the N vectors
  (0.125) are pulled out as the adversarial holdout; their clouds use
  `base_seed + seed_offset` (default 100000).

### Parameter vector → PointCloud (`generate_clouds_for_design`)

- Region: unit square, `Box(low=[0,0], high=[1,1])` (`default_region(2)`); the
  repo is 2-D-only for now.
- **Constructor-signature filter**: only param keys matching
  `inspect.signature(<Process>.__init__)` are passed. Reparam keys (`K`, `EN`,
  `c`) and cross-check keys (`c1`) are computed in `derived` for
  sampling/constraints/manifest but dropped before construction.
- Cloud i: `Process(**filtered).sample(region=unit square, seed=base_seed + i)`.
  `base_seed = process.seed` (0 in all configs).
- Record = `cloud_to_record(cloud)` = `{points, params, seed, region, ...}`.
  `params` is `Process.params` — this dict is the **label source**.

### Labels

- `target_label_names` (config): which `params` keys the model regresses.
- `log_label_names` (config): which of those are fit in log space.
- `seeds` (config): model-training seeds (10: 9371–9380), unrelated to cloud seeds.

### Simulation mechanism families

- **Poisson superposition / Neyman–Scott**: `poisson`, `thomas`,
  `matern_cluster`, `nested_thomas`, `aniso_thomas`, `trend_thomas`,
  `inhom_thomas` (offspring stage).
- **Cox / dominated thinning**: `lgcp`.
- **Dependent / independent thinning**: `matern` (hard-core), `inhom_thomas`
  (retention stage).
- **Birth–death Metropolis–Hastings** (no tractable normaliser): `strauss`,
  `lgcp_strauss`.

### Per-process recipes (registry key → mechanism, ctor params, config reparam)

- **`poisson`** — `PoissonProcess(intensity)`. Count ~ Poisson(intensity·|W|),
  locations uniform on W.

- **`matern`** — `MaternHardCoreProcess(parent_intensity, hardcore_radius)`;
  internal `name = "matern_ii"`. Matérn Type II: homogeneous Poisson parents at
  `parent_intensity`; iid uniform marks; keep a point iff no neighbour within
  `hardcore_radius` has a smaller mark. Repulsive. No edge buffer. No config in
  `configs/runs/` (data-only in the report).

- **`thomas`** — `ThomasProcess(parent_intensity, mean_offspring, cluster_scale,
  edge_buffer=None)`. Homogeneous Poisson parents on W expanded by `edge_buffer`
  (default `GaussianKernel.support_radius(1e-4) ≈ 3.72·cluster_scale`);
  per-parent offspring ~ Poisson(`mean_offspring`); displacement ~
  N(0, cluster_scale²·I₂); clip to W. Labels `(parent_intensity, mean_offspring,
  cluster_scale)`; diagnostic `c1 = 2·cluster_scale·√parent_intensity` (overlap
  index ν).
  Config reparam: sample `K` (log), `EN` (log), `c` (log); derive
  `parent_intensity=K`, `mean_offspring=EN/K`, `cluster_scale=c/(2√K)`,
  `c1=2·cluster_scale·√parent_intensity`; constraint `mean_offspring ≥ 2.5`.

- **`matern_cluster`** — `MaternClusterProcess(parent_intensity, mean_offspring,
  cluster_radius, edge_buffer=None)`. As `thomas` but offspring uniform on a disk
  of radius `cluster_radius` (`BallKernel`); `edge_buffer = cluster_radius`
  exactly (compact support, no tail margin). Config: `cluster_radius = c/√K`
  (r = R/2 for a disk, so ν = c/2 matches `thomas` across families);
  `c1 = cluster_radius·√parent_intensity`.

- **`nested_thomas`** — `NestedThomasProcess(meta_parent_intensity,
  meta_offspring, meta_cluster_scale, mean_offspring, cluster_scale,
  edge_buffer=None, meta_edge_buffer=None)`. Two-level NS: an inner `ThomasProcess`
  (meta_*) supplies cluster centres via `parent_sampler`; each centre spawns
  Poisson(`mean_offspring`) offspring ~ N(0, cluster_scale²·I₂). Diagnostics
  `c1` (meta), `c2` (leaf).

- **`inhom_thomas`** — `InhomThomas(parent_intensity, cluster_scale, beta,
  expected_points=800, length_scale=0.25, n_modes=32, covariate_correlation=0.0,
  field_seed=None, ...)`. Homogeneous Poisson parents; Thomas offspring; then
  **independent thinning**: offspring at u kept w.p. `exp(η(u) − η_max)`,
  `η = Z(u)·beta`, `Z` = `len(beta)` standardised RFF Gaussian random fields
  (RBF kernel, `length_scale`, `n_modes`), standardised to mean 0 / unit var over
  a reference sample of the window. `expected_points` inflates the base offspring
  rate by `1/f` (`f = mean exp(η−η_max)`) so the retained count is ~constant across
  the beta sweep (no identifiable intercept). `field_seed=None` → fresh field per
  realisation. `sample()` attaches `covariates` to the PointCloud. Labels
  `(parent_intensity, cluster_scale, beta)`. No config in `configs/runs/`.

- **`lgcp`** — `LGCPProcess(mu, sigma2, s, covariance="exp", n_modes=512,
  field_seed=None, probe_size=8192)`. Cox: `Λ(u) = exp(mu + Y(u))`, `Y` zero-mean
  stationary GRF via RFF, `covariance="exp"` (Matérn-½, Vihrs' choice), marginal
  variance `sigma2`, correlation length `s`. Simulated by thinning a dominating
  homogeneous Poisson at rate `exp(log_lam_max)` where `log_lam_max = max over a
  probe_size uniform sample of (mu + Y) + 0.5`. No parent/offspring, no edge
  buffer (points depend only on the field inside W). Labels `(mu, sigma2, s)`.
  Config ranges (Vihrs 2022 §3.1): `mu∈[4,6]` lin, `sigma2∈[0.05,4]` lin,
  `s∈[0.02,0.10]` log; constraint `mu + sigma2/2 ≤ 7`.

- **`strauss`** — `StraussProcess(beta, gamma, radius, n_steps=None,
  margin_factor=2.0)`. Finite Gibbs, density ∝ `beta^n(x) · gamma^{S_R(x)}` w.r.t.
  unit-rate Poisson; `S_R` = # point pairs within `radius`; `gamma∈[0,1]` =
  inhibition (`gamma=1` or `radius=0` → Poisson). No tractable normaliser →
  **birth–death Metropolis–Hastings** (`_birth_death_mh`, Geyer–Møller 1994):
  init n ~ Poisson(`beta`·|W_sim|) uniform; then `n_steps` proposals, ½ birth
  (uniform location) / ½ death (uniform index); acceptance
  `beta·gamma^t·|W|/(n+1)` for a birth, reciprocal for a death, `t` = neighbours
  within `radius`. `n_steps` default = `max(15000, 40·beta·|W_sim|)` (adaptive;
  pass ~100k for strong interaction / publication). Runs on W expanded by
  `margin_factor·radius` (=2·radius), then clipped. Labels `(beta, gamma, radius)`.
  Config ranges (Vihrs 2022 §3.2, degenerate limits trimmed): `beta∈[200,900]`
  log, `gamma∈[0.05,0.95]` lin, `radius∈[0.005,0.05]` lin.

- **`lgcp_strauss`** — `LGCPStraussProcess(mu, sigma2, s, gamma, radius,
  covariance="exp", n_modes=512, field_seed=None, n_steps=None,
  margin_factor=2.0)`. Strauss inhibition `(gamma, radius)` with spatially varying
  activity `exp(mu + Y(u))` (LGCP field, `sigma2`, `s`). Same birth–death MH,
  `init_intensity = exp(mu + sigma2/2)`. Collapses to `lgcp` if `gamma=1`/`radius=0`,
  to `strauss` if `sigma2=0`. Labels `(mu, sigma2, s, gamma, radius)`.

### Notes

- Cloud RNG: `np.random.default_rng(base_seed + cloud_index)` per cloud;
  adversarial clouds use `base_seed + seed_offset + index`.
- Every current process config uses `n_param_vectors: 8000`, `reps: 1`,
  `adversarial.fraction: 0.125`, `seed_offset: 100000`, `seeds: 9371..9380` —
  training budget held constant across the degradation table.
- `configs/runs/` currently has configs for: `thomas`, `nested_thomas`,
  `matern_cluster`, `lgcp`, `strauss`, `lgcp_strauss`, `aniso_thomas` (new).
  `poisson`, `matern`, `inhom_thomas` have no run config.

---

## 2. Changes made today

### 2.1 `AnisotropicGaussianKernel` — `data_generation/point_processes/kernels.py`

- New `Kernel` subclass. Covariance
  `Σ = R(θ)·diag(σ₁², σ₂²)·R(θ)ᵀ`; sample = scale `N(0, I₂)` by `(σ₁, σ₂)`, then
  rotate by `θ` rad (axis-1 direction anticlockwise from +x).
- **2-D only** — `sample` raises for `dimension != 2`.
- `support_radius(eps) = max(σ₁, σ₂)·Φ⁻¹(1−eps)` (same per-axis tail-mass
  convention as `GaussianKernel`).
- `params` → `cluster_sigma_1`, `cluster_sigma_2`, `cluster_theta`.
- Degeneracies NOT enforced in-kernel — `(σ₁,σ₂,θ) ~ (σ₂,σ₁,θ±π/2)`, `θ ~ θ+π`;
  constrain at the config layer.

### 2.2 `AnisotropicThomasProcess` — registry `aniso_thomas` — `aniso_thomas.py`

- `ThomasProcess` with `GaussianKernel` → `AnisotropicGaussianKernel`. **Parent
  process unchanged** (homogeneous Poisson).
- Ctor: `(parent_intensity, mean_offspring, cluster_scale, cluster_aspect=1.0,
  cluster_theta=0.0, edge_buffer=None)`.
- Kernel axes derived so the geometric-mean scale is fixed:
  `σ₁ = cluster_scale·√cluster_aspect` (long), `σ₂ = cluster_scale/√cluster_aspect`
  (short) ⇒ `√(σ₁σ₂) = cluster_scale`.
- `cluster_aspect == 1.0` → falls back to `GaussianKernel` (kernel bit-identical
  to `thomas`).
- `edge_buffer` auto = `max(σ₁, σ₂)·Φ⁻¹(1−1e-4)` — wider than isotropic for
  elongated clusters, computed automatically.
- Labels exposed: `thomas` set + `cluster_aspect`, `cluster_theta`,
  `cluster_sigma_1`, `cluster_sigma_2`; `c1 = 2·cluster_scale·√parent_intensity`
  (isotropised overlap index, comparable to `thomas`).

### 2.3 `TrendThomasProcess` — registry `trend_thomas` — `trend_thomas.py`

- Thomas with **inhomogeneous parent intensity**:
  `ρ_parent(u) = parent_intensity · exp(beta·(u − c))`, `c` = window centre.
  `parent_intensity` = parent rate at the centre; `beta` = per-axis log-intensity
  gradient (spatstat `kppm(X ~ x + y, "Thomas")` trend). Offspring law unchanged,
  process still second-order stationary within clusters.
- Simulated via a `parent_sampler` doing Lewis–Shedler thinning of a homogeneous
  Poisson at the **exact box-corner** max rate (`_thin_loglinear_poisson`);
  `η_max` exact ⇒ no clip. `beta = 0` → homogeneous fast path.
- Optional anisotropic kernel via `cluster_aspect` (default 1 → `GaussianKernel`)
  and `cluster_theta`.
- Ctor: `(parent_intensity, mean_offspring, cluster_scale, beta=None,
  beta_0=None, beta_1=None, cluster_aspect=1.0, cluster_theta=0.0,
  edge_buffer=None)`. `beta` accepts a sequence OR scalar `beta_0`/`beta_1`
  (stopgap until the design spec samples vectors — ROADMAP §8.2).
- Labels: `parent_intensity` + `mean_offspring`, `cluster_scale`,
  `cluster_aspect`, `cluster_theta`, `cluster_sigma_1`, `cluster_sigma_2`,
  `beta_0`, `beta_1`, `beta_norm`.
- Identifiability: `parent_intensity` (κ₀) trades off with total count. Holding
  `E[N]` fixed across a beta sweep is a config-layer `derived` choice, NOT baked
  in.
- **Not yet used by any config; no clouds generated.**

### 2.4 Registry / exports — `point_processes/__init__.py`

- Registered `aniso_thomas`, `trend_thomas`.
- Exported `AnisotropicThomasProcess`, `TrendThomasProcess`,
  `AnisotropicGaussianKernel`.

### 2.5 Configs — `configs/runs/aniso_thomas/`

- `aniso_thomas_pi_multik_k5k10k15.yaml`, `aniso_thomas_vihrs.yaml` — cloned from
  the `thomas` equivalents.
- **Design delta vs `thomas`:**
  - `K` / `EN` / `c` ranges and the `parent_intensity` / `mean_offspring` /
    `cluster_scale` / `c1` derivations are **identical**. `cluster_scale` is now
    the geometric-mean per-axis std `√(σ₁σ₂)`, so the cluster-size marginal, the
    overlap index ν, the DTM k=5,10,15 filtration, and the PI calibration need no
    retuning.
  - Two NEW axes, **sampled directly under their constructor names** (no
    `derived`): `cluster_aspect ∈ [1, 4]` linear (1.0 in range → nests isotropic
    Thomas); `cluster_theta ∈ [0, π)` linear (ellipse orientation is π-periodic).
  - `target_label_names`: 3 → 5 (`+cluster_aspect`, `+cluster_theta`).
  - `log_label_names`: `+cluster_aspect` only — `cluster_theta` is a circular
    angle, may be ≈ 0.
- **Unchanged:** region, homogeneous parents, `n_param_vectors: 8000`, `reps: 1`,
  adversarial 0.125 / `seed_offset` 100000, seed list, filtration/features/method.

### 2.6 Dataset generated

- `python scripts/generate.py configs/runs/aniso_thomas/aniso_thomas_pi_multik_k5k10k15.yaml`
- `data/aniso_thomas/`: `clouds.pkl` (7000), `adversarial_clouds.pkl` (1000),
  `cloud_generation_manifest.yaml`. ~51 MB, gitignored.
- Sanity checks: point counts median 368, range [70, 1147] (≈ `thomas` envelope);
  `cluster_aspect ∈ [1.00, 4.00]`, `cluster_theta ∈ [0, π)`;
  `|σ₁σ₂ − cluster_scale²| < 2e-18`; `σ₁ ≥ σ₂` always; on an isolated cluster the
  recovered principal-axis ratio tracks `cluster_aspect` and the recovered
  major-axis angle tracks `cluster_theta`.

### 2.7 Tests — `tests/test_registries.py`

- `aniso_thomas`, `trend_thomas` added to the expected-names assertion.
- 6 new tests: anisotropic-kernel shape / orientation / 2-D guard;
  `aniso_thomas` aspect=1 reduction; geometric-mean-scale invariance;
  `trend_thomas` scalar-`beta` fallback ≡ vector `beta`; anisotropic
  geometric-mean check. Full suite: 50 passed.

---

## 3. Known follow-ups

- `cluster_theta`: plain MSE head has a 0/π wraparound seam. Use a
  `(cos 2θ, sin 2θ)` target or an angular loss if θ recovery is a headline number.
  Does not affect generated clouds.
- `trend_thomas`: needs a run config and design-spec vector-`beta` support
  (ROADMAP §8.2) before use.
- `aniso_thomas`: `featurize.py` / `train.py` not yet run.
