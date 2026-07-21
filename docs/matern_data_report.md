# Matérn Hard-Core Cloud Dataset: Generation Report

**Source:** `data/matern/{clouds.pkl, adversarial_clouds.pkl, cloud_generation_manifest.yaml}`
**Generator:** [`scripts/generate.py`](../scripts/generate.py) `configs/runs/matern_pi_multik.yaml` (the repo's generic, process-agnostic generator — no custom script needed, unlike `nested_thomas`)
**Process implementation:** [`src/cloudforger/processes/matern.py`](../src/cloudforger/processes/matern.py) (`MaternHardCoreProcess`, a Matérn Type II hard-core process via dependent thinning)
**Region:** unit square `[0,1]^2` (area 1), the same default every process in this repo samples over
**Base seed:** 47 (train/test); `47 + 100_000` (frozen held-out set)

This is a new dataset — there is no prior `data/matern/` to compare against (only
`data/legacy/matern/`, a `rips`-filtration dataset from an older pipeline generation,
unrelated to this DTM/`pi_multik` run). The config this data was generated from,
`configs/runs/matern_pi_multik.yaml`, itself started life as a copy of
`configs/runs/nested_thomas_pi_multik.yaml` with the process name swapped — its
sweep ranges and `target_label_names`/`log_label_names` were still nested_thomas's
before being corrected (§4, §7) as part of the same change that produced this data.

## 1. What the process does

`MaternHardCoreProcess` is a two-step dependent-thinning process:

1. **Parents** — an ordinary Poisson process at rate `parent_intensity` over the
   region, each parent given an i.i.d. `Uniform(0,1)` mark.
2. **Thinning** — a parent survives only if no other parent within
   `hardcore_radius` has a smaller mark. Ties never happen (marks are continuous),
   so every parent is either kept or discarded — this is exactly the classical
   Matérn Type II construction.

Only two free parameters describe a cloud: `parent_intensity` (density before
thinning) and `hardcore_radius` (minimum enforced spacing). Unlike `nested_thomas`
or `thomas`, there is no closed-form-simple `E[N] ≈ product of factors`; the
thinning step has a known closed form instead. Writing `x = parent_intensity · π ·
hardcore_radius²` (expected parent-neighbors within the exclusion disk), the
retained density is:

```
E[N] ≈ β(λ, r) = (1 - exp(-x)) / (π r²),   x = λ π r²
```

As `x → 0` (weak thinning), `β → λ`: the hard-core constraint barely bites and the
pattern looks close to plain Poisson. As `x → ∞` (strong thinning), `β → 1/(π r²)`:
the pattern saturates at the maximum packing density the radius allows, and
`parent_intensity` stops being identifiable from the point pattern (any large-enough
λ produces the same jammed configuration). Keeping the *sweep* away from that
saturating regime — not just the individual corners — is the main design
constraint (§4).

## 2. Dataset sizes and the frozen test set

| Split | Clouds | Distinct parameter vectors | Realizations per vector |
|---|---|---|---|
| Train/test (`clouds.pkl`) | 7,000 | 1,750 | 4 |
| Frozen held-out (`adversarial_clouds.pkl`) | 1,000 | 250 | 4 |

2,000 parameter vectors are drawn once from the ranges in §4, then split 12.5% /
87.5% **before** repetition via `cloudforger.core.design.split_adversarial_vectors`
(seeded independently via `base_seed + 100_000`), then each surviving vector is
realized 4 times with different point-process seeds (`reps: 4`) — the same
`n_param_vectors: 2000` / `reps: 4` / `fraction: 0.125` convention as
`nested_thomas_pi_multik.yaml`, run here through the generic `CloudDesign` /
`scripts/generate.py` path rather than a bespoke script, since `MaternHardCoreProcess`
has no per-cloud floor logic to special-case (§3).

Verified directly on the generated data: **0 overlap** between the 1,750 train/test
and 250 held-out raw `(parent_intensity, hardcore_radius)` vectors — the held-out
split tests generalization to unseen parameter combinations, not just unseen noise
realizations of training combinations.

## 3. Point-count robustness (no floor/resample — by range design instead)

Unlike `generate_nested_thomas_v2.py`, this dataset was built through the plain
`scripts/generate.py` path, which has **no per-cloud point-count floor or
seed-resampling**. Robustness against degenerate (near-empty) clouds instead comes
entirely from how the ranges in §4 were chosen: before finalizing them, the retained
`n_points` distribution was simulated against the config's actual seed (47, 2000
draws) to confirm the low tail already clears the floor a DTM `k=15` filtration
needs (≥16 points) with a comfortable margin, so no resampling mechanism was needed.

Realized statistics confirm the simulation:

| Split | Min | p1 | p5 | Median | Mean | Max | Clouds < 16 pts |
|---|---|---|---|---|---|---|---|
| Train/test | 67 | 86 | 101 | 231 | 269.6 | 811 | 0 / 7,000 |
| Frozen held-out | 59 | 86 | 99 | 220.5 | 255.3 | 797 | 0 / 1,000 |

No cloud in either split falls anywhere close to the `k=15` floor — the smallest
realized cloud (59 points, held-out split) still has 3.7× the minimum a `k=15` DTM
neighborhood needs. Both realized minimums (67, 59) came from a *moderate* `x`
region (`x ≈ 0.43–0.50`, not the theoretical worst-case low-`x`/low-`λ` corner),
i.e. from ordinary Poisson variance around a typical draw rather than from an
extreme parameter combination — expected, since 2,000 independent draws from a
continuous distribution essentially never land exactly on a range's corner.

## 4. Parameter ranges

Both parameters are sampled **log-uniformly**, independently, per parameter vector:

| Parameter | Range (log-uniform) | Role |
|---|---|---|
| `parent_intensity` | [100, 1000] | pre-thinning Poisson rate |
| `hardcore_radius` | [0.012, 0.04] | minimum enforced spacing |

Design choices behind these specific bounds:

- **Original ranges were wrong and got replaced.** The config this replaced had
  `parent_intensity: [8, 13]` and `hardcore_radius: [2.5, 4.0]` — copy-pasted from
  `nested_thomas_pi_multik.yaml`'s `meta_parent_intensity`/`meta_offspring` ranges
  without adjustment. A `hardcore_radius` of 2.5–4.0 is several times the unit
  region's diagonal (√2 ≈ 1.41), which would thin almost every parent point away
  regardless of intensity. The `target_label_names`/`log_label_names` fields had
  the same copy-paste problem (listing `meta_offspring`, `meta_cluster_scale`,
  `mean_offspring`, `cluster_scale` — fields `MaternHardCoreProcess` doesn't have),
  which would have raised `KeyError` in `load_multik_split` the first time training
  ran; both were fixed alongside the ranges.
- **Identifiability spread via `x = λπr²`.** The ranges were chosen jointly (not
  independently picked and then checked) so `x` spans roughly two orders of
  magnitude across the sweep — realized `x` ranges from 0.050 to 4.76 (train/test)
  and 0.064 to 3.34 (held-out), covering everything from weak thinning
  (`x` near 0, pattern close to Poisson, `hardcore_radius` has little visible
  effect) to strong thinning (`x` near 5, pattern close to jammed, where the
  hard-core spacing dominates the visible geometry). This keeps both parameters
  estimable from the point pattern across most of the sweep, rather than having
  one saturate the other at either extreme (§1).
- **Point-count coupling.** `E[N] = β(λ, r)` isn't a simple product of the two
  parameters the way `nested_thomas`'s is a product of three, so the ranges were
  tuned by direct simulation against the theoretical formula (§5) rather than by
  picking corner values and checking a closed form — see §3 for the resulting
  realized distribution.

## 5. Why the point-count design holds

The theoretical retained-density formula `β(λ,r) = (1-exp(-x))/(πr²)`, `x = λπr²`,
predicts each cloud's expected point count from its own `(parent_intensity,
hardcore_radius)`. Comparing this per-cloud prediction against the actual realized
`n_points` (finite-sample Poisson/thinning noise around the expectation):

| Split | Median relative residual | Mean absolute relative residual |
|---|---|---|
| Train/test | 0.89% | 4.37% |
| Frozen held-out | 1.32% | 4.67% |

The close match (median residual under 1.5%, consistent with unbiased Poisson-type
noise around the theoretical mean) confirms the formula — and therefore the range
design built on it — accurately describes the realized data, not just the
intended sampling distribution.

Full realized-`n_points` histogram (bands chosen relative to the `k=15` DTM floor
and a ~100–500 "typical" band, mirroring `nested_thomas_v2_data_report.md`'s bands):

| Band | Train/test | Frozen held-out |
|---|---|---|
| < 15 | 0.00% | 0.00% |
| 15–74 | 0.07% | 0.20% |
| 75–99 | 4.36% | 5.00% |
| **100–499 (typical band)** | **86.30%** | **87.30%** |
| 500–699 | 8.06% | 6.90% |
| ≥ 700 | 1.21% | 0.60% |

No cloud falls below 59 points in either split, and no cloud exceeds 811, so DTM
cost stays bounded at both ends without any rejection or resampling filter — purely
a consequence of the range choice in §4.

## 6. Reproducibility

- Base seed 47 (`process.seed` in `configs/runs/matern_pi_multik.yaml`) drives the
  design RNG (parameter-vector sampling) and every train/test cloud's seed; the
  frozen set uses `seed + 100_000`, this repo's standard adversarial-holdout offset
  (`cloudforger.core.design.DEFAULT_ADVERSARIAL_SEED_OFFSET`).
- Re-running `python scripts/generate.py configs/runs/matern_pi_multik.yaml` without
  `--force` is a no-op if `data/matern/clouds.pkl` already exists; `--force`
  regenerates deterministically from the same seed.
- `cloud_generation_manifest.yaml` records the exact split sizes, the list of
  adversarial parameter-vector indices, and the adversarial vectors themselves
  (`CloudDesign.manifest`) — it does **not** record the sweep ranges or
  per-cloud point-count stats the way `generate_nested_thomas_v2.py`'s
  hand-written manifest does, since the generic `scripts/generate.py` path doesn't
  carry a floor-resample loop to report on. The ranges live only in
  `configs/runs/matern_pi_multik.yaml`; §4 of this report is the range-design
  record for this dataset.

## 7. Anticipated reviewer questions

**Q: Could the frozen test set leak into training through shared parameter
vectors, even with different noise seeds?**
No — verified directly on the generated data: the 1,750 train/test and 250
held-out raw parameter vectors have zero set intersection (checked as rounded
`(parent_intensity, hardcore_radius)` tuples). Every held-out cloud's parameters
are absent from every train/test cloud.

**Q: Why is there no point-count floor/resample mechanism here, unlike
`nested_thomas`?**
`nested_thomas` needed one because its 5-dimensional design space made some
corners genuinely risky (its old dataset had `min_points: 1`). `MaternHardCoreProcess`
has only 2 parameters and a closed-form retained-density formula, so the ranges
could be tuned directly against that formula (§4, §5) to keep the low tail clear
of the DTM floor without needing runtime resampling — the same approach already
used for `thomas_dtm_k5_betti_cnn.yaml`'s ranges, which also runs through the plain
`scripts/generate.py` path with no floor logic.

**Q: Is 100–1000 too wide a range for `parent_intensity` — could some clouds have
thousands of parent points before thinning?**
At the top of the range (`λ=1000`), the *pre-thinning* parent count is expected
~1000, but the emitted (post-thinning) count stays bounded (max realized: 811,
§3) because thinning removes most of them whenever `hardcore_radius` isn't
negligible. Generation cost was checked directly: sampling the worst case
(`λ=1000, r=0.04`) takes ~2.7ms per cloud, so the full 8,000-cloud dataset
generates in well under a minute.

**Q: How was the `x = λπr²` identifiability spread verified, not just designed
for?**
By direct measurement on the generated data, not just the input ranges: realized
`x` spans 0.050–4.757 (train/test) and 0.064–3.342 (held-out) (§4) — comparable
span to the ranges' own theoretical bounds (`x_min ≈ 100·π·0.012² = 0.045`,
`x_max ≈ 1000·π·0.04² = 5.03`), confirming the sweep isn't concentrated near
either the pure-Poisson or fully-jammed extreme.

**Q: Is this dataset ready to run through the existing DTM / persistence-image /
regression pipeline as-is?**
Structurally yes — `clouds.pkl`/`adversarial_clouds.pkl` use the same record
schema (`points`, `params`, `process`, `seed`, `n_points`, `dimension`, `region`)
that `cloud_to_record`/`to_pointcloud` and every downstream script already expect;
verified all point arrays are finite and lie within the `[0,1]^2` region for every
cloud in both splits. One cosmetic detail worth flagging: each record's `process`
field reads `"matern_ii"` (from `MaternHardCoreProcess.name`), not `"matern"` (the
registry key used for `data/matern/`'s directory name and the config's
`process.name`). Nothing in the pipeline compares the two — paths are keyed off
the registry name, not the record's `process` field — so this is metadata-only,
not a functional issue, but it's a source of confusion if someone greps
`clouds.pkl` for `"matern"` and gets no hits.

**Q: Why 4 realizations per parameter vector instead of, say, 1 or 8?**
`reps=4` matches the convention already used for `nested_thomas_pi_multik.yaml`
(and for `nested_thomas` more broadly): several noise instances per label so the
network can't just memorize one exemplar per label, while keeping 1,750 distinct
parameter draws for reasonable coverage of the 2-dimensional design space rather
than spending the point budget on many repeats of few parameter combinations.
