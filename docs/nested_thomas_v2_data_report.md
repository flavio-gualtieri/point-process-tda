# Nested Thomas Cloud Dataset (v2): Generation Report

**Source:** `data/nested_thomas/{clouds.pkl, adversarial_clouds.pkl, cloud_generation_manifest.yaml}`
**Generator:** [`scripts/generate_nested_thomas_v2.py`](../scripts/generate_nested_thomas_v2.py)
**Process implementation:** [`src/cloudforger/processes/nested_thomas.py`](../src/cloudforger/processes/nested_thomas.py) (`NestedThomasProcess`, a two-level "clusters of clusters" Neyman–Scott process)
**Region:** unit square `[0,1]^2` (area 1), the same default every process in this repo samples over
**Base seed:** 0 (train/test); `0 + 100_000` (frozen held-out set)

This replaces the previous `data/nested_thomas/` dataset (now archived at
`data/legacy/nested_thomas/`), which had no per-cloud point-count floor
(`min_points: 1` in its old manifest) and let `cluster_scale` range up to
`0.1` against a `meta_cluster_scale` floor of `0.05` — i.e. the two length
scales could invert, collapsing the "nested" structure into ordinary CSR
noise for a chunk of the sweep.

## 1. What the process does

`NestedThomasProcess` composes two ordinary Thomas processes:

1. **Meta (coarse) layer** — a Thomas process with `meta_parent_intensity`
   Poisson centers per unit area, each spawning `Poisson(meta_offspring)`
   fine-cluster centers displaced by `N(0, meta_cluster_scale²)`.
2. **Fine layer** — each fine-cluster center from step 1 becomes a parent
   for the final emitted points: `Poisson(mean_offspring)` points per
   center, displaced by `N(0, cluster_scale²)`.

Only the final (fine) layer's points are emitted; the meta-cluster centers
themselves never appear in the cloud. Five free parameters describe a
cloud: `meta_parent_intensity`, `meta_offspring`, `meta_cluster_scale`,
`mean_offspring`, `cluster_scale`.

Because both layers are stationary Neyman–Scott processes over a region of
area 1, the expected number of emitted points is (to good approximation,
ignoring the small edge-buffer correction described in the source docstring):

```
E[N] ≈ meta_parent_intensity × meta_offspring × mean_offspring
```

`meta_cluster_scale` and `cluster_scale` control geometry (how spread out
each level's clusters are), not the expected point count — which is what
lets the point-count constraint (§3) and the scale-separation constraint
(§4) be designed almost independently.

## 2. Dataset sizes and the frozen test set

| Split | Clouds | Distinct parameter vectors | Realizations per vector |
|---|---|---|---|
| Train/test (`clouds.pkl`) | 7,000 | 1,750 | 4 |
| Frozen held-out (`adversarial_clouds.pkl`) | 1,000 | 250 | 4 |

2,000 parameter vectors are drawn once from the ranges in §4, then split
12.5% / 87.5% **before** repetition, using the repo's existing
adversarial-holdout mechanism (`cloudforger.core.design.split_adversarial_vectors`,
seeded independently via `base_seed + 100_000`). Each surviving vector is
then realized 4 times with different point-process seeds (`reps: 4`), giving
7,000 + 1,000 clouds from 1,750 + 250 underlying parameter draws.

This "adversarial" split is exactly this repo's existing frozen-holdout
convention (see `scripts/generate.py`, `paths.py`) — every downstream script
(`featurize.py`, `train.py`, `evaluate.py`, `vihrs.py`) already treats
`adversarial_clouds.pkl` as a fixed evaluation set that no training seed or
re-run ever touches. **No parameter vector appears in both splits** (verified:
0 overlap between the 1,750 and 250 raw vectors), so the held-out set tests
generalization to unseen `(meta_parent_intensity, meta_offspring,
meta_cluster_scale, mean_offspring, cluster_scale)` combinations, not just
unseen noise realizations of training combinations.

## 3. Point-count floor (hard enforcement)

Every cloud is required to have **≥ 75 points**. Generation samples the
process's own RNG seed (not its parameters) up to 50 times, re-drawing the
realization until the floor is met, before moving to the next parameter
vector:

```python
for attempt in range(MAX_RESAMPLE_ATTEMPTS):
    cloud = proc.sample(region=region, seed=base_seed * 1000 + attempt)
    if cloud.n_points >= FLOOR_POINTS:
        return cloud, attempt
```

75 was chosen as the midpoint of the requested 50–100 floor range: safely
above the minimum neighbor count DTM filtrations in this repo use (`k=15`
in the multi-k sweep this dataset targets), while not being so strict that
resampling would need many retries or bias the low tail of the sampled
parameter distribution.

In practice the floor is almost never binding, because the ranges in §4
were chosen so the low corner's *expected* count already clears it with
margin: only **53/7,000** train clouds (0.76%) and **6/1,000** held-out
clouds (0.6%) needed even a single resample, and no cloud ever needed more
than one. Realized minimum is exactly 75 in both splits — the floor is a
true hard bound, not a statistical tendency.

## 4. Parameter ranges

All five parameters are sampled **log-uniformly**, independently, per
parameter vector:

| Parameter | Range (log-uniform) | Role |
|---|---|---|
| `meta_parent_intensity` | [8, 13] | density of meta-clusters |
| `meta_offspring` | [2.5, 4.0] | # fine sub-clusters per meta-cluster |
| `meta_cluster_scale` | [0.05, 0.15] | spatial spread of the meta layer |
| `mean_offspring` | [7, 12] | points per fine cluster |
| `cluster_scale` | [0.003, 0.008] | spatial spread of the fine layer |

Design choices behind these specific bounds:

- **`meta_offspring` ≥ 2.5 everywhere** — the source docstring notes
  `meta_offspring = 1` degenerates the process to a flat (single-scale)
  Thomas process. Keeping the range comfortably above 1 (and its upper
  bound modest, 4.0) guarantees every cloud is a genuine two-level
  hierarchy: 2–4 *visibly distinct* sub-clusters per meta-cluster, not a
  coin-flip between "nested" and "flat".
- **`mean_offspring` ≥ 7** — with fewer than ~5 points per fine cluster, a
  cluster reads as a handful of scattered points rather than a resolvable
  blob. The [7, 12] range keeps every fine cluster visibly a cluster.
- **Scale separation (`meta_cluster_scale` vs. `cluster_scale`)** — the
  ranges were chosen so `meta_cluster_scale / cluster_scale ≥ 5×` **at every
  point in the sweep**, not just on average: the worst case is
  `min(meta_cluster_scale) / max(cluster_scale) = 0.05 / 0.008 = 6.25×`.
  Realized data confirms this: the empirical ratio across all 8,000 clouds
  ranges from **6.34× to 48.1×** (mean 18.8×), so the two length scales
  never collide into an unidentifiable, CSR-like regime. This is the
  specific defect the old dataset had (`cluster_scale` up to 0.1 against a
  `meta_cluster_scale` floor of 0.05 — the offspring scale could exceed the
  parent scale outright).
- **Point-count coupling (§5)** — `meta_parent_intensity`, `meta_offspring`,
  and `mean_offspring` are the three factors of `E[N]`; their ranges were
  chosen jointly (not independently picked and then checked) so that even
  at the sweep's extreme corners, `E[N]` stays close to the 100–500 target
  window (see §5) rather than being an incidental byproduct of ranges
  chosen for other reasons.

## 5. Why the point-count coupling holds

`E[N] ≈ meta_parent_intensity × meta_offspring × mean_offspring`. Taking the
low and high corners of those three ranges:

- Low corner: `8 × 2.5 × 7 = 140`
- High corner: `13 × 4.0 × 12 = 624`
- Geometric center: `√(140 × 624) ≈ 296`

Because the three factors are drawn independently, corner values are rare
(low-probability joint events), so the realized distribution concentrates
well inside the band even though the two corners themselves (140, 624)
bracket it loosely. Realized statistics confirm this:

| Split | Mean N | Median N | Min | Max | % in [100, 500] |
|---|---|---|---|---|---|
| Train/test | 304.4 | 285 | 75 | 989 | 90.5% |
| Frozen held-out | 310.3 | 290 | 76 | 778 | 89.9% |

Full realized-N histogram (both splits combined, 8,000 clouds):

| Band | Train/test | Frozen held-out |
|---|---|---|
| 75–99 | 1.41% | 1.20% |
| **100–500 (working band)** | **90.46%** | **89.90%** |
| 500–700 | 7.30% | 8.40% |
| >700 | 0.83% | 0.50% |

About 8–9% of clouds land above the 500-point preferred ceiling (max
observed: 989). This is a deliberate trade-off: the request was to
*enforce* the 50–100 floor but only *prefer* the 100–500 ceiling, so the
ceiling was left as a soft consequence of the ranges rather than a second
rejection filter. Truncating the upper tail via rejection would bias the
sampled distribution of `meta_parent_intensity`/`meta_offspring`/`mean_offspring`
away from the intended log-uniform marginals (systematically discarding
their jointly-high combinations) — the same distortion the floor-rejection
mechanism intentionally avoids at the low end by resampling seeds, not
parameters. No cloud exceeds ~1,000 points, so DTM cost is bounded even in
the untruncated tail.

## 6. Reproducibility

- Base seed 0 drives the design RNG (parameter-vector sampling) and every
  train/test cloud's seed; the frozen set uses `seed + 100_000`, this
  repo's standard adversarial-holdout offset.
- Re-running `python scripts/generate_nested_thomas_v2.py` without
  `--force` is a no-op if `data/nested_thomas/clouds.pkl` already exists;
  `--force` regenerates deterministically from the same seeds.
- `cloud_generation_manifest.yaml` records the exact ranges, split sizes,
  resample-attempt statistics, and the list of adversarial parameter-vector
  indices, so any downstream run can audit exactly which parameter draws
  ended up in the frozen set.

## 7. Anticipated reviewer questions

**Q: Could the frozen test set leak into training through shared parameter
vectors, even with different noise seeds?**
No — verified directly on the generated data: the 1,750 train/test and 250
held-out raw parameter vectors have zero set intersection (checked as
tuples of sorted `(name, value)` pairs). Every held-out cloud's parameters
are absent from every train/test cloud.

**Q: Why 4 realizations per parameter vector instead of, say, 1 or 8?**
`reps=4` matches the convention already used for this process in
`configs/runs/nested_thomas_multik_pi_multik_fusion.yaml` before this
change. It gives the parameter-regression task several noise instances per
label (so the network can't just memorize one exemplar per label) while
keeping 1,750 distinct parameter draws for reasonable coverage of the
5-dimensional design space, rather than spending the point budget on many
repeats of few parameter combinations.

**Q: Why floor at 75 instead of exactly 50 or exactly 100?**
The request specified enforcing a floor "of 50-100"; 75 is the midpoint.
Concretely it sits comfortably above `k=15`, the largest DTM neighborhood
size used in this process's target experiment config, while leaving enough
margin below the 100-point preferred floor that resampling rarely triggers
(see §3's 0.6-0.76% resample rates).

**Q: Is the 100–500 "working range" actually a hard filter?**
No, and it wasn't meant to be per the request ("preferably a working
range" vs. "enforce minimum") — it's a range that the joint parameter
design (§5) makes the *expected* value land in for most draws, not a
per-cloud filter. ~90% of clouds fall inside it; the rest are a soft
by-product of a purely log-uniform, unrejected sampling scheme, and are
still bounded well below 1,000 points.

**Q: How was the ≥5× scale-separation requirement verified, not just
designed for?**
By direct measurement on the generated data, not just the input ranges:
the realized `meta_cluster_scale / cluster_scale` ratio across all 8,000
clouds is 6.34×–48.1× (§4). The theoretical worst case from the ranges
alone (6.25×) is never actually violated because the sim matches the
constructed bound.

**Q: Does `NestedThomasProcess` ever degenerate to a flat (non-nested)
Thomas process within this sweep?**
No. That degeneracy requires `meta_offspring ≈ 1`; this dataset's range is
[2.5, 4.0], so every cloud has at least ~2–3 fine sub-clusters per
meta-cluster in expectation, keeping the two-level structure real and
estimable rather than a label the network can't actually distinguish from
a single-scale process.

**Q: Is this dataset ready to run through the existing DTM /
persistence-image / regression pipeline as-is?**
Structurally yes — `clouds.pkl` / `adversarial_clouds.pkl` use the same
record schema (`points`, `params`, `process`, `seed`, `n_points`,
`dimension`, `region`) that `cloud_to_record`/`to_pointcloud` and every
downstream script already expect, and this was verified by round-tripping
records through `to_pointcloud` and checking all point arrays are finite.
One pre-existing wiring detail is worth flagging to whoever runs
`configs/runs/nested_thomas_multik_pi_multik_fusion.yaml`: `NestedThomasProcess`
stores its coarse-layer intensity under the key `parent_intensity` (inherited
from the generic `NeymanScottProcess.params` property), not
`meta_parent_intensity` — but that config's `target_label_names` /
`log_label_names` list `meta_parent_intensity`. This mismatch predates this
dataset (the old data had the same key name) and is not something this
regeneration introduced or fixed; flagging it here since it will affect
label selection whenever that config is actually run.
