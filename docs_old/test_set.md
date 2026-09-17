# The test set

One table, 50,000 patterns, balanced over model family and distance from
complete spatial randomness. It replaces the three separately-named evaluation
sets the generator emits, and it is assembled entirely by selection from
patterns that already exist — nothing was generated, re-simulated or
re-featurized to build it.

Built by `scripts/build_testset.py`, defined in
`src/cloudforger/evaluation/testset.py`, written to `configs/testset.csv`,
scored by `scripts/evaluate_testset.py`.

---

## 1. Why it was rebuilt

The generator produces three evaluation products: a draw from the prior, a grid
of fixed parameter vectors with hundreds of replicates each, and ladders whose
amplitude walks down to CSR. Reporting all three made the headline number depend
on which one a reader happened to look at, and none of the three is a defensible
headline on its own.

The prior draw is the worst offender. Its aggregate is a property of a
log-uniform prior chosen for convenience, not a scientific claim, and that prior
puts **41–52% of its mass below the detection threshold** — where the classical
5% test provably cannot see the departure at all. An aggregate over it is
measuring the floor of the problem more than the method.

The threshold is not a round number chosen after the fact. `departure()` in
`generation/nulls.py` normalises by `c95`, the 95% quantile of the studentised
envelope statistic under CSR, so

> **δ̃ ≤ 1 means exactly: the Monte Carlo 5% test on L(r) − r does not reject CSR.**

That makes δ̃ = 1 a pre-registered, interpretable boundary, and it is the
boundary this test set is cut on.

## 2. Structure

A balanced factorial, **2,500 patterns in every cell**:

| | below | weak | moderate | strong | total |
|---|---|---|---|---|---|
| poisson | 2500 | 2500 | 2500 | 2500 | 10000 |
| thomas | 2500 | 2500 | 2500 | 2500 | 10000 |
| nested | 2500 | 2500 | 2500 | 2500 | 10000 |
| matern2 | 2500 | 2500 | 2500 | 2500 | 10000 |
| lgcp | 2500 | 2500 | 2500 | 2500 | 10000 |

with the strata cut on δ̃:

| stratum | δ̃ | meaning |
|---|---|---|
| `below` | < 1 | the classical test does not reject CSR |
| `weak` | 1 – 2 | just past the detection threshold |
| `moderate` | 2 – 4 | comfortably detectable |
| `strong` | ≥ 4 | the departure is unambiguous |

Equal cells mean the aggregate over the table is an **explicit, stated
weighting** rather than an accident of the prior, and every per-regime number is
read off the same table with `groupby(stratum)` — there is no second set to
consult and no third.

## 3. CSR is a class, not a stratum

This is the one piece of non-obvious semantics in the table.

Poisson patterns have δ̃ = 0 by construction, so a pure δ̃ cut would put every CSR
pattern in `below`. That breaks per-stratum comparison: a stratum with no CSR
cases is a 4-way classification problem whose chance level is 25%, and its
accuracy cannot be set beside a stratum where chance is 20%.

So the CSR quota is held constant instead. Each stratum receives 2,500 CSR
patterns, drawn **disjointly** from the CSR pool, and the class prior stays
uniform at 1/5 everywhere.

> A CSR row's `stratum` names the band it is **paired with**, not a δ̃ band it
> falls in. Its `delta_tilde` is 0.0 and its `is_null` flag is set.

The per-family parameter tasks never see these rows — a parameter run predicts
only its own family, so it covers that family's 10,000 rows and nothing else.

## 4. What is deliberately not balanced

**n̄ is a recorded covariate, not a balanced factor**, and the table does not
pretend otherwise. Reaching a given δ̃ at fixed shape requires points, so δ̃ and
n̄ are structurally entangled: the pool holds only 142 Matérn II patterns at
n̄ ≈ 250 in the `weak` stratum, and 120 at n̄ ≈ 125 in `strong`. No selection can
undo that — it is a property of the models, not of the sampling.

Each cell is therefore filled as evenly across the three n̄ octaves as the
available capacity allows, and the realised composition is recorded per row
(`nbar_band`) and summarised in `configs/testset.summary.json`. For most cells
it is flat at 33/33/33. Where it is not, it is visible rather than assumed away:

| cell | [100,200) | [200,400) | [400,800) |
|---|---|---|---|
| matern2/weak | 47% | **6%** | 47% |
| matern2/strong | **6%** | 33% | 61% |
| nested/weak | 11% | 44% | 44% |

Matérn II is the extreme case and the reason is geometric, not a design
oversight: its hard core is bounded by the packing constraint πτ² < 1, which
caps its attainable δ̃ near 8 no matter how the prior is drawn.

## 5. Replicates are not independent draws

The grid and the ladders carry hundreds of replicates at **one** parameter
vector; the prior draw has a fresh one per row. Pooling them puts repeated
parameter vectors in the table, so a naive standard error over rows would count
replicates as independent draws and understate uncertainty.

Every row therefore carries a `theta_id`, constant within a replicate group and
unique per prior-draw row. The 50,000 rows resolve to **14,041 distinct
parameter vectors**, about 3.6 replicates each. `scripts/evaluate_testset.py`
clusters on it: it averages within each parameter vector, then takes the
standard error over those means. The reported `clustered` error bars are the
honest ones.

## 6. Provenance

Every row keeps a `source_set` naming which product it came from, under
descriptive names rather than letters:

| `source_set` | what it is | rows selected |
|---|---|---|
| `prior` | one fresh θ per pattern, drawn from the prior | 13,794 |
| `grid` | fixed-θ cells, hundreds of replicates each | 17,171 |
| `ladder` | fixed shape, amplitude walking down to CSR | 19,035 |

The three sources become a provenance column, not three answers. Nothing is
discarded — the underlying products remain on disk untouched, and any row not
selected here is still addressable by `case_id`.

## 7. Columns

| column | meaning |
|---|---|
| `case_id` | the pattern's identifier; joins to clouds, diagrams and predictions |
| `family` | generating model |
| `stratum` | `below` / `weak` / `moderate` / `strong` (§2; for CSR see §3) |
| `nbar_band` | n̄ octave, a covariate (§4) |
| `theta_id` | parameter-vector identifier; cluster on this (§5) |
| `is_null` | 1 for CSR patterns |
| `source_set` | `prior` / `grid` / `ladder` (§6) |
| `cell_id`, `level_id`, `index` | the source product's own addressing; −1 where unused |
| `nbar`, `n` | expected and realised point count |
| `delta_tilde` | distance from CSR in units of the 5% critical value |
| `scale`, `amplitude` | the family's length and amplitude coordinates |

## 8. Reproducibility

The selection is deterministic: a fixed seed (20260916), case ids sorted before
sampling, and an allocation rule that spills evenly off full bins. Rebuilding
on any machine reproduces `configs/testset.csv` byte for byte. A cell the pool
cannot fill is **reported, never padded and never silently dropped** — the count
travels with the row group, so a thin cell shows up as a wider error bar rather
than a missing one. At 2,500 per cell every cell fills to quota.

## 9. Worked example

5-way classification, 10 seeds, scored with
`python scripts/evaluate_testset.py <run dirs>`:

| run | overall | below | weak | moderate | strong |
|---|---|---|---|---|---|
| classical union, L+F+G+J | **0.716** | 0.466 | **0.680** | **0.819** | **0.901** |
| classical, L only | 0.685 | **0.474** | 0.659 | 0.766 | 0.841 |
| persistence images | 0.666 | 0.358 | 0.643 | 0.784 | 0.878 |

The single aggregate says persistence loses. The strata say something more
specific and more useful: persistence is **competitive from `moderate` upward**
and beats the L-only baseline there by 1.8 and 3.7 points, and the entire
aggregate deficit is incurred in `below` — the stratum where, by the
pre-registered criterion, the classical test itself cannot reject CSR.

That is the distinction the three-set arrangement could not express, and it is
the reason for reporting strata rather than a number.

---

## Appendix: caveats a reader should carry

**The stratum axis is defined by the competitor's statistic.** δ̃ is built from
L(r) − r, so conditioning on it pins L's signal-to-noise within a stratum while
leaving a topological method's free to vary. This is a mild structural handicap
for persistence, not a neutral axis. `generation/nulls.py` already builds null
tables for G and F; recomputing the strata under δ̃_G and δ̃_F is the robustness
check that shows whether the ordering depends on which statistic defines the
axis, and it should be reported alongside.

**Equal weighting is a choice and should be defended as one.** Weighting strata
equally raises the influence of regimes where signal exists relative to the
prior draw. The defence is that δ̃ = 1 is a pre-registered detectability
threshold, that every stratum including `below` is reported, and that the
weighting is stated rather than inherited. It is not a defence if the aggregate
is quoted without the strata beside it.

**`strong` is a floor, not a ceiling.** δ̃ ≥ 4 is open-ended and the pool's upper
tail is thin and uneven across families (Matérn II cannot exceed ≈ 8 at all).
Finer resolution above 4 — separating 4–8 from 8–16 from beyond — needs either
wider priors or new patterns at high δ̃, and is the natural next extension.

**On-disk paths still carry the old names.** `data/dv3/`, `results/dv3_*` and
`configs/runs/dv3/` are unchanged, because renaming them would invalidate every
recorded result path and every config in the tree. That rename is a separate,
purely mechanical pass; this table and its tooling use neutral names throughout
and do not depend on it.
