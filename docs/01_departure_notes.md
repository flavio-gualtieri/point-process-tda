# 01 — Departure from CSR: statistical notes

Detail moved out of `01_departure.tex`. Notation as there.

## Smoothing in n

- True m₀, s₀, c₉₅ are smooth in n; raw per-n estimates are noisy
- A denser grid alone does not help: leave-one-out interpolation error on the old 20-point grid ≈ Monte Carlo noise
- Fit across n ⇒ borrow strength from neighbouring n; defined at every n, no interpolation rule
- s₀ ∝ 1/n at small r, slower at large r ⇒ fit n·m₀, n·s₀ (slowly varying); the scalar extremum
  moments behave the same way (n·m_lo ≈ 0.83–0.98, n·s_lo ≈ 0.30–0.45 across the grid)
- Cubic B-splines in log n; 8 interior knots (moments), 4 (c₉₅)

## Pipeline details

- **Simulate** (`simulate --jobs 10`, ~55 min on 10 cores)
  - Seed key: (`pilot`, `poisson`, n·10⁵ + rep, PATTERN)
  - Validation key: (`pilot`, `poisson`, i < 10⁵); n from PARAMS, points from PATTERN
  - One float32 file per n (12000 × 512), atomic, resumable; 3.6 GB total
- **Fit** (`fit`)
  - Per n, fit batch: pointwise mean, s.d., Monte Carlo s.e.s, and the scalar extremum moments
  - Weighted least squares per radius: n·m₀, n·s₀ on the spline basis in log n; likewise n·m, n·s
    for each scalar reduction
  - Per n, calibration batch: each reduction's statistic under the *fitted* moments → raw c₉₅, s.e.
  - Weighted spline fit of c₉₅ in log n; then `ext`'s c₉₅ (= the joint scale k) under the fitted
    c₉₅ of its two arms
  - Stores knots + coefficients, plus the per-n raw c₉₅ and its s.e. so lack of fit is auditable
    without re-reading the 3.6 GB pool (100 KB)
- **Validate** (`validate`): rejection rate on validation patterns, overall and in 6 log n bands
- **Use**: `Tables().statistic(curves, n, reduction)` (test), `Tables().delta_tilde(L_minus_r, nbar, reduction)` (δ̃)

## Reductions

- A reduction maps one curve to a number that is 1 at the α boundary, so `S > 1` is rejection
  under any of them. Two kinds, separated by *when* they studentise
- **Pointwise** — `sup` (Myllymäki et al. 2017): `max_r |L̂ − r − m₀|/s₀ / c_α`. Windowed
- **Extremum-first** — `lo`, `hi`: take `min_r φ` / `max_r φ` of the raw curve, then studentise
  that scalar against its own `(m, s)` in n. No division by a vanishing `s₀(r)` ⇒ no window
- **Composite** — `ext` (default): `max(T_lo/c_lo, T_hi/c_hi)`, jointly calibrated. Its own `c_α`
  *is* the joint scale `k ≈ 1.19–1.36` that holds the two-arm rule at α
- One `c_α` spline per reduction (plus two moment splines per scalar reduction) from the same
  stored curves ⇒ adding a reduction is a refit, never a re-simulation
- Every reduction is calibrated at α, but they sit on different scales: a table built on one
  must not be read against another

### Why the window costs the sup its repulsion power

- `s₀(r) → 0` at small r (CSR patterns rarely hold any pair there), so the pointwise ratio is
  unstable and needs `R(n) = {r : expected CSR pairs ≥ min_pairs}`
- `min_pairs` is forced by an atom, not by smoothness: a fraction `e^{−min_pairs}` of CSR patterns
  have no pair below the window edge, where `L̂ − r = −r` exactly for all of them. That shared
  value is an atom of mass `e^{−min_pairs}` in the calibration batch and must sit well below the
  `1 − α` quantile. At `min_pairs = 1` it is 37% and lands on the boundary (ties at exactly `c_α`
  at n = 781, 1588; `quantile_se` collapses and those n get ~25× their weight). At 5 it is 0.7%
- But a Matérn hard core of radius R lives entirely below 2R, and its detection boundary is at
  `n R ≈ 1.44` for `lo` alone and `≈ 1.61` for the two-arm `ext` (measured over the bank), i.e.
  `R/r_min ≈ 0.8–0.9` — *inside* the radii the window discards. The sup therefore draws its line
  just above where hard cores actually become detectable
- Measured power (each reduction at its own 5%, matched n): at n̄ = 200, core = 0.1
  (`R/r_min = 0.79`) the sup gets 0.193 and `lo` alone 0.810; at n̄ = 75, core = 0.2 the sup gets
  0.654 and `lo` 1.000

### Why the extremum-first arms work

- Below a hard core no point has a neighbour, so `K̂ = 0` and `φ(r) = −r` *exactly*: the curve is
  pinned to the line `−r` up to R, and `min_r φ = −R` noiselessly. `T_lo` reads R off the curve
  rather than inferring it from a ratio of two small noisy numbers
- Taking the extremum first also escapes the small-r blow-up: unwindowed, `c_α` for the two-sided
  pointwise sup runs 4.30 → 3.19 over n (spread 1.39) and no smooth fit tracks it, while the
  extremum arms are flat enough to need no window at all
- The arms are complementary, so no per-family choice is needed (and none would be legitimate —
  choosing the statistic from the true model is circular). Each is at or *below* α wherever the
  other fires: on Matérn `hi` reaches 0.003–0.05, on Thomas `lo` reaches 0.007–0.06
- The raw extremum is quantised onto `RADII`, so at a few n the 94–96% band lands on one grid
  step; `quantile_se` widens its band on ties rather than reporting a zero s.e.

## Sample splitting

- Three disjoint sets: fit (m₀, s₀), calibration (c₉₅), validation (size)
- c₉₅ from the same patterns as the moments ⇒ downward-biased critical value
- Size on the calibration batch is 5% by construction ⇒ validation needs fresh patterns
- c₉₅ computed under the *fitted* moments ⇒ calibrates the statistic actually used
- Validation n are random integers ⇒ mostly off-grid, tests the smoother between grid points

## Fit diagnostics

- z = (raw − fit)/s.e.; pure noise ⇒ mean z² ≈ 1 − params/grid points
- s.e. of s₀ uses the fourth moment: √((μ₄ − s⁴)/N)/(2s) (small-r tails non-Gaussian)
- c₉₅ s.e. from order statistics: (q.96 − q.94)/0.02 · √(.95·.05/N), with the band widened on
  ties; agrees with a 400-resample bootstrap to within 10% at every n checked
- `lo` has genuine lack of fit (mean z² = 8.1, |z| up to 10): its c₉₅ is rough in n at the ±8%
  level because the deep-dip tail is a *mixture* of two mechanisms — the hard floor at small r,
  quantised onto `RADII`, and ordinary negative fluctuations at moderate r — whose proportions
  shift with n. It is not Monte Carlo noise and more knots would be chasing a quantisation
  artefact. It does not propagate: validated size is 0.0494 overall and within ±2 s.e. in all six
  n bands, because the statistic's density is flat where the threshold sits. `ext` inherits a
  milder version (mean z² = 2.4) and validates at 0.0501
- Per-n mean z² varies widely (0.07–4.5): residuals strongly correlated across r ⇒ few effective d.o.f. per n; band averages ≈ 1, no trend

## Centring

- K̂ unbiased under the binomial null; L̂ = √(K̂/π) is not (Jensen) ⇒ m₀ ≠ 0
- δ̃ uncentred, test centred ⇒ gap ≤ 0.03 in δ̃ units on R(n)
- Old tables (r ≥ 0.01 for all n): gap 0.16 at n = 50, caused by low-pair radii

## Interpretation of δ̃

- Defined through a classical L test ⇒ the yardstick of that test, not an omnibus distance
- Classical baseline partly calibrated to the axis: state explicitly
- Sup over r: one radius decides; multi-scale models (nested Thomas) summarised by their most
  visible scale. `ext` shares this property — it is also an extremum over r, just of the raw curve
- L only: the departure is whatever L sees, so what F, G or the MST see does not enter
- Under `ext` the zero of the scale MOVES: δ̃ = 0 means "this departure equals CSR's own noise
  floor", not "this is CSR". Poisson sits near −0.7, and a hard core below the floor is negative.
  Values are left unclamped so they stay ordered below the detection boundary
- Hard cores: `min_r φ = −R` exactly, so `lo` reads the core radius off the curve. With
  n·m_lo ≈ 0.87 and n·s_lo ≈ 0.32, δ̃_lo ≈ (nR − 0.87)/(0.32·c₉₅·k), so the whole hard-core axis
  collapses to n·R = core·√n̄, and the boundary is n·R ≈ 1.44 for `lo` alone, ≈ 1.61 for `ext`
  (measured 1.607 over the bank, against 1.779 for `sup`)
- Resolution limit: `RADII` has step 0.25/512, so at n ≳ 500 a core at the detection boundary
  (R ≈ 1.6/n) is only ~1–3 grid steps wide and `lo` loses its edge over `sup` there
- On the sub-r_min stratum δ̃ orders n·R (ρ = 0.98) and NOT R alone (ρ = 0.17, sup 0.08). That is
  correct: detectability depends on the core in mean-spacing units, and n̄ varies across the bank
- Design δ̃ at n̄, not realised n: per-pattern difficulty scatters around the label (most for Cox processes)
- δ̃ = 1 at n̄ does not mean S = 1 on a pattern: S is random around δ̃

## Scope and limits

- Tables valid for n ∈ [20, 2000]; outside raises (simulated pool: n = 44–961)
- W fixed unit square; weight formula requires r ≤ 0.5
- R(n) depends on n: at small n, small-scale structure is invisible by construction
- Curves stored ⇒ p, α, knots changeable without re-simulation
