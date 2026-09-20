# 01 — Departure from CSR: statistical notes

Detail moved out of `01_departure.tex`. Notation as there.

## Smoothing in n

- True m₀, s₀, c₉₅ are smooth in n; raw per-n estimates are noisy
- A denser grid alone does not help: leave-one-out interpolation error on the old 20-point grid ≈ Monte Carlo noise
- Fit across n ⇒ borrow strength from neighbouring n; defined at every n, no interpolation rule
- s₀ ∝ 1/n at small r, slower at large r ⇒ fit n·m₀, n·s₀ (slowly varying)
- Cubic B-splines in log n; 8 interior knots (moments), 4 (c₉₅)

## Pipeline details

- **Simulate** (`simulate --jobs 10`, ~55 min on 10 cores)
  - Seed key: (`pilot`, `poisson`, n·10⁵ + rep, PATTERN)
  - Validation key: (`pilot`, `poisson`, i < 10⁵); n from PARAMS, points from PATTERN
  - One float32 file per n (12000 × 512), atomic, resumable; 3.6 GB total
- **Fit** (`fit`)
  - Per n, fit batch: pointwise mean, s.d., Monte Carlo s.e.s
  - Weighted least squares per radius: n·m₀, n·s₀ on the spline basis in log n
  - Per n, calibration batch: max statistic under the *fitted* moments → raw c₉₅, s.e.
  - Weighted spline fit of c₉₅ in log n
  - Stores knots + coefficients (100 KB)
- **Validate** (`validate`): rejection rate on validation patterns, overall and in 6 log n bands
- **Use**: `Tables().statistic(curves, n)` (test), `Tables().delta_tilde(L_minus_r, nbar)` (δ̃)

## Sample splitting

- Three disjoint sets: fit (m₀, s₀), calibration (c₉₅), validation (size)
- c₉₅ from the same patterns as the moments ⇒ downward-biased critical value
- Size on the calibration batch is 5% by construction ⇒ validation needs fresh patterns
- c₉₅ computed under the *fitted* moments ⇒ calibrates the statistic actually used
- Validation n are random integers ⇒ mostly off-grid, tests the smoother between grid points

## Fit diagnostics

- z = (raw − fit)/s.e.; pure noise ⇒ mean z² ≈ 1 − params/grid points
- s.e. of s₀ uses the fourth moment: √((μ₄ − s⁴)/N)/(2s) (small-r tails non-Gaussian)
- c₉₅ s.e. from order statistics: (q.96 − q.94)/0.02 · √(.95·.05/N)
- Per-n mean z² varies widely (0.07–4.5): residuals strongly correlated across r ⇒ few effective d.o.f. per n; band averages ≈ 1, no trend

## Centring

- K̂ unbiased under the binomial null; L̂ = √(K̂/π) is not (Jensen) ⇒ m₀ ≠ 0
- δ̃ uncentred, test centred ⇒ gap ≤ 0.03 in δ̃ units on R(n)
- Old tables (r ≥ 0.01 for all n): gap 0.16 at n = 50, caused by low-pair radii

## Interpretation of δ̃

- Defined through a classical L test ⇒ the yardstick of that test, not an omnibus distance
- Classical baseline partly calibrated to the axis: state explicitly
- Sup over r: one radius decides; multi-scale models (nested Thomas) summarised by their most visible scale
- L only: hard cores below r_min are invisible to δ̃ (Matérn II; see the Matérn section of the .tex)
- Design δ̃ at n̄, not realised n: per-pattern difficulty scatters around the label (most for Cox processes)
- δ̃ = 1 at n̄ does not mean S = 1 on a pattern: S is random around δ̃

## Scope and limits

- Tables valid for n ∈ [20, 2000]; outside raises (simulated pool: n = 44–961)
- W fixed unit square; weight formula requires r ≤ 0.5
- R(n) depends on n: at small n, small-scale structure is invisible by construction
- Curves stored ⇒ p, α, knots changeable without re-simulation
