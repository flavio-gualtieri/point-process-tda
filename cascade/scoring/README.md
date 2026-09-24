# cascade/scoring — candidate end-to-end scores and their power check

The cascade is meant to be scored end to end by how well its fitted model reproduces the observed
pattern. The first score tried, an energy score with Wasserstein-1 between whole patterns
(`cascade/wasserstein.py`), had no power: it could not tell the true model from CSR even for
strongly clustered clouds. This package holds its replacements and the experiment that decides
between them.

## Separability

Self-contained: imports only `cloudforger` and reads `data/bank/`. Nothing from the rest of
`cascade/` (which is being changed independently), and nothing there imports this. Results and
logs stay in `cascade/scoring/{results,logs}/` (gitignored).

When a score is adopted, its interface into the cascade is small. Both scores are functions of
the observed pattern and a list of simulated patterns (`dss.statistics` + `dss.score`,
`local_kernel.Component(x, ...).scores(sims, ...)`). So `cascade/evaluate.py` only needs to call
one of them in place of `wasserstein.energy_score`.

## Candidates

| file | score | proper | simulations / model | overlap with features |
|---|---|---|---|---|
| `dss.py` | Dawid–Sebastiani (Gaussian synthetic log-likelihood) on 14 geometric statistics: Voronoi, Delaunay, MST, Euler characteristic, H₁ | yes, through the first two moments | 150 | MST, Euler and H₁ are persistent-homology statistics; `no_ph` drops them |
| `local_kernel.py` | kernel score on local configurations (radius R around points and grid centres), R = 1.5 and 4 mean spacings | strictly, for the law of radius-R configurations | 16 | contains K/g/F/G as marginals; the τ sweep (τ → ∞ is the pair-correlation limit) measures what it adds |
| `energy_w.py` | the failed whole-pattern W₁ energy score | yes | 8 | none; baseline only |

Each module's docstring records its choices and how to change them; every knob is in `config.yaml`.

## The power check

```bash
sbatch cascade/scoring/power.sh            # cascade/scoring/config.yaml
```

Test clouds are drawn stratified by δ̃ (15 per family and bin, plus 60 Poisson clouds as the noise
floor). Only two models are scored per cloud: the true model and CSR. `gain = S(csr) − S(oracle)`,
and a usable score has gain clearly > 0 at large δ̃ and ≈ 0 on Poisson clouds. The printed table
gives d′ = mean gain / sd per cloud, which is the score's own separability, and z = the sample's
resolution. For comparison, the failed score had d′ ≈ 0.1–0.5 even at δ̃ > 4.
