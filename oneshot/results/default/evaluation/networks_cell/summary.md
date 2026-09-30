# End-to-end evaluation: pipeline, set networks_cell

Skill = 1 − Σregret / Σgain: 1 = as good as the true model, 0 = no better than CSR; shown only where the gain (true model vs CSR) is resolved. Regret = S(fit) − S(true), ≥ 0 in expectation. Fits come from replicate 0 and are scored on the independent replicate 1. A fit the sampler cannot realise ("failed") is scored as CSR, regret = gain.

## kernel score

| | classical | curves_estimators | fusion_estimators | ph_estimators |
|---|---|---|---|---|
| overall | 0.92 (regret +0.00013 ± 5e-05) | 0.91 (regret +0.00015 ± 5e-05) | 0.97 (regret +4.3e-05 ± 5e-05) | 1.00 (regret +7.9e-07 ± 5e-05) |
| structured clouds | 0.92 (regret +0.00013 ± 5e-05) | 0.91 (regret +0.00015 ± 5e-05) | 0.97 (regret +4.3e-05 ± 5e-05) | 1.00 (regret +7.9e-07 ± 5e-05) |
| poisson | — | — | — | — |
| thomas | — | — | — | — |
| nested | — | — | — | — |
| lgcp | — | — | — | — |
| matern2 | — | — | — | — |
| ring | — | — | — | — |
| matern1 | — | — | — | — |
| cell | 0.92 (regret +0.00013 ± 5e-05) | 0.91 (regret +0.00015 ± 5e-05) | 0.97 (regret +4.3e-05 ± 5e-05) | 1.00 (regret +7.9e-07 ± 5e-05) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | classical | curves_estimators | fusion_estimators | ph_estimators |
|---|---|---|---|---|---|---|
| cell | k 5-30 | 1000 | +0.0016 ± 6e-05 | 0.92 (regret +0.00013 ± 5e-05) | 0.91 (regret +0.00015 ± 5e-05) | 0.97 (regret +4.3e-05 ± 5e-05) | 1.00 (regret +7.9e-07 ± 5e-05) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| classical | 3 | +0.00053 ± 0.0008 | 997 | 0.93 (regret +0.00012 ± 5e-05) |
| curves_estimators | 3 | +0.00053 ± 0.0008 | 997 | 0.91 (regret +0.00015 ± 5e-05) |
| fusion_estimators | 3 | +0.00053 ± 0.0008 | 997 | 0.98 (regret +3.5e-05 ± 5e-05) |
| ph_estimators | 3 | +0.00053 ± 0.0008 | 997 | 1.00 (regret -7.3e-06 ± 5e-05) |

## dss score

| | classical | curves_estimators | fusion_estimators | ph_estimators |
|---|---|---|---|---|
| overall | 0.99 (regret +0.57 ± 0.08) | 0.99 (regret +0.79 ± 0.1) | 1.00 (regret +0.24 ± 0.08) | 1.00 (regret +0.24 ± 0.07) |
| structured clouds | 0.99 (regret +0.57 ± 0.08) | 0.99 (regret +0.79 ± 0.1) | 1.00 (regret +0.24 ± 0.08) | 1.00 (regret +0.24 ± 0.07) |
| poisson | — | — | — | — |
| thomas | — | — | — | — |
| nested | — | — | — | — |
| lgcp | — | — | — | — |
| matern2 | — | — | — | — |
| ring | — | — | — | — |
| matern1 | — | — | — | — |
| cell | 0.99 (regret +0.57 ± 0.08) | 0.99 (regret +0.79 ± 0.1) | 1.00 (regret +0.24 ± 0.08) | 1.00 (regret +0.24 ± 0.07) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | classical | curves_estimators | fusion_estimators | ph_estimators |
|---|---|---|---|---|---|---|
| cell | k 5-30 | 1000 | +70 ± 2 | 0.99 (regret +0.57 ± 0.08) | 0.99 (regret +0.79 ± 0.1) | 1.00 (regret +0.24 ± 0.08) | 1.00 (regret +0.24 ± 0.07) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| classical | 3 | +9.7 ± 1 | 997 | 0.99 (regret +0.54 ± 0.08) |
| curves_estimators | 3 | +9.7 ± 1 | 997 | 0.99 (regret +0.76 ± 0.1) |
| fusion_estimators | 3 | +9.7 ± 1 | 997 | 1.00 (regret +0.21 ± 0.07) |
| ph_estimators | 3 | +9.7 ± 1 | 997 | 1.00 (regret +0.21 ± 0.07) |

