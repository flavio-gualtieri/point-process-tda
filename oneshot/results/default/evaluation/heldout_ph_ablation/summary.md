# End-to-end evaluation: default, scored on heldout

Skill = 1 − Σregret / Σgain: 1 = as good as the true model, 0 = no better than CSR; shown only where the gain (true model vs CSR) is resolved. Regret = S(fit) − S(true), ≥ 0 in expectation. Fits come from replicate 0 and are scored on the independent replicate 1.

## kernel score

| | classical | ph | ph_estimators |
|---|---|---|---|
| overall | 0.94 (regret +0.00026 ± 6e-05) | 0.94 (regret +0.00029 ± 6e-05) | 0.95 (regret +0.00022 ± 7e-05) |
| structured clouds | 0.94 (regret +0.00026 ± 6e-05) | 0.94 (regret +0.00029 ± 6e-05) | 0.95 (regret +0.00022 ± 7e-05) |
| poisson | — | — | — |
| thomas | 0.95 (regret +0.00029 ± 0.0002) | 0.93 (regret +0.00045 ± 0.0002) | 0.94 (regret +0.00033 ± 0.0002) |
| nested | 0.92 (regret +0.0012 ± 0.0003) | 0.94 (regret +0.00096 ± 0.0003) | 0.94 (regret +0.00086 ± 0.0003) |
| lgcp | 0.93 (regret +0.00034 ± 0.0003) | 0.86 (regret +0.00068 ± 0.0003) | 0.96 (regret +0.00021 ± 0.0004) |
| matern2 | 1.00 (regret +5.6e-06 ± 3e-05) | 1.00 (regret +5.6e-06 ± 2e-05) | 1.01 (regret -8.7e-06 ± 2e-05) |
| ring | 0.97 (regret +0.00029 ± 0.0002) | 0.96 (regret +0.00041 ± 0.0003) | 0.95 (regret +0.00052 ± 0.0003) |
| matern1 | 0.99 (regret +3.2e-06 ± 3e-05) | 1.00 (regret +1.5e-06 ± 3e-05) | 0.98 (regret +7.5e-06 ± 3e-05) |
| cell | 0.87 (regret +7e-05 ± 5e-05) | 0.96 (regret +2.2e-05 ± 4e-05) | 0.96 (regret +2.2e-05 ± 4e-05) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | classical | ph | ph_estimators |
|---|---|---|---|---|---|
| cell | k 2-2 | 120 | -2.6e-06 ± 3e-05 | — (regret +9.5e-06 ± 4e-05) | — (regret -3.1e-05 ± 3e-05) | — (regret -2.5e-05 ± 3e-05) |
| cell | k 3-4 | 120 | -4e-06 ± 4e-05 | — (regret +4e-05 ± 4e-05) | — (regret +2.9e-06 ± 3e-05) | — (regret -4.6e-06 ± 4e-05) |
| cell | k 5-30 | 120 | +0.0016 ± 0.0001 | 0.90 (regret +0.00016 ± 0.0001) | 0.94 (regret +9.5e-05 ± 0.0001) | 0.94 (regret +9.5e-05 ± 0.0001) |
| lgcp | above 0.9 | 120 | +0.0049 ± 0.0008 | 0.93 (regret +0.00034 ± 0.0003) | 0.86 (regret +0.00068 ± 0.0003) | 0.96 (regret +0.00021 ± 0.0004) |
| matern1 | above 0.9 | 120 | +0.0005 ± 7e-05 | 0.99 (regret +3.2e-06 ± 3e-05) | 1.00 (regret +1.5e-06 ± 3e-05) | 0.98 (regret +7.5e-06 ± 3e-05) |
| matern2 | above 0.9 | 120 | +0.0016 ± 0.0002 | 1.00 (regret +5.6e-06 ± 3e-05) | 1.00 (regret +5.6e-06 ± 2e-05) | 1.01 (regret -8.7e-06 ± 2e-05) |
| nested | above 0.9 | 120 | +0.016 ± 0.002 | 0.92 (regret +0.0012 ± 0.0003) | 0.94 (regret +0.00096 ± 0.0003) | 0.94 (regret +0.00086 ± 0.0003) |
| ring | above 0.9 | 120 | +0.011 ± 0.001 | 0.97 (regret +0.00029 ± 0.0002) | 0.96 (regret +0.00041 ± 0.0003) | 0.95 (regret +0.00052 ± 0.0003) |
| thomas | above 0.9 | 120 | +0.006 ± 0.0008 | 0.95 (regret +0.00029 ± 0.0002) | 0.93 (regret +0.00045 ± 0.0002) | 0.94 (regret +0.00033 ± 0.0002) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| classical | 49 | +0.0002 ± 0.0001 | 1031 | 0.95 (regret +0.00027 ± 6e-05) |
| ph | 44 | +0.00013 ± 8e-05 | 1036 | 0.94 (regret +0.00029 ± 6e-05) |
| ph_estimators | 49 | +0.0002 ± 0.0001 | 1031 | 0.95 (regret +0.00022 ± 7e-05) |

## dss score

| | classical | ph | ph_estimators |
|---|---|---|---|
| overall | 0.99 (regret +0.98 ± 0.1) | 0.99 (regret +0.94 ± 0.2) | 0.99 (regret +0.92 ± 0.2) |
| structured clouds | 0.99 (regret +0.98 ± 0.1) | 0.99 (regret +0.94 ± 0.2) | 0.99 (regret +0.92 ± 0.2) |
| poisson | — | — | — |
| thomas | 0.99 (regret +2.2 ± 0.7) | 0.98 (regret +3.2 ± 1) | 0.98 (regret +3.1 ± 1) |
| nested | 0.99 (regret +1.3 ± 0.5) | 1.00 (regret +1.1 ± 0.4) | 1.00 (regret +0.68 ± 0.3) |
| lgcp | 0.97 (regret +1.9 ± 0.4) | 0.98 (regret +1.8 ± 0.4) | 0.98 (regret +1.7 ± 0.4) |
| matern2 | 0.99 (regret +0.18 ± 0.2) | 0.99 (regret +0.34 ± 0.2) | 0.99 (regret +0.16 ± 0.2) |
| ring | 0.99 (regret +1.5 ± 0.7) | 0.99 (regret +1.2 ± 0.6) | 0.99 (regret +1.7 ± 0.7) |
| matern1 | 0.97 (regret +0.15 ± 0.2) | 1.00 (regret +0.0084 ± 0.2) | 1.00 (regret +0.025 ± 0.2) |
| cell | 0.98 (regret +0.51 ± 0.2) | 0.99 (regret +0.25 ± 0.2) | 0.99 (regret +0.33 ± 0.2) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | classical | ph | ph_estimators |
|---|---|---|---|---|---|
| cell | k 2-2 | 120 | +4.2 ± 0.5 | 0.97 (regret +0.15 ± 0.3) | 1.01 (regret -0.033 ± 0.3) | 1.01 (regret -0.055 ± 0.3) |
| cell | k 3-4 | 120 | +4 ± 0.6 | 0.76 (regret +0.97 ± 0.3) | 0.85 (regret +0.59 ± 0.3) | 0.79 (regret +0.84 ± 0.3) |
| cell | k 5-30 | 120 | +72 ± 6 | 0.99 (regret +0.42 ± 0.3) | 1.00 (regret +0.2 ± 0.3) | 1.00 (regret +0.2 ± 0.3) |
| lgcp | above 0.9 | 120 | +76 ± 1e+01 | 0.97 (regret +1.9 ± 0.4) | 0.98 (regret +1.8 ± 0.4) | 0.98 (regret +1.7 ± 0.4) |
| matern1 | above 0.9 | 120 | +6.2 ± 0.9 | 0.97 (regret +0.15 ± 0.2) | 1.00 (regret +0.0084 ± 0.2) | 1.00 (regret +0.025 ± 0.2) |
| matern2 | above 0.9 | 120 | +27 ± 4 | 0.99 (regret +0.18 ± 0.2) | 0.99 (regret +0.34 ± 0.2) | 0.99 (regret +0.16 ± 0.2) |
| nested | above 0.9 | 120 | +2.2e+02 ± 3e+01 | 0.99 (regret +1.3 ± 0.5) | 1.00 (regret +1.1 ± 0.4) | 1.00 (regret +0.68 ± 0.3) |
| ring | above 0.9 | 120 | +1.9e+02 ± 2e+01 | 0.99 (regret +1.5 ± 0.7) | 0.99 (regret +1.2 ± 0.6) | 0.99 (regret +1.7 ± 0.7) |
| thomas | above 0.9 | 120 | +1.8e+02 ± 4e+01 | 0.99 (regret +2.2 ± 0.7) | 0.98 (regret +3.2 ± 1) | 0.98 (regret +3.1 ± 1) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| classical | 49 | +1 ± 0.4 | 1031 | 0.99 (regret +0.98 ± 0.2) |
| ph | 44 | +0.98 ± 0.4 | 1036 | 0.99 (regret +0.94 ± 0.2) |
| ph_estimators | 49 | +1 ± 0.4 | 1031 | 0.99 (regret +0.92 ± 0.2) |

