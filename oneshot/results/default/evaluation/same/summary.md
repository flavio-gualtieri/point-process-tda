# End-to-end evaluation: default, scored on same

Skill = 1 − Σregret / Σgain: 1 = as good as the true model, 0 = no better than CSR; shown only where the gain (true model vs CSR) is resolved. Regret = S(fit) − S(true), ≥ 0 in expectation. Fits are scored on the pattern they were estimated from.

## kernel score

| | best | classical |
|---|---|---|
| overall | 1.01 (regret -1.3e-05 ± 5e-05) | 1.03 (regret -6.7e-05 ± 4e-05) |
| structured clouds | 1.01 (regret -2.1e-05 ± 5e-05) | 1.03 (regret -7.8e-05 ± 4e-05) |
| poisson | — (regret +3.8e-05 ± 4e-05) | — (regret +6.5e-06 ± 6e-05) |
| thomas | 1.02 (regret -4.8e-05 ± 0.0001) | 1.01 (regret -2.1e-05 ± 0.0001) |
| nested | 1.01 (regret -6.1e-05 ± 0.0002) | 1.00 (regret -5.8e-06 ± 0.0002) |
| lgcp | 1.08 (regret -0.00027 ± 0.0002) | 1.10 (regret -0.00033 ± 0.0002) |
| matern2 | 1.07 (regret -1.8e-05 ± 5e-05) | 0.89 (regret +2.8e-05 ± 5e-05) |
| ring | 0.96 (regret +0.00014 ± 0.0002) | 1.02 (regret -6.4e-05 ± 0.0001) |
| matern1 | 0.63 (regret +5.7e-05 ± 5e-05) | 1.34 (regret -5.2e-05 ± 5e-05) |
| cell | 0.90 (regret +5.2e-05 ± 7e-05) | 1.19 (regret -9.5e-05 ± 5e-05) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | best | classical |
|---|---|---|---|---|
| cell | k 2-2 | 20 | -4.8e-06 ± 8e-05 | — (regret +3e-05 ± 8e-05) | — (regret -3.6e-06 ± 6e-05) |
| cell | k 3-4 | 20 | -5.9e-05 ± 6e-05 | — (regret -2.5e-05 ± 8e-05) | — (regret -7.7e-05 ± 6e-05) |
| cell | k 5-30 | 20 | +0.0016 ± 0.0004 | 0.90 (regret +0.00015 ± 0.0002) | 1.13 (regret -0.0002 ± 0.0001) |
| lgcp | 0.5-0.9 | 20 | -0.00012 ± 0.0002 | — (regret -0.00028 ± 0.0001) | — (regret -0.00027 ± 0.0001) |
| lgcp | above 0.9 | 20 | +0.01 ± 0.003 | 1.05 (regret -0.00051 ± 0.0005) | 1.07 (regret -0.00067 ± 0.0004) |
| lgcp | below 0.5 | 20 | +5.5e-05 ± 5e-05 | — (regret -1.5e-05 ± 5e-05) | — (regret -7e-05 ± 5e-05) |
| matern1 | 0.5-0.9 | 20 | +2.3e-05 ± 6e-05 | — (regret -1.2e-05 ± 0.0001) | — (regret -6.3e-05 ± 6e-05) |
| matern1 | above 0.9 | 20 | +0.00029 ± 0.0001 | 0.79 (regret +6.3e-05 ± 8e-05) | 0.98 (regret +6.4e-06 ± 8e-05) |
| matern1 | below 0.5 | 20 | +0.00014 ± 9e-05 | — (regret +0.00012 ± 7e-05) | — (regret -9.8e-05 ± 0.0001) |
| matern2 | 0.5-0.9 | 20 | +7.8e-05 ± 5e-05 | — (regret +5.2e-05 ± 5e-05) | — (regret +9.6e-05 ± 5e-05) |
| matern2 | above 0.9 | 20 | +0.00065 ± 0.0002 | 1.19 (regret -0.00012 ± 8e-05) | 1.08 (regret -5.1e-05 ± 8e-05) |
| matern2 | below 0.5 | 20 | +3.7e-05 ± 8e-05 | — (regret +1.6e-05 ± 0.0001) | — (regret +4e-05 ± 0.0001) |
| nested | 0.5-0.9 | 20 | -0.00016 ± 0.0003 | — (regret -0.00023 ± 0.0003) | — (regret -0.00026 ± 0.0002) |
| nested | above 0.9 | 20 | +0.031 ± 0.007 | 1.00 (regret +0.00011 ± 0.0006) | 1.00 (regret +0.00012 ± 0.0004) |
| nested | below 0.5 | 20 | -6e-05 ± 9e-05 | — (regret -5.9e-05 ± 0.0002) | — (regret +0.00012 ± 0.0003) |
| poisson | all | 60 | +5.9e-05 ± 4e-05 | — (regret +3.8e-05 ± 4e-05) | — (regret +6.5e-06 ± 6e-05) |
| ring | 0.5-0.9 | 20 | -0.00013 ± 0.0004 | — (regret -0.00029 ± 0.0003) | — (regret -0.00027 ± 0.0002) |
| ring | above 0.9 | 20 | +0.0099 ± 0.002 | 0.93 (regret +0.0007 ± 0.0004) | 0.99 (regret +9.1e-05 ± 0.0002) |
| ring | below 0.5 | 20 | +0.00014 ± 7e-05 | — (regret +8.5e-06 ± 0.0001) | — (regret -1.3e-05 ± 9e-05) |
| thomas | 0.5-0.9 | 20 | -4.5e-05 ± 8e-05 | — (regret -8.5e-05 ± 7e-05) | — (regret -4.4e-05 ± 6e-05) |
| thomas | above 0.9 | 20 | +0.0057 ± 0.001 | 1.02 (regret -0.00011 ± 0.0003) | 0.99 (regret +3.4e-05 ± 0.0004) |
| thomas | below 0.5 | 20 | +6.7e-05 ± 5e-05 | — (regret +4.8e-05 ± 5e-05) | — (regret -5.5e-05 ± 7e-05) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| best | 102 | -5.5e-05 ± 4e-05 | 318 | 1.00 (regret -9.4e-06 ± 7e-05) |
| classical | 108 | -6.8e-05 ± 5e-05 | 312 | 1.02 (regret -8.1e-05 ± 6e-05) |

## dss score

| | best | classical |
|---|---|---|
| overall | 1.02 (regret -0.72 ± 0.2) | 1.02 (regret -0.69 ± 0.2) |
| structured clouds | 1.02 (regret -0.8 ± 0.2) | 1.02 (regret -0.79 ± 0.3) |
| poisson | — (regret -0.12 ± 0.4) | — (regret +0.042 ± 0.3) |
| thomas | 1.02 (regret -0.75 ± 0.6) | 1.03 (regret -1 ± 0.5) |
| nested | 1.01 (regret -2.2 ± 1) | 1.01 (regret -2.3 ± 2) |
| lgcp | 1.01 (regret -0.43 ± 0.4) | 1.00 (regret -0.11 ± 0.4) |
| matern2 | 1.42 (regret -0.92 ± 0.3) | 1.36 (regret -0.78 ± 0.3) |
| ring | 1.01 (regret -0.75 ± 0.7) | 1.01 (regret -0.94 ± 0.6) |
| matern1 | — (regret -0.25 ± 0.3) | — (regret -0.41 ± 0.3) |
| cell | 1.01 (regret -0.37 ± 0.4) | 1.00 (regret -0.059 ± 0.4) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | best | classical |
|---|---|---|---|---|
| cell | k 2-2 | 20 | +3.8 ± 2 | 1.12 (regret -0.44 ± 0.5) | 0.97 (regret +0.1 ± 0.6) |
| cell | k 3-4 | 20 | +5.2 ± 2 | 1.06 (regret -0.32 ± 1) | 0.93 (regret +0.39 ± 1) |
| cell | k 5-30 | 20 | +76 ± 2e+01 | 1.00 (regret -0.34 ± 0.3) | 1.01 (regret -0.67 ± 0.4) |
| lgcp | 0.5-0.9 | 20 | +3.3 ± 3 | — (regret -1.3 ± 0.7) | — (regret -1.3 ± 0.5) |
| lgcp | above 0.9 | 20 | +1.1e+02 ± 3e+01 | 1.00 (regret +0.18 ± 0.7) | 0.99 (regret +1.3 ± 1) |
| lgcp | below 0.5 | 20 | +0.1 ± 0.4 | — (regret -0.11 ± 0.4) | — (regret -0.28 ± 0.3) |
| matern1 | 0.5-0.9 | 20 | -0.35 ± 0.5 | — (regret -0.059 ± 0.5) | — (regret +0.23 ± 0.5) |
| matern1 | above 0.9 | 20 | +1.7 ± 1 | — (regret -0.18 ± 0.6) | — (regret -0.67 ± 0.7) |
| matern1 | below 0.5 | 20 | -0.88 ± 0.4 | — (regret -0.52 ± 0.5) | — (regret -0.79 ± 0.5) |
| matern2 | 0.5-0.9 | 20 | -0.15 ± 0.5 | — (regret -0.72 ± 0.4) | — (regret -0.51 ± 0.4) |
| matern2 | above 0.9 | 20 | +7.5 ± 3 | 1.11 (regret -0.83 ± 0.4) | 1.07 (regret -0.53 ± 0.6) |
| matern2 | below 0.5 | 20 | -0.8 ± 0.6 | — (regret -1.2 ± 0.5) | — (regret -1.3 ± 0.5) |
| nested | 0.5-0.9 | 20 | +2.2 ± 1 | — (regret -1 ± 0.8) | — (regret -0.18 ± 0.8) |
| nested | above 0.9 | 20 | +5e+02 ± 2e+02 | 1.00 (regret -1.7 ± 1) | 1.00 (regret -0.82 ± 0.8) |
| nested | below 0.5 | 20 | -2 ± 1 | — (regret -3.8 ± 3) | — (regret -5.8 ± 5) |
| poisson | all | 60 | +0.022 ± 0.3 | — (regret -0.12 ± 0.4) | — (regret +0.042 ± 0.3) |
| ring | 0.5-0.9 | 20 | +8.1 ± 4 | — (regret -1.4 ± 1) | — (regret -1.4 ± 0.8) |
| ring | above 0.9 | 20 | +2.7e+02 ± 8e+01 | 1.00 (regret +1.2 ± 1) | 1.00 (regret -0.011 ± 1) |
| ring | below 0.5 | 20 | +0.23 ± 0.7 | — (regret -2 ± 1) | — (regret -1.4 ± 1) |
| thomas | 0.5-0.9 | 20 | +0.31 ± 2 | — (regret -2.5 ± 0.9) | — (regret -2.4 ± 1) |
| thomas | above 0.9 | 20 | +1.1e+02 ± 4e+01 | 1.00 (regret +0.42 ± 1) | 1.00 (regret -0.3 ± 1) |
| thomas | below 0.5 | 20 | +0.14 ± 0.4 | — (regret -0.16 ± 0.5) | — (regret -0.34 ± 0.4) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| best | 102 | -0.65 ± 0.3 | 318 | 1.01 (regret -0.85 ± 0.3) |
| classical | 108 | -0.83 ± 0.3 | 312 | 1.01 (regret -0.78 ± 0.4) |

