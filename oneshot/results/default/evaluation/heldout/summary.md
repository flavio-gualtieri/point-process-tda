# End-to-end evaluation: default, scored on heldout

Skill = 1 − Σregret / Σgain: 1 = as good as the true model, 0 = no better than CSR; shown only where the gain (true model vs CSR) is resolved. Regret = S(fit) − S(true), ≥ 0 in expectation. Fits come from replicate 0 and are scored on the independent replicate 1.

## kernel score

| | best | classical |
|---|---|---|
| overall | 0.93 (regret +0.00017 ± 4e-05) | 0.91 (regret +0.00023 ± 6e-05) |
| structured clouds | 0.93 (regret +0.00019 ± 5e-05) | 0.91 (regret +0.00025 ± 6e-05) |
| poisson | — (regret +4.3e-05 ± 5e-05) | — (regret +9.4e-05 ± 7e-05) |
| thomas | 0.78 (regret +0.00031 ± 0.0002) | 0.63 (regret +0.00054 ± 0.0003) |
| nested | 0.97 (regret +0.00033 ± 0.0001) | 0.93 (regret +0.0007 ± 0.0002) |
| lgcp | 0.95 (regret +0.00018 ± 0.0001) | 0.98 (regret +6.7e-05 ± 0.0001) |
| matern2 | 0.84 (regret +4.2e-05 ± 5e-05) | 0.77 (regret +6.2e-05 ± 6e-05) |
| ring | 0.93 (regret +0.00026 ± 0.0002) | 0.97 (regret +0.00011 ± 0.0001) |
| matern1 | — (regret +7.9e-05 ± 7e-05) | — (regret +3.2e-05 ± 5e-05) |
| cell | 0.83 (regret +0.00013 ± 9e-05) | 0.69 (regret +0.00024 ± 0.0001) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | best | classical |
|---|---|---|---|---|
| cell | k 2-2 | 20 | -2.3e-05 ± 7e-05 | — (regret +4.1e-05 ± 7e-05) | — (regret -7.5e-06 ± 6e-05) |
| cell | k 3-4 | 20 | +0.00011 ± 0.0001 | — (regret +1.7e-05 ± 0.0001) | — (regret +0.00012 ± 0.0002) |
| cell | k 5-30 | 20 | +0.0022 ± 0.0005 | 0.85 (regret +0.00034 ± 0.0002) | 0.72 (regret +0.00061 ± 0.0002) |
| lgcp | 0.5-0.9 | 20 | +0.00015 ± 0.0001 | — (regret +6.9e-05 ± 0.0002) | — (regret +2.2e-05 ± 0.0002) |
| lgcp | above 0.9 | 20 | +0.0099 ± 0.002 | 0.96 (regret +0.00042 ± 0.0002) | 0.98 (regret +0.00023 ± 0.0003) |
| lgcp | below 0.5 | 20 | -8e-06 ± 5e-05 | — (regret +5.4e-05 ± 8e-05) | — (regret -5.1e-05 ± 8e-05) |
| matern1 | 0.5-0.9 | 20 | -1.3e-05 ± 5e-05 | — (regret +4.9e-05 ± 7e-05) | — (regret +6.4e-05 ± 6e-05) |
| matern1 | above 0.9 | 20 | +0.00011 ± 8e-05 | — (regret -7.5e-05 ± 7e-05) | — (regret -0.00014 ± 7e-05) |
| matern1 | below 0.5 | 20 | +5.7e-05 ± 0.0001 | — (regret +0.00026 ± 0.0002) | — (regret +0.00017 ± 0.0001) |
| matern2 | 0.5-0.9 | 20 | +1.4e-05 ± 7e-05 | — (regret +6.9e-05 ± 9e-05) | — (regret +2e-05 ± 7e-05) |
| matern2 | above 0.9 | 20 | +0.00062 ± 0.0002 | 0.84 (regret +9.9e-05 ± 6e-05) | 0.92 (regret +5.2e-05 ± 5e-05) |
| matern2 | below 0.5 | 20 | +0.00016 ± 0.0001 | — (regret -4.2e-05 ± 0.0001) | — (regret +0.00011 ± 0.0002) |
| nested | 0.5-0.9 | 20 | +7.4e-05 ± 0.0001 | — (regret +0.00023 ± 0.0002) | — (regret +0.00053 ± 0.0002) |
| nested | above 0.9 | 20 | +0.03 ± 0.007 | 0.98 (regret +0.00072 ± 0.0004) | 0.96 (regret +0.0013 ± 0.0006) |
| nested | below 0.5 | 20 | -5.5e-05 ± 7e-05 | — (regret +5e-05 ± 0.0001) | — (regret +0.00029 ± 0.0001) |
| poisson | all | 60 | -1.5e-05 ± 4e-05 | — (regret +4.3e-05 ± 5e-05) | — (regret +9.4e-05 ± 7e-05) |
| ring | 0.5-0.9 | 20 | +0.00021 ± 0.0002 | — (regret +7.2e-05 ± 0.0001) | — (regret +5.3e-05 ± 8e-05) |
| ring | above 0.9 | 20 | +0.012 ± 0.002 | 0.94 (regret +0.00072 ± 0.0005) | 0.98 (regret +0.00027 ± 0.0004) |
| ring | below 0.5 | 20 | +1.9e-05 ± 6e-05 | — (regret -1e-05 ± 0.0001) | — (regret +2.6e-05 ± 9e-05) |
| thomas | 0.5-0.9 | 20 | -7.3e-05 ± 0.0001 | — (regret +8.8e-06 ± 0.0001) | — (regret +9.9e-06 ± 0.0001) |
| thomas | above 0.9 | 20 | +0.0044 ± 0.0009 | 0.80 (regret +0.00089 ± 0.0004) | 0.66 (regret +0.0015 ± 0.0009) |
| thomas | below 0.5 | 20 | +1.5e-05 ± 3e-05 | — (regret +3.1e-05 ± 3e-05) | — (regret +0.00012 ± 6e-05) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| best | 102 | +5e-05 ± 5e-05 | 318 | 0.93 (regret +0.00025 ± 6e-05) |
| classical | 108 | +2.7e-05 ± 5e-05 | 312 | 0.91 (regret +0.00034 ± 8e-05) |

## dss score

| | best | classical |
|---|---|---|
| overall | 0.99 (regret +0.6 ± 0.2) | 0.99 (regret +0.69 ± 0.2) |
| structured clouds | 0.99 (regret +0.64 ± 0.2) | 0.99 (regret +0.69 ± 0.2) |
| poisson | — (regret +0.37 ± 0.2) | — (regret +0.69 ± 0.3) |
| thomas | 0.98 (regret +0.83 ± 0.8) | 0.98 (regret +0.63 ± 0.7) |
| nested | 1.00 (regret +0.63 ± 0.5) | 0.99 (regret +1.2 ± 0.5) |
| lgcp | 0.98 (regret +0.85 ± 0.5) | 0.97 (regret +1.2 ± 0.6) |
| matern2 | 1.08 (regret -0.19 ± 0.4) | 1.10 (regret -0.22 ± 0.4) |
| ring | 0.99 (regret +0.84 ± 0.6) | 0.99 (regret +0.5 ± 0.5) |
| matern1 | -0.01 (regret +0.69 ± 0.3) | 0.45 (regret +0.37 ± 0.3) |
| cell | 0.97 (regret +0.8 ± 0.4) | 0.96 (regret +1.1 ± 0.4) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | best | classical |
|---|---|---|---|---|
| cell | k 2-2 | 20 | +3.9 ± 0.9 | 0.83 (regret +0.66 ± 0.7) | 0.73 (regret +1 ± 0.8) |
| cell | k 3-4 | 20 | +3.6 ± 1 | 0.64 (regret +1.3 ± 0.9) | 0.55 (regret +1.6 ± 0.8) |
| cell | k 5-30 | 20 | +77 ± 2e+01 | 0.99 (regret +0.46 ± 0.4) | 0.99 (regret +0.77 ± 0.4) |
| lgcp | 0.5-0.9 | 20 | +3.3 ± 1 | 0.77 (regret +0.75 ± 0.9) | 0.66 (regret +1.1 ± 1) |
| lgcp | above 0.9 | 20 | +1.2e+02 ± 4e+01 | 0.99 (regret +1.3 ± 1) | 0.98 (regret +2 ± 0.8) |
| lgcp | below 0.5 | 20 | -1.1 ± 0.7 | — (regret +0.48 ± 0.9) | — (regret +0.64 ± 1) |
| matern1 | 0.5-0.9 | 20 | -0.68 ± 0.3 | — (regret +0.28 ± 0.4) | — (regret -0.045 ± 0.5) |
| matern1 | above 0.9 | 20 | +2.2 ± 0.8 | 0.83 (regret +0.38 ± 0.3) | 0.75 (regret +0.54 ± 0.4) |
| matern1 | below 0.5 | 20 | +0.58 ± 0.3 | — (regret +1.4 ± 0.5) | — (regret +0.63 ± 0.4) |
| matern2 | 0.5-0.9 | 20 | -0.78 ± 0.6 | — (regret -0.97 ± 0.9) | — (regret -1.1 ± 0.8) |
| matern2 | above 0.9 | 20 | +7.8 ± 2 | 0.98 (regret +0.18 ± 0.5) | 1.03 (regret -0.24 ± 0.6) |
| matern2 | below 0.5 | 20 | +0.04 ± 0.5 | — (regret +0.21 ± 0.4) | — (regret +0.7 ± 0.6) |
| nested | 0.5-0.9 | 20 | +0.85 ± 2 | — (regret -0.033 ± 0.9) | — (regret +1.2 ± 1) |
| nested | above 0.9 | 20 | +5.7e+02 ± 2e+02 | 1.00 (regret +0.93 ± 1) | 1.00 (regret +1.4 ± 0.7) |
| nested | below 0.5 | 20 | -0.99 ± 0.4 | — (regret +0.98 ± 0.8) | — (regret +0.95 ± 0.8) |
| poisson | all | 60 | +0.036 ± 0.3 | — (regret +0.37 ± 0.2) | — (regret +0.69 ± 0.3) |
| ring | 0.5-0.9 | 20 | +7.7 ± 4 | 0.82 (regret +1.4 ± 1) | 0.95 (regret +0.37 ± 1) |
| ring | above 0.9 | 20 | +2.3e+02 ± 6e+01 | 1.00 (regret +0.57 ± 1) | 1.00 (regret +0.9 ± 1) |
| ring | below 0.5 | 20 | -0.024 ± 0.5 | — (regret +0.55 ± 0.5) | — (regret +0.23 ± 0.5) |
| thomas | 0.5-0.9 | 20 | +1.7 ± 2 | — (regret +0.056 ± 2) | — (regret +0.21 ± 2) |
| thomas | above 0.9 | 20 | +1e+02 ± 4e+01 | 0.98 (regret +2.2 ± 1) | 0.99 (regret +1.3 ± 1) |
| thomas | below 0.5 | 20 | -0.62 ± 0.4 | — (regret +0.2 ± 0.5) | — (regret +0.35 ± 0.5) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| best | 102 | +0.3 ± 0.5 | 318 | 0.99 (regret +0.68 ± 0.2) |
| classical | 108 | +0.28 ± 0.5 | 312 | 0.99 (regret +0.79 ± 0.2) |

