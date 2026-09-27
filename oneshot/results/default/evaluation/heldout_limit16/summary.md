# End-to-end evaluation: default, scored on heldout

Skill = 1 − Σregret / Σgain: 1 = as good as the true model, 0 = no better than CSR; shown only where the gain (true model vs CSR) is resolved. Regret = S(fit) − S(true), ≥ 0 in expectation. Fits come from replicate 0 and are scored on the independent replicate 1.

## kernel score

| | best | classical |
|---|---|---|
| overall | — (regret +0.00052 ± 0.0003) | — (regret +0.0001 ± 8e-05) |
| structured clouds | — (regret +0.00059 ± 0.0003) | — (regret +0.0001 ± 9e-05) |
| poisson | — (regret +3.4e-05 ± 6e-05) | — (regret +9e-05 ± 7e-06) |
| thomas | — (regret -3.2e-05 ± 0.0003) | — (regret -3.2e-05 ± 0.0003) |
| nested | — (regret +0.0013 ± 0.001) | — (regret +0.00013 ± 0.0003) |
| lgcp | — (regret -5.6e-06 ± inf) | — (regret +4e-05 ± inf) |
| matern2 | -0.10 (regret +0.0005 ± 7e-05) | 0.13 (regret +0.00039 ± 9e-05) |
| ring | — (regret +0.00088 ± 0.0008) | — (regret +0.00018 ± 8e-05) |
| matern1 | — (regret +8.7e-05 ± inf) | — (regret -8e-05 ± inf) |
| cell | — (regret +8.2e-05 ± 0.0001) | — (regret -6.9e-05 ± 7e-05) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | best | classical |
|---|---|---|---|---|
| cell | k 2-2 | 1 | +3.6e-05 ± inf | — (regret -5.2e-05 ± inf) | — (regret -7.3e-07 ± inf) |
| cell | k 3-4 | 1 | +0.00016 ± inf | — (regret +0.00022 ± inf) | — (regret -0.00014 ± inf) |
| lgcp | below 0.5 | 1 | -4e-05 ± inf | — (regret -5.6e-06 ± inf) | — (regret +4e-05 ± inf) |
| matern1 | above 0.9 | 1 | +3.3e-05 ± inf | — (regret +8.7e-05 ± inf) | — (regret -8e-05 ± inf) |
| matern2 | 0.5-0.9 | 1 | +0.00054 ± inf | — (regret +0.00056 ± inf) | — (regret +0.00048 ± inf) |
| matern2 | above 0.9 | 1 | +0.00036 ± inf | — (regret +0.00043 ± inf) | — (regret +0.0003 ± inf) |
| nested | 0.5-0.9 | 1 | +0.00062 ± inf | — (regret +0.00054 ± inf) | — (regret +0.00028 ± inf) |
| nested | above 0.9 | 2 | +0.085 ± 0.01 | 0.97 (regret +0.0023 ± 0.002) | 1.00 (regret +3.4e-05 ± 0.0007) |
| nested | below 0.5 | 1 | +5.3e-05 ± inf | — (regret +8.3e-05 ± inf) | — (regret +0.00017 ± inf) |
| poisson | all | 2 | -2.6e-06 ± 0.0002 | — (regret +3.4e-05 ± 6e-05) | — (regret +9e-05 ± 7e-06) |
| ring | above 0.9 | 1 | +0.011 ± inf | — (regret +0.0017 ± inf) | — (regret +0.00026 ± inf) |
| ring | below 0.5 | 1 | -9.1e-06 ± inf | — (regret +9.6e-05 ± inf) | — (regret +9.6e-05 ± inf) |
| thomas | below 0.5 | 2 | -9.1e-05 ± 0.0002 | — (regret -3.2e-05 ± 0.0003) | — (regret -3.2e-05 ± 0.0003) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| best | 4 | -7.7e-06 ± 0.0001 | 10 | — (regret +0.00079 ± 0.0004) |
| classical | 3 | -6.4e-05 ± 0.0001 | 11 | — (regret +0.00013 ± 0.0001) |

## dss score

| | best | classical |
|---|---|---|
| overall | — (regret -0.69 ± 0.6) | — (regret -0.84 ± 0.5) |
| structured clouds | — (regret -0.89 ± 0.7) | — (regret -0.97 ± 0.6) |
| poisson | — (regret +0.71 ± 1) | — (regret +0.072 ± 2) |
| thomas | — (regret -3.7 ± 2) | — (regret -3.7 ± 2) |
| nested | — (regret -1.9 ± 1) | — (regret -1.5 ± 1) |
| lgcp | — (regret -0.44 ± inf) | — (regret -0.88 ± inf) |
| matern2 | — (regret -0.76 ± 2) | — (regret -1.1 ± 0.9) |
| ring | — (regret +2.5 ± 2) | — (regret +0.099 ± 0.2) |
| matern1 | — (regret -0.31 ± inf) | — (regret -0.41 ± inf) |
| cell | — (regret -0.088 ± 0.4) | — (regret +1.6 ± 0.1) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | best | classical |
|---|---|---|---|---|
| cell | k 2-2 | 1 | +7 ± inf | — (regret +0.26 ± inf) | — (regret +1.7 ± inf) |
| cell | k 3-4 | 1 | -0.68 ± inf | — (regret -0.44 ± inf) | — (regret +1.5 ± inf) |
| lgcp | below 0.5 | 1 | -3.3 ± inf | — (regret -0.44 ± inf) | — (regret -0.88 ± inf) |
| matern1 | above 0.9 | 1 | +1.4 ± inf | — (regret -0.31 ± inf) | — (regret -0.41 ± inf) |
| matern2 | 0.5-0.9 | 1 | +0.46 ± inf | — (regret -2.4 ± inf) | — (regret -2 ± inf) |
| matern2 | above 0.9 | 1 | +2.7 ± inf | — (regret +0.85 ± inf) | — (regret -0.21 ± inf) |
| nested | 0.5-0.9 | 1 | -3.6 ± inf | — (regret -5.4 ± inf) | — (regret -5 ± inf) |
| nested | above 0.9 | 2 | +2.3e+03 ± 2e+03 | — (regret -0.67 ± 0.8) | — (regret -0.56 ± 0.6) |
| nested | below 0.5 | 1 | -0.032 ± inf | — (regret -0.83 ± inf) | — (regret +0.03 ± inf) |
| poisson | all | 2 | -1.8 ± 3 | — (regret +0.71 ± 1) | — (regret +0.072 ± 2) |
| ring | above 0.9 | 1 | +4.4e+02 ± inf | — (regret +4.7 ± inf) | — (regret -0.11 ± inf) |
| ring | below 0.5 | 1 | -2.7 ± inf | — (regret +0.31 ± inf) | — (regret +0.31 ± inf) |
| thomas | below 0.5 | 2 | -3.2 ± 3 | — (regret -3.7 ± 2) | — (regret -3.7 ± 2) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| best | 4 | -2.5 ± 1 | 10 | — (regret -0.49 ± 0.8) |
| classical | 3 | -3.1 ± 2 | 11 | — (regret -0.59 ± 0.5) |

