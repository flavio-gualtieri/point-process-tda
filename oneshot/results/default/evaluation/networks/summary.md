# End-to-end evaluation: pipeline, set networks

Skill = 1 − Σregret / Σgain: 1 = as good as the true model, 0 = no better than CSR; shown only where the gain (true model vs CSR) is resolved. Regret = S(fit) − S(true), ≥ 0 in expectation. Fits come from replicate 0 and are scored on the independent replicate 1. A fit the sampler cannot realise ("failed") is scored as CSR, regret = gain.

## kernel score

| | best | classical | curves | fusion | ph |
|---|---|---|---|---|---|
| overall | 0.93 (regret +0.00017 ± 5e-05) | 0.91 (regret +0.00022 ± 6e-05) | 0.92 (regret +0.00019 ± 4e-05) | 0.96 (regret +0.00011 ± 4e-05) | 0.92 (regret +0.00019 ± 5e-05) |
| structured clouds | 0.93 (regret +0.00019 ± 5e-05) | 0.92 (regret +0.00023 ± 6e-05) | 0.93 (regret +0.0002 ± 5e-05) | 0.96 (regret +0.00011 ± 5e-05) | 0.93 (regret +0.00021 ± 6e-05) |
| poisson | — (regret +5.8e-05 ± 5e-05) | — (regret +9.4e-05 ± 7e-05) | — (regret +9.5e-05 ± 7e-05) | — (regret +6.3e-05 ± 6e-05) | — (regret +3.2e-05 ± 5e-05) |
| thomas | 0.78 (regret +0.00032 ± 0.0002) | 0.62 (regret +0.00054 ± 0.0003) | 0.77 (regret +0.00034 ± 0.0002) | 0.87 (regret +0.00019 ± 0.0001) | 0.76 (regret +0.00035 ± 0.0002) |
| nested | 0.97 (regret +0.00029 ± 0.0002) | 0.93 (regret +0.00068 ± 0.0002) | 0.96 (regret +0.00043 ± 0.0002) | 0.95 (regret +0.00046 ± 0.0002) | 0.96 (regret +0.00037 ± 0.0002) |
| lgcp | 0.94 (regret +0.00019 ± 0.0002) | 1.02 (regret -8.4e-05 ± 0.0001) | 0.94 (regret +0.00022 ± 0.0002) | 1.01 (regret -3.1e-05 ± 0.0001) | 0.94 (regret +0.00021 ± 0.0002) |
| matern2 | 0.82 (regret +4.9e-05 ± 6e-05) | 0.77 (regret +6.2e-05 ± 7e-05) | 1.01 (regret -3.4e-06 ± 5e-05) | 0.84 (regret +4.3e-05 ± 6e-05) | 0.92 (regret +2.2e-05 ± 6e-05) |
| ring | 0.94 (regret +0.00025 ± 0.0001) | 0.97 (regret +0.00011 ± 0.0001) | 0.95 (regret +0.0002 ± 0.0001) | 0.98 (regret +9e-05 ± 0.0001) | 0.92 (regret +0.00033 ± 0.0002) |
| matern1 | — (regret +4.5e-05 ± 6e-05) | — (regret +4.4e-05 ± 5e-05) | — (regret +0.0001 ± 6e-05) | — (regret -1.7e-05 ± 5e-05) | — (regret +9.3e-05 ± 6e-05) |
| cell | 0.76 (regret +0.00019 ± 0.0001) | 0.63 (regret +0.00028 ± 0.0001) | 0.81 (regret +0.00015 ± 8e-05) | 0.92 (regret +6.3e-05 ± 8e-05) | 0.88 (regret +9.3e-05 ± 0.0001) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | best | classical | curves | fusion | ph |
|---|---|---|---|---|---|---|---|
| cell | k 2-2 | 20 | -2.3e-05 ± 7e-05 | — (regret -4.3e-05 ± 7e-05) | — (regret -3.3e-05 ± 6e-05) | — (regret -6e-05 ± 6e-05) | — (regret -2e-05 ± 5e-05) | — (regret -2.2e-05 ± 8e-05) |
| cell | k 3-4 | 20 | +0.00011 ± 0.0001 | — (regret +5.4e-05 ± 0.0001) | — (regret +0.00015 ± 0.0002) | — (regret +7.3e-05 ± 0.0001) | — (regret +0.00013 ± 0.0001) | — (regret +9.6e-06 ± 0.0001) |
| cell | k 5-30 | 20 | +0.0022 ± 0.0005 | 0.75 (regret +0.00054 ± 0.0003) | 0.67 (regret +0.00072 ± 0.0003) | 0.81 (regret +0.00042 ± 0.0002) | 0.96 (regret +8.2e-05 ± 0.0002) | 0.87 (regret +0.00029 ± 0.0003) |
| lgcp | 0.5-0.9 | 20 | +0.00015 ± 0.0001 | — (regret +5.8e-05 ± 0.0002) | — (regret -2.3e-05 ± 0.0002) | — (regret -1.8e-05 ± 0.0002) | — (regret +7.7e-06 ± 0.0002) | — (regret -4.7e-05 ± 0.0002) |
| lgcp | above 0.9 | 20 | +0.0099 ± 0.002 | 0.95 (regret +0.00052 ± 0.0006) | 1.02 (regret -0.00016 ± 0.0003) | 0.93 (regret +0.00074 ± 0.0005) | 1.01 (regret -6e-05 ± 0.0004) | 0.94 (regret +0.00062 ± 0.0006) |
| lgcp | below 0.5 | 20 | -8e-06 ± 5e-05 | — (regret -1.3e-05 ± 9e-05) | — (regret -7.1e-05 ± 7e-05) | — (regret -7.1e-05 ± 7e-05) | — (regret -4.2e-05 ± 9e-05) | — (regret +4.6e-05 ± 7e-05) |
| matern1 | 0.5-0.9 | 20 | -1.3e-05 ± 5e-05 | — (regret -2.8e-05 ± 7e-05) | — (regret +8.4e-05 ± 6e-05) | — (regret +0.00021 ± 9e-05) | — (regret -2.9e-05 ± 7e-05) | — (regret +7.4e-05 ± 6e-05) |
| matern1 | above 0.9 | 20 | +0.00011 ± 8e-05 | — (regret -0.00014 ± 6e-05) | — (regret -0.00013 ± 7e-05) | — (regret -9.3e-05 ± 6e-05) | — (regret -0.00013 ± 6e-05) | — (regret -9.5e-05 ± 6e-05) |
| matern1 | below 0.5 | 20 | +5.7e-05 ± 0.0001 | — (regret +0.00031 ± 0.0002) | — (regret +0.00018 ± 0.0001) | — (regret +0.00018 ± 0.0001) | — (regret +0.00011 ± 0.0001) | — (regret +0.0003 ± 0.0002) |
| matern2 | 0.5-0.9 | 20 | +1.4e-05 ± 7e-05 | — (regret +0.00011 ± 0.0001) | — (regret -1.7e-05 ± 6e-05) | — (regret -6.7e-05 ± 7e-05) | — (regret +5.1e-05 ± 0.0001) | — (regret +2.9e-05 ± 9e-05) |
| matern2 | above 0.9 | 20 | +0.00062 ± 0.0002 | 0.96 (regret +2.6e-05 ± 7e-05) | 0.92 (regret +4.7e-05 ± 5e-05) | 0.80 (regret +0.00012 ± 6e-05) | 0.85 (regret +9e-05 ± 7e-05) | 1.02 (regret -1.2e-05 ± 5e-05) |
| matern2 | below 0.5 | 20 | +0.00016 ± 0.0001 | — (regret +1.3e-05 ± 0.0001) | — (regret +0.00015 ± 0.0002) | — (regret -6.5e-05 ± 0.0001) | — (regret -1.4e-05 ± 0.0001) | — (regret +5e-05 ± 0.0001) |
| nested | 0.5-0.9 | 20 | +7.4e-05 ± 0.0001 | — (regret +0.00033 ± 0.0002) | — (regret +0.00055 ± 0.0002) | — (regret +0.0002 ± 0.0002) | — (regret +0.00045 ± 0.0002) | — (regret +0.00029 ± 0.0002) |
| nested | above 0.9 | 20 | +0.03 ± 0.007 | 0.99 (regret +0.00044 ± 0.0005) | 0.96 (regret +0.0012 ± 0.0005) | 0.96 (regret +0.0011 ± 0.0005) | 0.97 (regret +0.00093 ± 0.0004) | 0.97 (regret +0.00075 ± 0.0004) |
| nested | below 0.5 | 20 | -5.5e-05 ± 7e-05 | — (regret +8.3e-05 ± 0.0001) | — (regret +0.00032 ± 0.0001) | — (regret +5.2e-05 ± 8e-05) | — (regret +1.1e-05 ± 8e-05) | — (regret +6.2e-05 ± 0.0001) |
| poisson | all | 60 | -1.5e-05 ± 4e-05 | — (regret +5.8e-05 ± 5e-05) | — (regret +9.4e-05 ± 7e-05) | — (regret +9.5e-05 ± 7e-05) | — (regret +6.3e-05 ± 6e-05) | — (regret +3.2e-05 ± 5e-05) |
| ring | 0.5-0.9 | 20 | +0.00021 ± 0.0002 | — (regret +0.00014 ± 0.0001) | — (regret +5.2e-05 ± 8e-05) | — (regret +6.8e-05 ± 0.0001) | — (regret +7.2e-05 ± 9e-05) | — (regret +5.5e-05 ± 8e-05) |
| ring | above 0.9 | 20 | +0.012 ± 0.002 | 0.95 (regret +0.00061 ± 0.0004) | 0.98 (regret +0.00026 ± 0.0004) | 0.96 (regret +0.00049 ± 0.0002) | 0.98 (regret +0.00026 ± 0.0004) | 0.92 (regret +0.00095 ± 0.0006) |
| ring | below 0.5 | 20 | +1.9e-05 ± 6e-05 | — (regret -8.9e-06 ± 9e-05) | — (regret +2.4e-05 ± 9e-05) | — (regret +2.6e-05 ± 0.0001) | — (regret -5.9e-05 ± 8e-05) | — (regret -2.6e-05 ± 9e-05) |
| thomas | 0.5-0.9 | 20 | -7.3e-05 ± 0.0001 | — (regret -6.3e-05 ± 0.0001) | — (regret -6.6e-05 ± 0.0001) | — (regret -2.9e-06 ± 0.0001) | — (regret -4e-05 ± 0.0001) | — (regret +2e-05 ± 0.0001) |
| thomas | above 0.9 | 20 | +0.0044 ± 0.0009 | 0.78 (regret +0.00096 ± 0.0004) | 0.65 (regret +0.0015 ± 0.0009) | 0.79 (regret +0.00092 ± 0.0006) | 0.89 (regret +0.0005 ± 0.0004) | 0.79 (regret +0.00094 ± 0.0004) |
| thomas | below 0.5 | 20 | +1.5e-05 ± 3e-05 | — (regret +5.6e-05 ± 4e-05) | — (regret +0.00016 ± 7e-05) | — (regret +9.1e-05 ± 6e-05) | — (regret +9.3e-05 ± 5e-05) | — (regret +7.9e-05 ± 5e-05) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| best | 102 | +5e-05 ± 5e-05 | 318 | 0.93 (regret +0.00025 ± 7e-05) |
| classical | 108 | +2.7e-05 ± 5e-05 | 312 | 0.92 (regret +0.00032 ± 8e-05) |
| curves | 125 | +4.8e-05 ± 4e-05 | 295 | 0.93 (regret +0.00028 ± 7e-05) |
| fusion | 123 | +5e-05 ± 4e-05 | 297 | 0.96 (regret +0.00016 ± 6e-05) |
| ph | 102 | +5e-05 ± 5e-05 | 318 | 0.93 (regret +0.00028 ± 7e-05) |

## dss score

| | best | classical | curves | fusion | ph |
|---|---|---|---|---|---|
| overall | 0.99 (regret +0.67 ± 0.2) | 0.99 (regret +0.7 ± 0.2) | 0.99 (regret +0.69 ± 0.2) | 0.99 (regret +0.64 ± 0.2) | 0.99 (regret +0.63 ± 0.2) |
| structured clouds | 0.99 (regret +0.72 ± 0.2) | 0.99 (regret +0.7 ± 0.2) | 0.99 (regret +0.74 ± 0.2) | 0.99 (regret +0.63 ± 0.2) | 0.99 (regret +0.66 ± 0.2) |
| poisson | — (regret +0.34 ± 0.2) | — (regret +0.69 ± 0.3) | — (regret +0.37 ± 0.3) | — (regret +0.72 ± 0.3) | — (regret +0.46 ± 0.2) |
| thomas | 0.98 (regret +0.77 ± 0.7) | 0.98 (regret +0.63 ± 0.7) | 0.98 (regret +0.81 ± 0.7) | 0.97 (regret +0.92 ± 0.7) | 0.98 (regret +0.73 ± 0.7) |
| nested | 1.00 (regret +0.92 ± 0.5) | 0.99 (regret +1.2 ± 0.5) | 0.99 (regret +0.97 ± 0.5) | 0.99 (regret +1.2 ± 0.5) | 0.99 (regret +1.3 ± 0.5) |
| lgcp | 0.98 (regret +0.86 ± 0.5) | 0.97 (regret +1.3 ± 0.6) | 0.97 (regret +1 ± 0.6) | 0.98 (regret +0.68 ± 0.5) | 0.98 (regret +0.73 ± 0.6) |
| matern2 | 1.03 (regret -0.079 ± 0.3) | 1.10 (regret -0.22 ± 0.4) | 0.98 (regret +0.058 ± 0.4) | 1.03 (regret -0.061 ± 0.3) | 1.01 (regret -0.019 ± 0.4) |
| ring | 0.99 (regret +0.97 ± 0.5) | 0.99 (regret +0.5 ± 0.5) | 0.99 (regret +0.56 ± 0.5) | 0.99 (regret +0.5 ± 0.6) | 0.99 (regret +0.46 ± 0.6) |
| matern1 | 0.06 (regret +0.64 ± 0.3) | 0.45 (regret +0.37 ± 0.3) | 0.33 (regret +0.46 ± 0.3) | 0.42 (regret +0.4 ± 0.3) | 0.17 (regret +0.57 ± 0.2) |
| cell | 0.97 (regret +0.97 ± 0.4) | 0.96 (regret +1.1 ± 0.4) | 0.96 (regret +1.3 ± 0.5) | 0.97 (regret +0.77 ± 0.4) | 0.97 (regret +0.85 ± 0.4) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | best | classical | curves | fusion | ph |
|---|---|---|---|---|---|---|---|
| cell | k 2-2 | 20 | +3.9 ± 0.9 | 0.59 (regret +1.6 ± 0.6) | 0.73 (regret +1 ± 0.8) | 0.80 (regret +0.76 ± 0.8) | 0.81 (regret +0.72 ± 0.8) | 0.81 (regret +0.73 ± 0.7) |
| cell | k 3-4 | 20 | +3.6 ± 1 | 0.67 (regret +1.2 ± 0.8) | 0.55 (regret +1.6 ± 0.8) | 0.59 (regret +1.5 ± 1) | 0.71 (regret +1.1 ± 0.9) | 0.63 (regret +1.3 ± 0.9) |
| cell | k 5-30 | 20 | +77 ± 2e+01 | 1.00 (regret +0.14 ± 0.4) | 0.99 (regret +0.77 ± 0.4) | 0.98 (regret +1.6 ± 0.7) | 0.99 (regret +0.55 ± 0.3) | 0.99 (regret +0.46 ± 0.4) |
| lgcp | 0.5-0.9 | 20 | +3.3 ± 1 | 0.80 (regret +0.65 ± 0.9) | 0.66 (regret +1.1 ± 1) | 0.68 (regret +1.1 ± 0.9) | 0.78 (regret +0.72 ± 0.7) | 0.82 (regret +0.6 ± 0.7) |
| lgcp | above 0.9 | 20 | +1.2e+02 ± 4e+01 | 0.99 (regret +1.1 ± 1) | 0.98 (regret +2.1 ± 0.7) | 0.99 (regret +0.94 ± 1) | 1.00 (regret +0.48 ± 0.9) | 0.99 (regret +1.1 ± 1) |
| lgcp | below 0.5 | 20 | -1.1 ± 0.7 | — (regret +0.87 ± 1) | — (regret +0.64 ± 1) | — (regret +1.1 ± 1) | — (regret +0.84 ± 1) | — (regret +0.5 ± 0.9) |
| matern1 | 0.5-0.9 | 20 | -0.68 ± 0.3 | — (regret +0.095 ± 0.6) | — (regret -0.045 ± 0.5) | — (regret -0.026 ± 0.5) | — (regret +0.12 ± 0.6) | — (regret +0.22 ± 0.4) |
| matern1 | above 0.9 | 20 | +2.2 ± 0.8 | 0.65 (regret +0.75 ± 0.5) | 0.75 (regret +0.54 ± 0.4) | 0.76 (regret +0.51 ± 0.4) | 0.71 (regret +0.63 ± 0.4) | 0.81 (regret +0.41 ± 0.3) |
| matern1 | below 0.5 | 20 | +0.58 ± 0.3 | — (regret +1.1 ± 0.5) | — (regret +0.63 ± 0.4) | — (regret +0.9 ± 0.5) | — (regret +0.43 ± 0.4) | — (regret +1.1 ± 0.5) |
| matern2 | 0.5-0.9 | 20 | -0.78 ± 0.6 | — (regret -0.69 ± 0.6) | — (regret -1.1 ± 0.8) | — (regret -0.78 ± 0.9) | — (regret -0.65 ± 0.7) | — (regret -0.86 ± 1) |
| matern2 | above 0.9 | 20 | +7.8 ± 2 | 1.00 (regret +0.037 ± 0.4) | 1.03 (regret -0.24 ± 0.6) | 0.93 (regret +0.56 ± 0.5) | 1.02 (regret -0.17 ± 0.4) | 0.96 (regret +0.31 ± 0.6) |
| matern2 | below 0.5 | 20 | +0.04 ± 0.5 | — (regret +0.41 ± 0.5) | — (regret +0.7 ± 0.6) | — (regret +0.39 ± 0.5) | — (regret +0.64 ± 0.4) | — (regret +0.49 ± 0.6) |
| nested | 0.5-0.9 | 20 | +0.85 ± 2 | — (regret +0.54 ± 0.8) | — (regret +1.2 ± 1) | — (regret +0.47 ± 0.9) | — (regret +1.7 ± 1) | — (regret +1 ± 0.9) |
| nested | above 0.9 | 20 | +5.7e+02 ± 2e+02 | 1.00 (regret +1.1 ± 1) | 1.00 (regret +1.4 ± 0.7) | 1.00 (regret +1.6 ± 1) | 1.00 (regret +0.84 ± 0.8) | 1.00 (regret +1.6 ± 1) |
| nested | below 0.5 | 20 | -0.99 ± 0.4 | — (regret +1.1 ± 0.8) | — (regret +0.95 ± 0.8) | — (regret +0.81 ± 0.7) | — (regret +1 ± 0.7) | — (regret +1.2 ± 0.8) |
| poisson | all | 60 | +0.036 ± 0.3 | — (regret +0.34 ± 0.2) | — (regret +0.69 ± 0.3) | — (regret +0.37 ± 0.3) | — (regret +0.72 ± 0.3) | — (regret +0.46 ± 0.2) |
| ring | 0.5-0.9 | 20 | +7.7 ± 4 | 0.83 (regret +1.3 ± 1) | 0.95 (regret +0.37 ± 1) | 0.93 (regret +0.5 ± 1) | 0.89 (regret +0.86 ± 1) | 0.91 (regret +0.73 ± 1) |
| ring | above 0.9 | 20 | +2.3e+02 ± 6e+01 | 0.99 (regret +1.4 ± 1) | 1.00 (regret +0.9 ± 1) | 1.00 (regret +0.68 ± 1) | 1.00 (regret +0.11 ± 1) | 1.00 (regret +0.33 ± 1) |
| ring | below 0.5 | 20 | -0.024 ± 0.5 | — (regret +0.29 ± 0.5) | — (regret +0.23 ± 0.5) | — (regret +0.51 ± 0.5) | — (regret +0.53 ± 0.5) | — (regret +0.31 ± 0.5) |
| thomas | 0.5-0.9 | 20 | +1.7 ± 2 | — (regret +0.3 ± 2) | — (regret +0.21 ± 2) | — (regret -0.2 ± 2) | — (regret +0.41 ± 2) | — (regret -0.31 ± 2) |
| thomas | above 0.9 | 20 | +1e+02 ± 4e+01 | 0.98 (regret +1.9 ± 1) | 0.99 (regret +1.3 ± 1) | 0.98 (regret +2.3 ± 1) | 0.98 (regret +1.7 ± 1) | 0.98 (regret +1.8 ± 1) |
| thomas | below 0.5 | 20 | -0.62 ± 0.4 | — (regret +0.15 ± 0.5) | — (regret +0.35 ± 0.5) | — (regret +0.29 ± 0.5) | — (regret +0.68 ± 0.7) | — (regret +0.75 ± 0.6) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| best | 102 | +0.3 ± 0.5 | 318 | 0.99 (regret +0.79 ± 0.2) |
| classical | 108 | +0.28 ± 0.5 | 312 | 0.99 (regret +0.8 ± 0.2) |
| curves | 125 | +0.28 ± 0.4 | 295 | 0.99 (regret +0.9 ± 0.2) |
| fusion | 123 | +0.4 ± 0.4 | 297 | 0.99 (regret +0.64 ± 0.2) |
| ph | 102 | +0.3 ± 0.5 | 318 | 0.99 (regret +0.71 ± 0.2) |

