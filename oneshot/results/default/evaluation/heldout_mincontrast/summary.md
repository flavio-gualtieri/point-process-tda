# End-to-end evaluation: default, scored on heldout

Skill = 1 − Σregret / Σgain: 1 = as good as the true model, 0 = no better than CSR; shown only where the gain (true model vs CSR) is resolved. Regret = S(fit) − S(true), ≥ 0 in expectation. Fits come from replicate 0 and are scored on the independent replicate 1. A fit the sampler cannot realise ("failed") is scored as CSR, regret = gain.

## kernel score

| | classical | mincontrast | oracle_mincontrast | oracle_ph | ph |
|---|---|---|---|---|---|
| overall | 0.91 (regret +0.00023 ± 6e-05) | 0.84 (regret +0.00039 ± 8e-05) | 0.84 (regret +0.00039 ± 0.0001) | 0.94 (regret +0.00016 ± 4e-05) | 0.92 (regret +0.00019 ± 4e-05) |
| structured clouds | 0.91 (regret +0.00025 ± 6e-05) | 0.85 (regret +0.00043 ± 9e-05) | 0.84 (regret +0.00044 ± 0.0001) | 0.94 (regret +0.00018 ± 5e-05) | 0.92 (regret +0.00021 ± 5e-05) |
| poisson | — (regret +9.5e-05 ± 7e-05) | — (regret +0.00013 ± 8e-05) | — (regret +2.8e-05 ± 4e-05) | — (regret +2.8e-05 ± 4e-05) | — (regret +2.6e-05 ± 5e-05) |
| thomas | 0.64 (regret +0.00053 ± 0.0003) | 0.60 (regret +0.00058 ± 0.0004) | 0.13 (regret +0.0013 ± 0.0008) | 0.82 (regret +0.00026 ± 0.0002) | 0.76 (regret +0.00035 ± 0.0002) |
| nested | 0.93 (regret +0.00067 ± 0.0002) | 0.93 (regret +0.00066 ± 0.0002) | 0.93 (regret +0.00072 ± 0.0004) | 0.96 (regret +0.00044 ± 0.0002) | 0.96 (regret +0.00044 ± 0.0002) |
| lgcp | 0.98 (regret +6.1e-05 ± 0.0001) | 0.91 (regret +0.00029 ± 0.0002) | 0.96 (regret +0.00012 ± 0.0002) | 0.93 (regret +0.00022 ± 0.0001) | 0.95 (regret +0.00017 ± 0.0001) |
| matern2 | 0.73 (regret +7.3e-05 ± 7e-05) | 0.42 (regret +0.00015 ± 7e-05) | 0.72 (regret +7.4e-05 ± 6e-05) | 0.82 (regret +4.8e-05 ± 7e-05) | 0.87 (regret +3.6e-05 ± 6e-05) |
| ring | 0.98 (regret +8.9e-05 ± 0.0001) | 0.95 (regret +0.00022 ± 0.0001) | 0.93 (regret +0.00028 ± 0.0002) | 0.95 (regret +0.00018 ± 0.0002) | 0.93 (regret +0.0003 ± 0.0002) |
| matern1 | — (regret +4.4e-05 ± 5e-05) | — (regret +9.5e-05 ± 5e-05) | — (regret +2.5e-05 ± 5e-05) | — (regret +1.9e-05 ± 6e-05) | — (regret +0.00011 ± 6e-05) |
| cell | 0.63 (regret +0.00028 ± 0.0001) | -0.30 (regret +0.00099 ± 0.0003) | 0.21 (regret +0.0006 ± 0.0001) | 0.91 (regret +7e-05 ± 0.0001) | 0.88 (regret +9.3e-05 ± 0.0001) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | classical | mincontrast | oracle_mincontrast | oracle_ph | ph |
|---|---|---|---|---|---|---|---|
| cell | k 2-2 | 20 | -2.3e-05 ± 7e-05 | — (regret +1.1e-05 ± 6e-05) | — (regret +6.4e-05 ± 7e-05) | — (regret +0.00056 ± 0.0001) | — (regret -4.8e-05 ± 7e-05) | — (regret -1.2e-05 ± 9e-05) |
| cell | k 3-4 | 20 | +0.00011 ± 0.0001 | — (regret +0.00017 ± 0.0002) | — (regret +0.00044 ± 0.0002) | — (regret +0.00025 ± 0.0001) | — (regret +2.1e-06 ± 0.0001) | — (regret +3.6e-05 ± 0.0001) |
| cell | k 5-30 | 20 | +0.0022 ± 0.0005 | 0.70 (regret +0.00067 ± 0.0003) | -0.12 (regret +0.0025 ± 0.0007) | 0.55 (regret +0.00099 ± 0.0004) | 0.88 (regret +0.00025 ± 0.0003) | 0.88 (regret +0.00025 ± 0.0003) |
| lgcp | 0.5-0.9 | 20 | +0.00015 ± 0.0001 | — (regret +1.4e-05 ± 0.0002) | — (regret -2.8e-05 ± 0.0001) | — (regret -0.00023 ± 0.0002) | — (regret +7.2e-05 ± 0.0002) | — (regret -6.4e-05 ± 0.0002) |
| lgcp | above 0.9 | 20 | +0.0099 ± 0.002 | 0.98 (regret +0.00023 ± 0.0003) | 0.91 (regret +0.00088 ± 0.0006) | 0.94 (regret +0.00056 ± 0.0006) | 0.95 (regret +0.00048 ± 0.0003) | 0.95 (regret +0.00053 ± 0.0003) |
| lgcp | below 0.5 | 20 | -8e-06 ± 5e-05 | — (regret -6e-05 ± 9e-05) | — (regret +1.5e-05 ± 7e-05) | — (regret +3.4e-05 ± 7e-05) | — (regret +0.00012 ± 0.0001) | — (regret +5e-05 ± 7e-05) |
| matern1 | 0.5-0.9 | 20 | -1.3e-05 ± 5e-05 | — (regret +0.00011 ± 6e-05) | — (regret +0.00018 ± 6e-05) | — (regret +0.00011 ± 4e-05) | — (regret +7.3e-05 ± 6e-05) | — (regret +8.3e-05 ± 5e-05) |
| matern1 | above 0.9 | 20 | +0.00011 ± 8e-05 | — (regret -0.00015 ± 7e-05) | — (regret -0.00011 ± 9e-05) | — (regret -0.0001 ± 6e-05) | — (regret -0.00012 ± 5e-05) | — (regret -6.4e-05 ± 6e-05) |
| matern1 | below 0.5 | 20 | +5.7e-05 ± 0.0001 | — (regret +0.00018 ± 0.0001) | — (regret +0.00021 ± 0.0001) | — (regret +7.2e-05 ± 0.0001) | — (regret +9.9e-05 ± 0.0002) | — (regret +0.00032 ± 0.0002) |
| matern2 | 0.5-0.9 | 20 | +1.4e-05 ± 7e-05 | — (regret +7.9e-06 ± 7e-05) | — (regret +0.00017 ± 0.0001) | — (regret -6.7e-05 ± 8e-05) | — (regret +4.3e-05 ± 0.0001) | — (regret +3.6e-05 ± 9e-05) |
| matern2 | above 0.9 | 20 | +0.00062 ± 0.0002 | 0.94 (regret +3.7e-05 ± 5e-05) | 0.81 (regret +0.00012 ± 6e-05) | 0.68 (regret +0.0002 ± 7e-05) | 0.90 (regret +6.3e-05 ± 8e-05) | 0.98 (regret +1.3e-05 ± 5e-05) |
| matern2 | below 0.5 | 20 | +0.00016 ± 0.0001 | — (regret +0.00017 ± 0.0002) | — (regret +0.00018 ± 0.0002) | — (regret +9.1e-05 ± 0.0001) | — (regret +3.9e-05 ± 0.0002) | — (regret +5.8e-05 ± 0.0001) |
| nested | 0.5-0.9 | 20 | +7.4e-05 ± 0.0001 | — (regret +0.00051 ± 0.0002) | — (regret +0.00054 ± 0.0003) | — (regret +0.00029 ± 0.0002) | — (regret +0.00013 ± 0.0002) | — (regret +0.00026 ± 0.0002) |
| nested | above 0.9 | 20 | +0.03 ± 0.007 | 0.96 (regret +0.0012 ± 0.0006) | 0.96 (regret +0.0013 ± 0.0006) | 0.95 (regret +0.0016 ± 0.001) | 0.96 (regret +0.0012 ± 0.0005) | 0.97 (regret +0.001 ± 0.0005) |
| nested | below 0.5 | 20 | -5.5e-05 ± 7e-05 | — (regret +0.00028 ± 0.0001) | — (regret +0.00013 ± 7e-05) | — (regret +0.00025 ± 0.0002) | — (regret +1.7e-05 ± 8e-05) | — (regret +4.2e-05 ± 0.0001) |
| poisson | all | 60 | -1.5e-05 ± 4e-05 | — (regret +9.5e-05 ± 7e-05) | — (regret +0.00013 ± 8e-05) | — (regret +2.8e-05 ± 4e-05) | — (regret +2.8e-05 ± 4e-05) | — (regret +2.6e-05 ± 5e-05) |
| ring | 0.5-0.9 | 20 | +0.00021 ± 0.0002 | — (regret +5e-05 ± 8e-05) | — (regret +0.00029 ± 0.0001) | — (regret +0.00016 ± 9e-05) | — (regret +5.5e-05 ± 0.0001) | — (regret +8.4e-05 ± 8e-05) |
| ring | above 0.9 | 20 | +0.012 ± 0.002 | 0.98 (regret +0.00018 ± 0.0004) | 0.97 (regret +0.0004 ± 0.0004) | 0.94 (regret +0.00065 ± 0.0005) | 0.95 (regret +0.00057 ± 0.0005) | 0.92 (regret +0.00087 ± 0.0005) |
| ring | below 0.5 | 20 | +1.9e-05 ± 6e-05 | — (regret +3.3e-05 ± 9e-05) | — (regret -4e-05 ± 0.0001) | — (regret +1.8e-05 ± 0.0002) | — (regret -8.9e-05 ± 0.0001) | — (regret -7.1e-05 ± 9e-05) |
| thomas | 0.5-0.9 | 20 | -7.3e-05 ± 0.0001 | — (regret -5.3e-05 ± 0.0001) | — (regret -5e-06 ± 0.0001) | — (regret +5.4e-05 ± 0.0001) | — (regret -7.7e-05 ± 0.0001) | — (regret +1.3e-05 ± 0.0001) |
| thomas | above 0.9 | 20 | +0.0044 ± 0.0009 | 0.65 (regret +0.0015 ± 0.0009) | 0.63 (regret +0.0016 ± 0.001) | 0.17 (regret +0.0037 ± 0.002) | 0.83 (regret +0.00074 ± 0.0004) | 0.78 (regret +0.00096 ± 0.0004) |
| thomas | below 0.5 | 20 | +1.5e-05 ± 3e-05 | — (regret +0.0001 ± 6e-05) | — (regret +0.00011 ± 5e-05) | — (regret +6.5e-05 ± 5e-05) | — (regret +0.0001 ± 6e-05) | — (regret +8.6e-05 ± 5e-05) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| classical | 108 | +2.7e-05 ± 5e-05 | 312 | 0.91 (regret +0.00034 ± 8e-05) |
| mincontrast | 122 | +6.7e-05 ± 5e-05 | 298 | 0.86 (regret +0.00056 ± 0.0001) |
| oracle_mincontrast | 0 | — | 420 | 0.84 (regret +0.00044 ± 0.0001) |
| oracle_ph | 0 | — | 420 | 0.94 (regret +0.00018 ± 5e-05) |
| ph | 102 | +5e-05 ± 5e-05 | 318 | 0.92 (regret +0.00028 ± 6e-05) |

## dss score

| | classical | mincontrast | oracle_mincontrast | oracle_ph | ph |
|---|---|---|---|---|---|
| overall | 0.99 (regret +0.69 ± 0.2) | 0.89 (regret +5.2 ± 1) | 0.88 (regret +5.6 ± 0.9) | 0.99 (regret +0.56 ± 0.2) | 0.99 (regret +0.64 ± 0.2) |
| structured clouds | 0.99 (regret +0.69 ± 0.2) | 0.89 (regret +5.8 ± 2) | 0.88 (regret +6.3 ± 1) | 0.99 (regret +0.59 ± 0.2) | 0.99 (regret +0.67 ± 0.2) |
| poisson | — (regret +0.69 ± 0.3) | — (regret +0.78 ± 0.3) | — (regret +0.34 ± 0.3) | — (regret +0.34 ± 0.3) | — (regret +0.46 ± 0.2) |
| thomas | 0.98 (regret +0.63 ± 0.7) | 0.97 (regret +0.93 ± 0.7) | 0.98 (regret +0.66 ± 0.6) | 0.97 (regret +1 ± 0.7) | 0.98 (regret +0.73 ± 0.7) |
| nested | 0.99 (regret +1.2 ± 0.5) | 0.99 (regret +1.6 ± 0.6) | 0.99 (regret +1.1 ± 0.6) | 0.99 (regret +1.6 ± 0.5) | 0.99 (regret +1.3 ± 0.5) |
| lgcp | 0.97 (regret +1.2 ± 0.6) | 0.96 (regret +1.7 ± 0.6) | 0.98 (regret +1 ± 0.6) | 0.98 (regret +0.71 ± 0.5) | 0.98 (regret +0.82 ± 0.6) |
| matern2 | 1.10 (regret -0.22 ± 0.4) | 0.82 (regret +0.43 ± 0.4) | 1.11 (regret -0.26 ± 0.4) | 1.21 (regret -0.49 ± 0.3) | 1.01 (regret -0.019 ± 0.4) |
| ring | 0.99 (regret +0.5 ± 0.5) | 0.99 (regret +0.86 ± 0.7) | 0.99 (regret +1 ± 0.6) | 0.99 (regret +0.86 ± 0.5) | 0.99 (regret +0.46 ± 0.6) |
| matern1 | 0.45 (regret +0.37 ± 0.3) | 0.38 (regret +0.42 ± 0.3) | 0.41 (regret +0.4 ± 0.3) | 0.62 (regret +0.26 ± 0.2) | 0.17 (regret +0.57 ± 0.2) |
| cell | 0.96 (regret +1.1 ± 0.4) | -0.23 (regret +35 ± 1e+01) | -0.42 (regret +40 ± 6) | 0.99 (regret +0.18 ± 0.3) | 0.97 (regret +0.85 ± 0.4) |

By family and stratum (regime: P(detected | u); cell: k):

| family / stratum | n | gain (true vs CSR) | classical | mincontrast | oracle_mincontrast | oracle_ph | ph |
|---|---|---|---|---|---|---|---|
| cell | k 2-2 | 20 | +3.9 ± 0.9 | 0.73 (regret +1 ± 0.8) | -0.16 (regret +4.5 ± 0.9) | -16.16 (regret +67 ± 9) | 0.97 (regret +0.11 ± 0.6) | 0.81 (regret +0.73 ± 0.7) |
| cell | k 3-4 | 20 | +3.6 ± 1 | 0.55 (regret +1.6 ± 0.8) | -0.69 (regret +6.1 ± 2) | -8.78 (regret +35 ± 1e+01) | 1.01 (regret -0.02 ± 0.6) | 0.63 (regret +1.3 ± 0.9) |
| cell | k 5-30 | 20 | +77 ± 2e+01 | 0.99 (regret +0.77 ± 0.4) | -0.21 (regret +94 ± 3e+01) | 0.77 (regret +18 ± 8) | 0.99 (regret +0.46 ± 0.4) | 0.99 (regret +0.46 ± 0.4) |
| lgcp | 0.5-0.9 | 20 | +3.3 ± 1 | 0.66 (regret +1.1 ± 1) | 0.85 (regret +0.5 ± 0.7) | 0.89 (regret +0.35 ± 0.7) | 0.78 (regret +0.73 ± 0.9) | 0.82 (regret +0.6 ± 0.7) |
| lgcp | above 0.9 | 20 | +1.2e+02 ± 4e+01 | 0.98 (regret +2 ± 0.8) | 0.98 (regret +2.8 ± 1) | 0.99 (regret +1.7 ± 1) | 0.99 (regret +1 ± 1) | 0.99 (regret +1.4 ± 1) |
| lgcp | below 0.5 | 20 | -1.1 ± 0.7 | — (regret +0.64 ± 1) | — (regret +1.9 ± 1) | — (regret +1 ± 1) | — (regret +0.4 ± 0.9) | — (regret +0.5 ± 0.9) |
| matern1 | 0.5-0.9 | 20 | -0.68 ± 0.3 | — (regret -0.045 ± 0.5) | — (regret -0.24 ± 0.4) | — (regret -0.29 ± 0.4) | — (regret -0.12 ± 0.4) | — (regret +0.22 ± 0.4) |
| matern1 | above 0.9 | 20 | +2.2 ± 0.8 | 0.75 (regret +0.54 ± 0.4) | 0.67 (regret +0.72 ± 0.5) | 0.90 (regret +0.22 ± 0.4) | 0.71 (regret +0.62 ± 0.4) | 0.81 (regret +0.41 ± 0.3) |
| matern1 | below 0.5 | 20 | +0.58 ± 0.3 | — (regret +0.63 ± 0.4) | — (regret +0.79 ± 0.5) | — (regret +1.3 ± 0.5) | — (regret +0.28 ± 0.3) | — (regret +1.1 ± 0.5) |
| matern2 | 0.5-0.9 | 20 | -0.78 ± 0.6 | — (regret -1.1 ± 0.8) | — (regret -0.36 ± 0.8) | — (regret -0.6 ± 1) | — (regret -1.3 ± 0.7) | — (regret -0.86 ± 1) |
| matern2 | above 0.9 | 20 | +7.8 ± 2 | 1.03 (regret -0.24 ± 0.6) | 0.92 (regret +0.64 ± 0.5) | 0.98 (regret +0.15 ± 0.4) | 0.99 (regret +0.051 ± 0.5) | 0.96 (regret +0.31 ± 0.6) |
| matern2 | below 0.5 | 20 | +0.04 ± 0.5 | — (regret +0.7 ± 0.6) | — (regret +1 ± 0.6) | — (regret -0.35 ± 0.4) | — (regret -0.22 ± 0.5) | — (regret +0.49 ± 0.6) |
| nested | 0.5-0.9 | 20 | +0.85 ± 2 | — (regret +1.2 ± 1) | — (regret +1.2 ± 0.9) | — (regret +0.96 ± 0.9) | — (regret +2.1 ± 0.9) | — (regret +1 ± 0.9) |
| nested | above 0.9 | 20 | +5.7e+02 ± 2e+02 | 1.00 (regret +1.4 ± 0.7) | 1.00 (regret +2.3 ± 1) | 1.00 (regret +1.8 ± 1) | 1.00 (regret +0.94 ± 0.9) | 1.00 (regret +1.6 ± 1) |
| nested | below 0.5 | 20 | -0.99 ± 0.4 | — (regret +0.95 ± 0.8) | — (regret +1.2 ± 0.6) | — (regret +0.67 ± 0.8) | — (regret +1.8 ± 0.8) | — (regret +1.2 ± 0.8) |
| poisson | all | 60 | +0.036 ± 0.3 | — (regret +0.69 ± 0.3) | — (regret +0.78 ± 0.3) | — (regret +0.34 ± 0.3) | — (regret +0.34 ± 0.3) | — (regret +0.46 ± 0.2) |
| ring | 0.5-0.9 | 20 | +7.7 ± 4 | 0.95 (regret +0.37 ± 1) | 0.88 (regret +0.89 ± 1) | 0.85 (regret +1.2 ± 0.9) | 0.88 (regret +0.9 ± 0.8) | 0.91 (regret +0.73 ± 1) |
| ring | above 0.9 | 20 | +2.3e+02 ± 6e+01 | 1.00 (regret +0.9 ± 1) | 1.00 (regret +0.95 ± 2) | 0.99 (regret +1.5 ± 2) | 1.00 (regret +0.8 ± 1) | 1.00 (regret +0.33 ± 1) |
| ring | below 0.5 | 20 | -0.024 ± 0.5 | — (regret +0.23 ± 0.5) | — (regret +0.75 ± 0.6) | — (regret +0.36 ± 0.5) | — (regret +0.87 ± 0.7) | — (regret +0.31 ± 0.5) |
| thomas | 0.5-0.9 | 20 | +1.7 ± 2 | — (regret +0.21 ± 2) | — (regret +0.74 ± 2) | — (regret +0.26 ± 1) | — (regret +0.11 ± 2) | — (regret -0.31 ± 2) |
| thomas | above 0.9 | 20 | +1e+02 ± 4e+01 | 0.99 (regret +1.3 ± 1) | 0.99 (regret +1.1 ± 1) | 0.99 (regret +1.3 ± 1) | 0.98 (regret +1.9 ± 0.9) | 0.98 (regret +1.8 ± 1) |
| thomas | below 0.5 | 20 | -0.62 ± 0.4 | — (regret +0.35 ± 0.5) | — (regret +0.91 ± 0.6) | — (regret +0.43 ± 0.6) | — (regret +0.97 ± 0.7) | — (regret +0.75 ± 0.6) |

Structured clouds the pipeline sent to poisson ("might as well be Poisson" ⇔ gain ≈ 0):

| pipeline | n | gain | n sent to a family | skill there |
|---|---|---|---|---|
| classical | 108 | +0.28 ± 0.5 | 312 | 0.99 (regret +0.79 ± 0.2) |
| mincontrast | 122 | +3.1 ± 1 | 298 | 0.91 (regret +6.9 ± 2) |
| oracle_mincontrast | 0 | — | 420 | 0.88 (regret +6.3 ± 1) |
| oracle_ph | 0 | — | 420 | 0.99 (regret +0.59 ± 0.2) |
| ph | 102 | +0.3 ± 0.5 | 318 | 0.99 (regret +0.73 ± 0.2) |

