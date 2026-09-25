# Family pilot

Gradient-boosted trees throughout; feature sets differ only in what the trees see. Intervals: 95% bootstrap over test thetas (within family), paired across models.

## Parameter inference: RMSE(log θ)/s.d., mean over targets (lower is better)

| family | classical | ph_alpha | ph_dtm | ph | classical+ph | PH − classical | classical+PH − classical |
|---|---|---|---|---|---|---|---|
| thomas | 1.550 | 1.500 | 1.580 | 1.569 | 1.559 | 0.019 [-0.155, 0.180] | 0.009 [-0.250, 0.238] |
| nested | 0.890 | 0.837 | 0.858 | 0.825 | 0.796 | -0.065 [-0.156, 0.044] | -0.094 [-0.164, -0.011] |
| matern2 | 0.482 | 0.428 | 0.565 | 0.477 | 0.459 | -0.005 [-0.124, 0.116] | -0.023 [-0.140, 0.099] |
| lgcp | 0.779 | 0.756 | 0.821 | 0.750 | 0.720 | -0.029 [-0.149, 0.083] | -0.058 [-0.144, 0.040] |
| ring | 1.294 | 1.261 | 1.204 | 1.174 | 1.155 | -0.121 [-0.263, 0.108] | -0.139 [-0.234, 0.027] |
| matern1 | 0.328 | 0.391 | 0.515 | 0.390 | 0.334 | 0.061 [0.002, 0.113] | 0.006 [-0.027, 0.037] |
| cell | 0.483 | 0.429 | 0.355 | 0.314 | 0.312 | -0.169 [-0.313, -0.037] | -0.171 [-0.351, -0.017] |

In regime (τ = 0.5):

| family | classical | ph_alpha | ph_dtm | ph | classical+ph | PH − classical | classical+PH − classical |
|---|---|---|---|---|---|---|---|
| thomas | 1.613 | 1.612 | 1.681 | 1.700 | 1.673 | 0.086 [-0.146, 0.284] | 0.060 [-0.256, 0.350] |
| nested | 0.950 | 0.876 | 0.874 | 0.861 | 0.811 | -0.089 [-0.176, 0.012] | -0.139 [-0.221, -0.065] |
| matern2 | 0.506 | 0.410 | 0.378 | 0.364 | 0.366 | -0.142 [-0.222, 0.004] | -0.140 [-0.222, -0.025] |
| lgcp | 0.779 | 0.756 | 0.821 | 0.750 | 0.720 | -0.029 [-0.149, 0.083] | -0.058 [-0.144, 0.040] |
| ring | 1.798 | 1.467 | 1.529 | 1.437 | 1.499 | -0.362 [-0.702, 0.195] | -0.300 [-0.543, 0.132] |
| matern1 | 0.335 | 0.398 | 0.517 | 0.384 | 0.348 | 0.049 [-0.002, 0.110] | 0.013 [-0.015, 0.037] |

## Stage 2 with oracle routing: balanced accuracy

| branch | classical | ph_alpha | ph_dtm | ph | classical+ph |
|---|---|---|---|---|---|
| clustered | 0.475 [0.336, 0.592] | 0.400 [0.293, 0.487] | 0.338 [0.275, 0.412] | 0.400 [0.312, 0.507] | 0.438 [0.353, 0.547] |
| clustered, in regime | 0.490 [0.377, 0.606] | 0.372 [0.244, 0.507] | 0.355 [0.266, 0.466] | 0.399 [0.286, 0.519] | 0.430 [0.326, 0.543] |
| repulsive | 0.550 [0.375, 0.675] | 0.625 [0.456, 0.750] | 0.600 [0.431, 0.739] | 0.475 [0.300, 0.625] | 0.550 [0.400, 0.719] |
| repulsive, in regime | 0.569 [0.400, 0.740] | 0.736 [0.562, 0.894] | 0.667 [0.522, 0.841] | 0.500 [0.344, 0.687] | 0.611 [0.432, 0.812] |

Ring called ring / thomas, by jitter σ/ρ:

| jitter | classical | ph_alpha | ph_dtm | ph | classical+ph |
|---|---|---|---|---|---|
| 0.05-0.1 | 0.25 / 0.25 | 0.50 / 0.25 | 0.25 / 0.00 | 0.50 / 0.00 | 0.50 / 0.00 |
| 0.1-0.2 | 0.17 / 0.33 | 0.17 / 0.17 | 0.33 / 0.00 | 0.17 / 0.33 | 0.17 / 0.17 |
| 0.2-0.4 | 0.50 / 0.00 | 0.50 / 0.00 | 0.50 / 0.00 | 1.00 / 0.00 | 0.50 / 0.00 |
| 0.4-1 | 0.25 / 0.25 | 0.50 / 0.00 | 0.38 / 0.12 | 0.25 / 0.25 | 0.25 / 0.25 |

Matérn recall by core size R√n̄ (matern2 / matern1):

| core | classical | ph_alpha | ph_dtm | ph | classical+ph |
|---|---|---|---|---|---|
| 0.05-0.1 | 0.50 / 0.70 | 0.25 / 0.80 | 0.38 / 0.80 | 0.38 / 0.60 | 0.38 / 0.70 |
| 0.1-0.2 | 0.75 / 0.25 | 1.00 / 0.62 | 0.75 / 0.62 | 0.50 / 0.50 | 0.75 / 0.50 |
| 0.2-0.34 | 0.00 / 0.00 | 0.00 / 0.50 | 0.50 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| 0.34-0.55 | 1.00 / — | 0.83 / — | 0.67 / — | 0.67 / — | 0.83 / — |

## Stage 1 (7 families) and where it sends `cell`

| features | group accuracy | cell k 2-2.5: P / C / R | cell k 2.5-4.5: P / C / R | cell k 4.5-9.5: P / C / R | cell k 9.5-30: P / C / R |
|---|---|---|---|---|---|
| classical | 0.600 [0.551, 0.694] | 0.25 / 0.75 / 0.00 | 0.38 / 0.62 / 0.00 | 0.25 / 0.75 / 0.00 | 0.00 / 0.75 / 0.25 |
| ph | 0.696 [0.631, 0.763] | 0.00 / 1.00 / 0.00 | 0.25 / 0.38 / 0.38 | 0.25 / 0.25 / 0.50 | 0.50 / 0.00 / 0.50 |
| classical+ph | 0.638 [0.577, 0.719] | 0.00 / 1.00 / 0.00 | 0.62 / 0.25 / 0.12 | 0.25 / 0.75 / 0.00 | 0.00 / 0.75 / 0.25 |

Regime cutoffs (from stage-1 out-of-fold routing on train):

| family | coordinate | n̄ exponent | routed to own group | test in regime |
|---|---|---|---|---|
| thomas | omega | -0.24 | 0.804 | 0.800 |
| nested | omega_inner | -0.55 | 0.875 | 0.700 |
| lgcp | zeta | 2.79 | 0.911 | 1.000 |
| matern2 | zeta | -0.33 | 0.857 | 0.600 |
| ring | omega | -0.53 | 0.750 | 0.300 |
| matern1 | zeta | -0.34 | 0.946 | 0.900 |

## End to end: family accuracy over the 7 grouped families (balanced-groups prior)

| model | all | in regime | poisson | thomas | nested | lgcp | ring | matern2 | matern1 |
|---|---|---|---|---|---|---|---|---|---|
| cascade classical -> classical | 0.325 [0.260, 0.417] | 0.346 [0.247, 0.452] | 0.15 | 0.25 | 0.60 | 0.50 | 0.25 | 0.45 | 0.40 |
| cascade classical -> ph | 0.304 [0.226, 0.415] | 0.322 [0.242, 0.446] | 0.15 | 0.20 | 0.45 | 0.60 | 0.30 | 0.30 | 0.45 |
| cascade classical -> classical+ph | 0.321 [0.242, 0.428] | 0.350 [0.266, 0.454] | 0.15 | 0.15 | 0.50 | 0.65 | 0.25 | 0.35 | 0.50 |
| cascade classical+ph -> classical+ph | 0.358 [0.273, 0.470] | 0.393 [0.294, 0.510] | 0.25 | 0.10 | 0.50 | 0.60 | 0.20 | 0.45 | 0.50 |
| cascade ph -> ph | 0.388 [0.288, 0.482] | 0.405 [0.312, 0.505] | 0.40 | 0.20 | 0.45 | 0.60 | 0.20 | 0.35 | 0.45 |
| one-shot classical | 0.358 [0.289, 0.447] | 0.394 [0.315, 0.491] | 0.30 | 0.10 | 0.60 | 0.55 | 0.25 | 0.50 | 0.30 |
| one-shot ph | 0.388 [0.297, 0.490] | 0.428 [0.345, 0.530] | 0.45 | 0.10 | 0.50 | 0.50 | 0.05 | 0.40 | 0.45 |
| one-shot classical+ph | 0.375 [0.280, 0.461] | 0.427 [0.313, 0.530] | 0.40 | 0.10 | 0.55 | 0.45 | 0.20 | 0.35 | 0.45 |

| difference | all | in regime |
|---|---|---|
| one-shot classical+ph  minus  one-shot classical | 0.017 [-0.059, 0.090] | 0.033 [-0.053, 0.107] |
| cascade classical -> classical  minus  one-shot classical | -0.033 [-0.087, 0.037] | -0.048 [-0.090, 0.012] |
| cascade classical -> classical+ph  minus  one-shot classical+ph | -0.054 [-0.112, -0.005] | -0.077 [-0.134, -0.015] |
| cascade classical -> classical+ph  minus  cascade classical -> classical | -0.004 [-0.049, 0.041] | 0.004 [-0.028, 0.043] |
| cascade classical+ph -> classical+ph  minus  one-shot classical+ph | -0.017 [-0.062, 0.021] | -0.034 [-0.091, 0.009] |

## 8-way one-shot with `cell` as a class (uniform prior)

| features | macro accuracy | cell recall by k | other families called cell |
|---|---|---|---|
| classical | 0.338 [0.275, 0.392] | 2-2.5: 0.25, 2.5-4.5: 0.25, 4.5-9.5: 1.00, 9.5-30: 1.00 | poisson 0.05, ring 0.05 |
| ph | 0.344 [0.281, 0.422] | 2-2.5: 0.25, 2.5-4.5: 0.25, 4.5-9.5: 1.00, 9.5-30: 1.00 | poisson 0.05, thomas 0.05, matern2 0.05, matern1 0.05 |
| classical+ph | 0.375 [0.290, 0.444] | 2-2.5: 0.25, 2.5-4.5: 0.25, 4.5-9.5: 1.00, 9.5-30: 1.00 | poisson 0.05 |
