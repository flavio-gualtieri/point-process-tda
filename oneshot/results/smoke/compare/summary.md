# One-shot comparison: smoke

Test split, 95% bootstrap intervals over test thetas; differences are paired, against `hgb_classical`. Prior: equal.

## Classifiers

| model | accuracy all | accuracy regime 0.5 | accuracy regime 0.9 | − baseline (all) | NLL |
|---|---|---|---|---|---|
| hgb_classical | 0.467 [0.428, 0.505] | 0.475 [0.437, 0.514] | 0.530 [0.484, 0.583] | — | 1.777 |
| hgb_small | 0.461 [0.422, 0.497] | 0.474 [0.432, 0.510] | 0.529 [0.491, 0.572] | -0.007 [-0.020, 0.014] | 1.358 |
| linear_classical | 0.436 [0.399, 0.473] | 0.448 [0.411, 0.480] | 0.505 [0.472, 0.563] | -0.031 [-0.061, 0.001] | 1.345 |

Recall (called own family) / detected (not called poisson):

| model | poisson | thomas | nested | lgcp | matern2 | ring | matern1 | cell |
|---|---|---|---|---|---|---|---|---|
| hgb_classical | 0.44 / 0.56 | 0.29 / 0.78 | 0.35 / 0.99 | 0.44 / 0.85 | 0.55 / 0.97 | 0.25 / 0.77 | 0.52 / 0.97 | 0.90 / 0.94 |
| hgb_small | 0.46 / 0.54 | 0.25 / 0.78 | 0.36 / 0.98 | 0.45 / 0.85 | 0.46 / 0.94 | 0.31 / 0.79 | 0.52 / 0.95 | 0.88 / 0.95 |
| linear_classical | 0.49 / 0.51 | 0.23 / 0.75 | 0.31 / 0.94 | 0.53 / 0.82 | 0.44 / 0.93 | 0.21 / 0.80 | 0.52 / 0.94 | 0.76 / 0.89 |

## Regime (reference classifier `hgb_classical`, event: detected)

| family | coordinate | n̄ exponent | event rate (fit) | test in regime τ=0.5 | test in regime τ=0.9 |
|---|---|---|---|---|---|
| thomas | omega | 0.29 | 0.730 (train) | 0.771 | 0.479 |
| nested | omega_inner | -0.06 | 0.909 (train) | 0.917 | 0.792 |
| lgcp | zeta | -0.69 | 0.841 (train) | 0.958 | 0.479 |
| matern2 | zeta | -0.38 | 0.939 (train) | 1.000 | 0.854 |
| ring | omega | 0.34 | 0.794 (train) | 0.792 | 0.417 |
| matern1 | zeta | -0.26 | 0.929 (train) | 0.979 | 0.792 |

## Estimators: RMSE(log θ)/s.d., mean over targets (lower is better)

**all** (bold = best on val)

| family | hgb_classical | linear_classical |
|---|---|---|
| thomas | 0.837 | **0.858** |
| nested | **0.786** | 0.791 |
| lgcp | **0.526** | 0.602 |
| matern2 | **0.217** | 0.156 |
| ring | **0.715** | 0.761 |
| matern1 | 0.241 | **0.265** |
| cell | **0.183** | 0.178 |

**regime 0.5** (bold = best on val)

| family | hgb_classical | linear_classical |
|---|---|---|
| thomas | 0.734 | **0.807** |
| nested | **0.784** | 0.792 |
| lgcp | **0.523** | 0.598 |
| matern2 | **0.217** | 0.156 |
| ring | **0.665** | 0.739 |
| matern1 | 0.240 | **0.257** |
| cell | **0.183** | 0.178 |

**regime 0.9** (bold = best on val)

| family | hgb_classical | linear_classical |
|---|---|---|
| thomas | 0.581 | **0.726** |
| nested | **0.737** | 0.782 |
| lgcp | **0.376** | 0.618 |
| matern2 | **0.219** | 0.145 |
| ring | **0.572** | 0.755 |
| matern1 | 0.226 | **0.242** |
| cell | **0.183** | 0.178 |

Difference against `hgb_classical` (all test clouds of the family):

| family | linear_classical |
|---|---|
| thomas | 0.020 [-0.056, 0.085] |
| nested | 0.005 [-0.064, 0.071] |
| lgcp | 0.076 [-0.006, 0.175] |
| matern2 | -0.062 [-0.099, -0.035] |
| ring | 0.046 [-0.006, 0.114] |
| matern1 | 0.023 [-0.021, 0.059] |
| cell | -0.004 [-0.028, 0.021] |

## Pipelines (classifier × estimators)

RMSE/s.d. on the clouds each pipeline identifies correctly; identified share in brackets.

| pipeline | accuracy all | poisson | thomas | nested | lgcp | matern2 | ring | matern1 | cell |
|---|---|---|---|---|---|---|---|---|---|
| hgb_classical x best | 0.467 [0.428, 0.505] | 0.071 (0.44) | 0.928 (0.29) | 0.690 (0.35) | 0.377 (0.44) | 0.196 (0.55) | 0.653 (0.25) | 0.196 (0.52) | 0.179 (0.90) |
| hgb_classical x hgb_classical | 0.467 [0.428, 0.505] | 0.071 (0.44) | 0.820 (0.29) | 0.690 (0.35) | 0.377 (0.44) | 0.196 (0.55) | 0.653 (0.25) | 0.207 (0.52) | 0.179 (0.90) |
| hgb_classical x mixed | 0.467 [0.428, 0.505] | 0.071 (0.44) | 0.928 (0.29) | 0.690 (0.35) | 0.377 (0.44) | 0.196 (0.55) | 0.668 (0.25) | 0.207 (0.52) | 0.179 (0.90) |
| hgb_small x best | 0.461 [0.422, 0.497] | 0.060 (0.46) | 0.896 (0.25) | 0.707 (0.36) | 0.382 (0.45) | 0.197 (0.46) | 0.696 (0.31) | 0.190 (0.52) | 0.173 (0.88) |
| hgb_small x hgb_classical | 0.461 [0.422, 0.497] | 0.060 (0.46) | 0.796 (0.25) | 0.707 (0.36) | 0.382 (0.45) | 0.197 (0.46) | 0.696 (0.31) | 0.199 (0.52) | 0.173 (0.88) |
| hgb_small x mixed | 0.461 [0.422, 0.497] | 0.060 (0.46) | 0.896 (0.25) | 0.707 (0.36) | 0.382 (0.45) | 0.197 (0.46) | 0.683 (0.31) | 0.199 (0.52) | 0.173 (0.88) |
| linear_classical x best | 0.436 [0.399, 0.473] | 0.068 (0.49) | 0.809 (0.23) | 0.775 (0.31) | 0.455 (0.53) | 0.184 (0.44) | 0.617 (0.21) | 0.207 (0.52) | 0.181 (0.76) |
| linear_classical x hgb_classical | 0.436 [0.399, 0.473] | 0.068 (0.49) | 0.848 (0.23) | 0.775 (0.31) | 0.455 (0.53) | 0.184 (0.44) | 0.617 (0.21) | 0.215 (0.52) | 0.181 (0.76) |
| linear_classical x mixed | 0.436 [0.399, 0.473] | 0.068 (0.49) | 0.809 (0.23) | 0.775 (0.31) | 0.455 (0.53) | 0.184 (0.44) | 0.738 (0.21) | 0.215 (0.52) | 0.181 (0.76) |
