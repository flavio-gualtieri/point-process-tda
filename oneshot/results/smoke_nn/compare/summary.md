# One-shot comparison: smoke_nn

Test split, 95% bootstrap intervals over test thetas; differences are paired, against `nn_curves`. Prior: equal.

## Classifiers

| model | accuracy all | accuracy regime 0.5 | − baseline (all) | NLL |
|---|---|---|---|---|
| nn_curves | 0.303 [0.282, 0.327] | 0.303 [0.282, 0.327] | — | 1.718 |
| perslay_dtm10 | 0.283 [0.250, 0.313] | 0.283 [0.250, 0.313] | -0.021 [-0.048, 0.004] | 1.725 |

Recall (called own family) / detected (not called poisson):

| model | poisson | thomas | nested | lgcp | matern2 | ring | matern1 | cell |
|---|---|---|---|---|---|---|---|---|
| nn_curves | 0.00 / 1.00 | 0.06 / 1.00 | 0.77 / 1.00 | 0.00 / 1.00 | 0.10 / 1.00 | 0.00 / 1.00 | 0.82 / 1.00 | 0.67 / 1.00 |
| perslay_dtm10 | 0.09 / 0.91 | 0.03 / 0.98 | 0.72 / 1.00 | 0.04 / 0.94 | 0.38 / 0.96 | 0.10 / 0.95 | 0.19 / 0.96 | 0.71 / 0.95 |

## Regime (reference classifier `nn_curves`, event: detected)

| family | coordinate | n̄ exponent | event rate (fit) | test in regime τ=0.5 |
|---|---|---|---|---|
| thomas | omega | 0.00 | 1.000 (val) | 1.000 |
| nested | omega_inner | 0.00 | 1.000 (val) | 1.000 |
| lgcp | zeta | 0.00 | 1.000 (val) | 1.000 |
| matern2 | zeta | 0.00 | 1.000 (val) | 1.000 |
| ring | omega | 0.00 | 1.000 (val) | 1.000 |
| matern1 | zeta | 0.00 | 1.000 (val) | 1.000 |

## Estimators: RMSE(log θ)/s.d., mean over targets (lower is better)

**all** (bold = best on val)

| family | nn_curves |
|---|---|
| thomas | **0.993** |
| nested | **1.028** |
| lgcp | **1.003** |
| matern2 | **0.984** |
| ring | **0.991** |
| matern1 | **1.048** |
| cell | **0.982** |

**regime 0.5** (bold = best on val)

| family | nn_curves |
|---|---|
| thomas | **0.993** |
| nested | **1.028** |
| lgcp | **1.003** |
| matern2 | **0.984** |
| ring | **0.991** |
| matern1 | **1.048** |
| cell | **0.982** |

Difference against `nn_curves` (all test clouds of the family):

| family |  |
|---|
| thomas |  |
| nested |  |
| lgcp |  |
| matern2 |  |
| ring |  |
| matern1 |  |
| cell |  |

## Pipelines (classifier × estimators)

RMSE/s.d. on the clouds each pipeline identifies correctly; identified share in brackets.

| pipeline | accuracy all | poisson | thomas | nested | lgcp | matern2 | ring | matern1 | cell |
|---|---|---|---|---|---|---|---|---|---|
| nn_curves x nn_curves | 0.303 [0.282, 0.327] | 0.000 (0.00) | 1.015 (0.06) | 0.993 (0.77) | 0.000 (0.00) | 1.079 (0.10) | 0.000 (0.00) | 1.036 (0.82) | 0.882 (0.67) |
| perslay_dtm10 x nn_curves | 0.283 [0.250, 0.313] | 0.209 (0.09) | 1.007 (0.03) | 0.991 (0.72) | 0.876 (0.04) | 0.914 (0.38) | 0.862 (0.10) | 1.031 (0.19) | 0.877 (0.71) |
