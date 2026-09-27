# Minimum-contrast baseline: default

Test clouds: 192000, every test cloud of every family, as in compare/summary.md; learned and classical columns are paired. Error = RMSE(log θ)/s.d., mean over targets (compare.py's normalisation); intervals are 95% over test θ.

## Estimators (true family known)

`mc default` = c 0.25, r_max 0.25; `mc tuned` = the setting with the lowest val error.

| family | mc default | mc tuned (c, r_max) | at bound | hgb_classical | hgb_classical_ph | nn_curves | tuned − hgb_classical_ph |
|---|---|---|---|---|---|---|---|
| thomas | 1.672 | 1.619 (0.5, 0.25) | 0.47 | 0.707 | 0.706 | 0.704 | +0.912 [+0.899, +0.926] |
| nested | 1.673 | 1.670 (0.5, 0.25) | 0.83 | 0.628 | 0.624 | 0.612 | +1.046 [+1.029, +1.060] |
| lgcp | 0.656 | 0.652 (0.5, 0.125) | 0.28 | 0.436 | 0.436 | 0.439 | +0.216 [+0.210, +0.221] |
| matern2 | 0.153 | 0.153 (0.25, 0.0625) | 0.03 | 0.131 | 0.130 | 0.134 | +0.023 [+0.022, +0.024] |
| ring | 1.443 | 1.417 (0.5, 0.25) | 0.56 | 0.648 | 0.642 | 0.645 | +0.775 [+0.765, +0.787] |
| matern1 | 0.189 | 0.189 (0.25, 0.0625) | 0.04 | 0.157 | 0.156 | 0.152 | +0.033 [+0.031, +0.034] |
| cell | 0.547 | 0.547 (n, median k) | — | 0.142 | 0.120 | 0.154 | +0.427 [+0.423, +0.432] |

By regime (tuned mc vs hgb_classical_ph; in regime = compare's frozen cutoffs, τ = 0.5 / 0.9):

| family | subset | n | mc tuned | mc at bound | hgb_classical_ph |
|---|---|---|---|---|---|
| thomas | all | 24000 | 1.619 | 0.47 | 0.706 |
| thomas | out of regime 0.5 | 9704 | 2.116 | 0.80 | 0.839 |
| thomas | in regime 0.5 | 14296 | 1.164 | 0.24 | 0.599 |
| thomas | in regime 0.9 | 8372 | 0.642 | 0.09 | 0.447 |
| nested | all | 24000 | 1.670 | 0.83 | 0.624 |
| nested | out of regime 0.5 | 1912 | 2.430 | 0.98 | 0.743 |
| nested | in regime 0.5 | 22088 | 1.581 | 0.82 | 0.611 |
| nested | in regime 0.9 | 15958 | 1.320 | 0.76 | 0.553 |
| lgcp | all | 24000 | 0.652 | 0.28 | 0.436 |
| lgcp | out of regime 0.5 | 5324 | 0.845 | 0.57 | 0.502 |
| lgcp | in regime 0.5 | 18676 | 0.578 | 0.19 | 0.412 |
| lgcp | in regime 0.9 | 11744 | 0.392 | 0.10 | 0.330 |
| matern2 | all | 24000 | 0.153 | 0.03 | 0.130 |
| matern2 | out of regime 0.5 | 1546 | 0.331 | 0.11 | 0.256 |
| matern2 | in regime 0.5 | 22454 | 0.129 | 0.02 | 0.115 |
| matern2 | in regime 0.9 | 19922 | 0.111 | 0.01 | 0.107 |
| ring | all | 24000 | 1.417 | 0.56 | 0.642 |
| ring | out of regime 0.5 | 6344 | 1.826 | 0.83 | 0.750 |
| ring | in regime 0.5 | 17656 | 1.231 | 0.46 | 0.596 |
| ring | in regime 0.9 | 10676 | 0.876 | 0.32 | 0.498 |
| matern1 | all | 24000 | 0.189 | 0.04 | 0.156 |
| matern1 | out of regime 0.5 | 1212 | 0.392 | 0.14 | 0.295 |
| matern1 | in regime 0.5 | 22788 | 0.169 | 0.03 | 0.143 |
| matern1 | in regime 0.9 | 18748 | 0.136 | 0.02 | 0.128 |

## Selection (the classical pipeline's classifier)

Tuned on val: c 0.5, r_max 0.125, λ 0.3; default: c 0.25, r_max 0.25, λ 0.016. Balanced accuracy (equal prior), same test clouds.

| classifier | accuracy | poisson | thomas | nested | lgcp | matern2 | ring | matern1 | cell |
|---|---|---|---|---|---|---|---|---|---|
| mincontrast (tuned) | 0.329 | 0.60 | 0.45 | 0.16 | 0.36 | 0.56 | 0.11 | 0.39 | 0.00 |
| mincontrast (default) | 0.294 | 0.18 | 0.29 | 0.30 | 0.22 | 0.57 | 0.20 | 0.58 | 0.00 |
| hgb_classical | 0.529 | 0.73 | 0.16 | 0.54 | 0.54 | 0.31 | 0.27 | 0.78 | 0.90 |
| hgb_classical_ph | 0.534 | 0.73 | 0.17 | 0.55 | 0.54 | 0.34 | 0.28 | 0.76 | 0.92 |

Cell is never called: its K is exactly πr², so any K-based selection sends it to poisson.

