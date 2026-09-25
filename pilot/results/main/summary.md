# Family pilot

Gradient-boosted trees throughout; feature sets differ only in what the trees see. Intervals: 95% bootstrap over test thetas (within family), paired across models.

## Parameter inference: RMSE(log θ)/s.d., mean over targets (lower is better)

| family | classical | ph_alpha | ph_dtm | ph | classical+ph | PH − classical | classical+PH − classical |
|---|---|---|---|---|---|---|---|
| thomas | 0.739 | 0.773 | 0.756 | 0.756 | 0.732 | 0.016 [0.002, 0.031] | -0.007 [-0.017, 0.003] |
| nested | 0.667 | 0.691 | 0.695 | 0.683 | 0.664 | 0.016 [0.006, 0.027] | -0.003 [-0.011, 0.004] |
| matern2 | 0.142 | 0.148 | 0.240 | 0.146 | 0.141 | 0.004 [-0.000, 0.009] | -0.001 [-0.003, 0.002] |
| lgcp | 0.450 | 0.479 | 0.476 | 0.462 | 0.448 | 0.011 [0.004, 0.019] | -0.002 [-0.008, 0.003] |
| ring | 0.675 | 0.688 | 0.686 | 0.679 | 0.664 | 0.004 [-0.005, 0.014] | -0.011 [-0.019, -0.003] |
| matern1 | 0.162 | 0.173 | 0.306 | 0.172 | 0.160 | 0.010 [0.005, 0.015] | -0.003 [-0.005, 0.000] |
| cell | 0.151 | 0.171 | 0.134 | 0.129 | 0.126 | -0.021 [-0.026, -0.016] | -0.025 [-0.029, -0.021] |

In regime (τ = 0.5):

| family | classical | ph_alpha | ph_dtm | ph | classical+ph | PH − classical | classical+PH − classical |
|---|---|---|---|---|---|---|---|
| thomas | 0.643 | 0.700 | 0.676 | 0.672 | 0.638 | 0.029 [0.012, 0.044] | -0.005 [-0.016, 0.006] |
| nested | 0.660 | 0.684 | 0.690 | 0.677 | 0.657 | 0.017 [0.006, 0.029] | -0.003 [-0.011, 0.004] |
| matern2 | 0.125 | 0.134 | 0.221 | 0.132 | 0.126 | 0.007 [0.003, 0.011] | 0.000 [-0.002, 0.003] |
| lgcp | 0.434 | 0.459 | 0.460 | 0.443 | 0.430 | 0.009 [0.001, 0.017] | -0.004 [-0.010, 0.002] |
| ring | 0.646 | 0.657 | 0.658 | 0.647 | 0.634 | 0.002 [-0.008, 0.012] | -0.012 [-0.020, -0.004] |
| matern1 | 0.148 | 0.158 | 0.288 | 0.159 | 0.145 | 0.011 [0.006, 0.016] | -0.003 [-0.006, -0.000] |

## Stage 2 with oracle routing: balanced accuracy

| branch | classical | ph_alpha | ph_dtm | ph | classical+ph |
|---|---|---|---|---|---|
| clustered | 0.459 [0.446, 0.470] | 0.442 [0.430, 0.454] | 0.431 [0.419, 0.443] | 0.448 [0.435, 0.461] | 0.469 [0.456, 0.480] |
| clustered, in regime | 0.466 [0.453, 0.479] | 0.449 [0.436, 0.462] | 0.440 [0.426, 0.452] | 0.456 [0.442, 0.469] | 0.477 [0.464, 0.490] |
| repulsive | 0.583 [0.567, 0.599] | 0.584 [0.566, 0.601] | 0.568 [0.551, 0.585] | 0.596 [0.581, 0.613] | 0.590 [0.574, 0.606] |
| repulsive, in regime | 0.589 [0.571, 0.605] | 0.590 [0.571, 0.608] | 0.573 [0.556, 0.591] | 0.599 [0.583, 0.616] | 0.594 [0.577, 0.612] |

Ring called ring / thomas, by jitter σ/ρ:

| jitter | classical | ph_alpha | ph_dtm | ph | classical+ph |
|---|---|---|---|---|---|
| 0.05-0.1 | 0.53 / 0.17 | 0.48 / 0.24 | 0.48 / 0.20 | 0.50 / 0.23 | 0.51 / 0.20 |
| 0.1-0.2 | 0.42 / 0.24 | 0.42 / 0.27 | 0.40 / 0.26 | 0.46 / 0.25 | 0.44 / 0.25 |
| 0.2-0.4 | 0.34 / 0.29 | 0.34 / 0.28 | 0.31 / 0.31 | 0.33 / 0.28 | 0.32 / 0.34 |
| 0.4-1 | 0.22 / 0.37 | 0.24 / 0.35 | 0.22 / 0.37 | 0.24 / 0.39 | 0.27 / 0.38 |

Matérn recall by core size R√n̄ (matern2 / matern1):

| core | classical | ph_alpha | ph_dtm | ph | classical+ph |
|---|---|---|---|---|---|
| 0.05-0.1 | 0.31 / 0.71 | 0.29 / 0.71 | 0.25 / 0.72 | 0.27 / 0.76 | 0.30 / 0.73 |
| 0.1-0.2 | 0.28 / 0.73 | 0.28 / 0.72 | 0.27 / 0.74 | 0.31 / 0.72 | 0.28 / 0.73 |
| 0.2-0.34 | 0.40 / 0.69 | 0.46 / 0.68 | 0.40 / 0.62 | 0.45 / 0.69 | 0.45 / 0.68 |
| 0.34-0.55 | 0.99 / — | 0.99 / — | 0.99 / — | 1.00 / — | 0.99 / — |

## Stage 1 (7 families) and where it sends `cell`

| features | group accuracy | cell k 2-2.5: P / C / R | cell k 2.5-4.5: P / C / R | cell k 4.5-9.5: P / C / R | cell k 9.5-30: P / C / R |
|---|---|---|---|---|---|
| classical | 0.754 [0.745, 0.764] | 0.35 / 0.56 / 0.09 | 0.55 / 0.35 / 0.10 | 0.35 / 0.51 / 0.14 | 0.25 / 0.42 / 0.34 |
| ph | 0.730 [0.721, 0.740] | 0.29 / 0.60 / 0.11 | 0.46 / 0.43 / 0.11 | 0.05 / 0.80 / 0.15 | 0.05 / 0.50 / 0.45 |
| classical+ph | 0.738 [0.729, 0.747] | 0.39 / 0.50 / 0.11 | 0.48 / 0.40 / 0.11 | 0.07 / 0.78 / 0.16 | 0.06 / 0.52 / 0.42 |

Regime cutoffs (from stage-1 out-of-fold routing on train):

| family | coordinate | n̄ exponent | routed to own group | test in regime |
|---|---|---|---|---|
| thomas | omega | -0.00 | 0.693 | 0.637 |
| nested | omega_inner | -0.25 | 0.884 | 0.941 |
| lgcp | zeta | -0.47 | 0.802 | 0.848 |
| matern2 | zeta | -0.16 | 0.914 | 0.929 |
| ring | omega | -0.10 | 0.775 | 0.837 |
| matern1 | zeta | -0.15 | 0.884 | 0.919 |

## End to end: family accuracy over the 7 grouped families (balanced-groups prior)

| model | all | in regime | poisson | thomas | nested | lgcp | ring | matern2 | matern1 |
|---|---|---|---|---|---|---|---|---|---|
| cascade classical -> classical | 0.503 [0.493, 0.513] | 0.528 [0.517, 0.538] | 0.60 | 0.23 | 0.48 | 0.50 | 0.31 | 0.43 | 0.63 |
| cascade classical -> ph | 0.502 [0.492, 0.513] | 0.527 [0.516, 0.538] | 0.60 | 0.21 | 0.48 | 0.50 | 0.31 | 0.44 | 0.63 |
| cascade classical -> classical+ph | 0.507 [0.497, 0.517] | 0.532 [0.522, 0.542] | 0.60 | 0.23 | 0.49 | 0.52 | 0.31 | 0.44 | 0.63 |
| cascade classical+ph -> classical+ph | 0.486 [0.475, 0.496] | 0.512 [0.501, 0.522] | 0.54 | 0.24 | 0.49 | 0.52 | 0.32 | 0.43 | 0.63 |
| cascade ph -> ph | 0.478 [0.468, 0.489] | 0.503 [0.492, 0.513] | 0.53 | 0.22 | 0.47 | 0.50 | 0.31 | 0.43 | 0.62 |
| one-shot classical | 0.557 [0.548, 0.566] | 0.585 [0.575, 0.594] | 0.82 | 0.16 | 0.46 | 0.46 | 0.28 | 0.42 | 0.60 |
| one-shot ph | 0.550 [0.541, 0.559] | 0.578 [0.569, 0.588] | 0.81 | 0.15 | 0.46 | 0.45 | 0.26 | 0.43 | 0.59 |
| one-shot classical+ph | 0.562 [0.553, 0.570] | 0.591 [0.582, 0.600] | 0.81 | 0.17 | 0.48 | 0.46 | 0.27 | 0.45 | 0.62 |

| difference | all | in regime |
|---|---|---|
| one-shot classical+ph  minus  one-shot classical | 0.004 [-0.004, 0.013] | 0.006 [-0.002, 0.015] |
| cascade classical -> classical  minus  one-shot classical | -0.054 [-0.062, -0.046] | -0.057 [-0.065, -0.048] |
| cascade classical -> classical+ph  minus  one-shot classical+ph | -0.055 [-0.064, -0.046] | -0.059 [-0.068, -0.049] |
| cascade classical -> classical+ph  minus  cascade classical -> classical | 0.004 [-0.003, 0.010] | 0.004 [-0.002, 0.011] |
| cascade classical+ph -> classical+ph  minus  one-shot classical+ph | -0.076 [-0.084, -0.066] | -0.079 [-0.088, -0.069] |

## 8-way one-shot with `cell` as a class (uniform prior)

| features | macro accuracy | cell recall by k | other families called cell |
|---|---|---|---|
| classical | 0.508 [0.499, 0.515] | 2-2.5: 0.92, 2.5-4.5: 0.46, 4.5-9.5: 0.98, 9.5-30: 1.00 | poisson 0.02, thomas 0.01 |
| ph | 0.494 [0.486, 0.502] | 2-2.5: 0.79, 2.5-4.5: 0.40, 4.5-9.5: 0.99, 9.5-30: 1.00 | poisson 0.01 |
| classical+ph | 0.515 [0.507, 0.524] | 2-2.5: 0.91, 2.5-4.5: 0.48, 4.5-9.5: 0.99, 9.5-30: 1.00 | poisson 0.01 |
