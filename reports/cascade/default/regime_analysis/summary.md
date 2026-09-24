# Regime analysis of stage 1

## Cutoffs (cutoffs.json, used to select component training data)

| family | rule | explained | tau 0.3: boundary (pool) | tau 0.5: boundary (pool) | tau 0.7: boundary (pool) | tau 0.9: boundary (pool) |
|---|---|---|---|---|---|---|
| thomas | omega · n̄^-0.18 <= boundary | 0.975 | 0.624 (60%) | 0.428 (49%) | 0.335 (43%) | 0.254 (35%) |
| nested | omega_inner · n̄^-0.32 <= boundary | 0.983 | 0.31 (91%) | 0.219 (83%) | 0.162 (74%) | 0.124 (64%) |
| matern2 | zeta · n̄^-0.15 >= boundary | 0.992 | 0.384 (94%) | 0.462 (88%) | 0.52 (84%) | 0.6 (78%) |
| lgcp | zeta · n̄^-0.23 >= boundary | 0.965 | 0.367 (77%) | 0.7 (66%) | 1.05 (58%) | 2.09 (44%) |

## thomas
- best overall: **omega + nbar** (physical), explained 0.992
- best 1-D non-delta: **omega**, explained 0.935, n̄ adds 0.057, ambiguous 0.64
- its boundary at tau = 0.5: omega <= 1.247, leaving 51% of the family's train clouds

| coordinate | kind | dims | explained | n̄ gain | ambiguous |
|---|---|---|---|---|---|
| omega + nbar | physical | 2 | 0.992 |  |  |
| omega · n̄^a | combined | 2 | 0.975 |  |  |
| omega + mu | physical | 2 | 0.938 |  |  |
| omega | physical | 1 | 0.935 | 0.057 | 0.64 |
| delta_tilde | delta | 1 | 0.861 | 0.039 | 0.70 |
| delta_tilde_hi | delta | 1 | 0.861 | 0.039 | 0.70 |
| sigma | raw | 1 | 0.790 | 0.003 | 0.77 |
| zeta | snr | 1 | 0.724 | 0.003 | 0.75 |
| kappa | raw | 1 | 0.001 | -0.001 | 1.00 |
| mu | raw | 1 | -0.001 | -0.000 | 1.00 |

## nested
- best overall: **omega_inner · n̄^a** (combined), explained 0.983
- best 1-D non-delta: **omega_inner**, explained 0.887, n̄ adds 0.097, ambiguous 0.38
- its boundary at tau = 0.5: omega_inner <= 1.277, leaving 84% of the family's train clouds

| coordinate | kind | dims | explained | n̄ gain | ambiguous |
|---|---|---|---|---|---|
| omega_inner · n̄^a | combined | 2 | 0.983 |  |  |
| omega_inner + omega_outer | physical | 2 | 0.898 |  |  |
| omega_inner | physical | 1 | 0.887 | 0.097 | 0.38 |
| delta_tilde | delta | 1 | 0.845 | 0.047 | 0.40 |
| delta_tilde_hi | delta | 1 | 0.844 | 0.049 | 0.40 |
| zeta | snr | 1 | 0.772 | -0.001 | 0.43 |
| zeta_inner | snr | 1 | 0.710 | 0.019 | 0.47 |
| sigma2 | raw | 1 | 0.691 | -0.002 | 0.49 |
| omega_outer | physical | 1 | 0.607 | 0.010 | 0.48 |
| zeta_outer | snr | 1 | 0.515 | 0.008 | 0.52 |
| sigma1 | raw | 1 | 0.514 | -0.009 | 0.53 |
| mu2 | raw | 1 | 0.026 | 0.006 | 0.96 |
| mu1 | raw | 1 | 0.021 | 0.019 | 1.00 |
| meta_size | physical | 1 | 0.009 | 0.000 | 0.99 |

## matern2
- best overall: **core + nbar** (physical), explained 1.002
- best 1-D non-delta: **zeta**, explained 0.972, n̄ adds 0.029, ambiguous 0.23
- its boundary at tau = 0.5: zeta >= 1.027, leaving 88% of the family's train clouds

| coordinate | kind | dims | explained | n̄ gain | ambiguous |
|---|---|---|---|---|---|
| core + nbar | physical | 2 | 1.002 |  |  |
| zeta · n̄^a | combined | 2 | 0.992 |  |  |
| delta_tilde | delta | 1 | 0.991 | 0.001 | 0.22 |
| delta_tilde_lo | delta | 1 | 0.991 | 0.001 | 0.23 |
| zeta | snr | 1 | 0.972 | 0.029 | 0.23 |
| core | physical | 1 | 0.724 | 0.278 | 0.30 |
| R | raw | 1 | 0.323 | 0.676 | 0.47 |
| lam_p | raw | 1 | 0.227 | 0.521 | 0.48 |

## lgcp
- best overall: **zeta** (snr), explained 0.965
- best 1-D non-delta: **zeta**, explained 0.965, n̄ adds 0.013, ambiguous 0.56
- its boundary at tau = 0.5: zeta >= 2.494, leaving 65% of the family's train clouds

| coordinate | kind | dims | explained | n̄ gain | ambiguous |
|---|---|---|---|---|---|
| zeta | snr | 1 | 0.965 | 0.013 | 0.56 |
| zeta · n̄^a | combined | 2 | 0.965 |  |  |
| delta_tilde_hi | delta | 1 | 0.959 | 0.009 | 0.54 |
| delta_tilde | delta | 1 | 0.958 | 0.009 | 0.54 |
| sigma2 + s_rel | physical | 2 | 0.939 |  |  |
| sigma2 | raw | 1 | 0.534 | 0.043 | 0.73 |
| s_rel | physical | 1 | 0.118 | 0.025 | 0.97 |
| s | raw | 1 | 0.081 | 0.063 | 1.00 |

## Stability across stage-1 models (val split, best 1-D non-delta coordinate)

| run | family | coordinate | boundary (tau 0.5) | routed to own group |
|---|---|---|---|---|
| default | thomas | omega | 1.186 | 0.544 |
| default | nested | omega_inner | 1.151 | 0.817 |
| default | matern2 | zeta | 1.089 | 0.874 |
| default | lgcp | zeta | 2.166 | 0.690 |
| logreg | thomas | omega | 1.335 | 0.537 |
| logreg | nested | omega_inner | 1.15 | 0.813 |
| logreg | matern2 | zeta | 0.9296 | 0.894 |
| logreg | lgcp | zeta | 2.48 | 0.677 |
| nn_curves | thomas | omega | 1.186 | 0.526 |
| nn_curves | nested | omega_inner | 1.151 | 0.808 |
| nn_curves | matern2 | zeta | 1.113 | 0.876 |
| nn_curves | lgcp | zeta | 2.408 | 0.669 |
