# Thomas diagrams: statistical summary

Filtration: `{'maxdim': 1, 'k': 10, 'q': 2, 'thresh': 0.4}` (DTM-weighted Rips, homology dims 0-1).

## Feature counts and persistence

Persistence = death - birth, computed over finite bars only (bars still alive at the filtration threshold are reported separately below).

| split | n diagrams | avg #H0 features | avg H0 persistence | avg #H1 features | avg H1 persistence |
|---|---|---|---|---|---|
| train_test | 28730 | 431.51 | 0.0159 | 44.18 | 0.0088 |
| adversarial | 5090 | 441.49 | 0.0157 | 45.40 | 0.0087 |
| **combined** | **33820** | **433.01** | **0.0159** | **44.36** | **0.0088** |

## Essential (infinite) bars

Bars whose death is still infinite at the filtration threshold (`thresh` in filtration_params) -- these are features that never closed before the complex stopped growing.

| split | H0 infinite | H0 total | H0 % | H1 infinite | H1 total | H1 % |
|---|---|---|---|---|---|---|
| train_test | 74016 | 12397313 | 0.60% | 9016 | 1269197 | 0.71% |
| adversarial | 12857 | 2247190 | 0.57% | 1474 | 231086 | 0.64% |
| **combined** | **86873** | **14644503** | **0.59%** | **10490** | **1500283** | **0.70%** |

## Clustering regimes

Regime is a tertile bin on `cluster_scale` (Low/Medium/High), bin edges computed over all diagrams (both splits combined) -- matches the definition used in clouds_filter_summary.md:

- Low: cluster_scale ≤ 0.01362
- Medium: 0.01362 < cluster_scale ≤ 0.03614
- High: cluster_scale > 0.03614

|    split    | regime | n diagrams | avg #H0 | avg #H1 | avg H1 persistence |
|---|---|---|---|---|---|
| train_test  | Low    |       9466 | 424.45 | 25.29 | 0.0077 |
| train_test  | Medium |       9686 | 445.69 | 43.15 | 0.0079 |
| train_test  | High   |       9578 | 424.16 | 63.88 | 0.0099 |
| adversarial | Low    |       1813 | 398.99 | 23.36 | 0.0077 |
| adversarial | Medium |       1586 | 483.60 | 46.34 | 0.0078 |
| adversarial | High   |       1691 | 447.57 | 68.15 | 0.0096 |

