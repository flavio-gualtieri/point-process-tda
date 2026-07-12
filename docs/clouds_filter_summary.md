# Thomas clouds: post-filter summary

Filter: discarded any point cloud with more than 1500 points (dense Thomas-process parameter regions produce clouds whose Rips/DTM complex is combinatorially infeasible regardless of available memory), or fewer than 10 points (below the DTM filtration's k-NN requirement of k=10 neighbors).

## Counts

| split | kept | dropped | avg n_points |
|---|---|---|---|
| train_test | 28730 | 5270 | 433.7 |
| adversarial | 5090 | 910 | 443.6 |
| **combined** | **33820** | **6180** | **435.2** |

## Clustering regimes

Regime is a tertile bin on `cluster_scale` (Low/Medium/High), with bin edges computed over all remaining clouds (both splits combined):

- Low: cluster_scale ≤ 0.01362
- Medium: 0.01362 < cluster_scale ≤ 0.03614
- High: cluster_scale > 0.03614

| split | Low | Medium | High |
|---|---|---|---|
| train_test | 9466 | 9686 | 9578 |
| adversarial | 1813 | 1586 | 1691 |
| **combined** | **11279** | **11272** | **11269** |

