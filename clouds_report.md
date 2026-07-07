# Thomas Process Clouds Report

Source: `data/params/2d/thomas/clouds.pkl` (2D Thomas cluster process point clouds)

## Overview

- **Total clouds:** 4,960
-  , per `configs/params/thomas_cloudgen.yaml`

## (1) Average number of points per cloud

- **Mean:** ~917.5 points

## (2) Range of points per cloud

- **Min:** 5 points
- **Max:** 9,110 points

## (3) Clouds per regime

Regimes are defined by tercile bins over the 9 log-spaced grid values for each parameter (bottom 3 = LOW/SMALL, middle 3 = MODERATE, top 3 = HIGH/LARGE).

| Regime | Definition | # Clouds | Mean n_points | Range n_points |
|---|---|---|---|---|
| **Strong, tight clusters** | parent_intensity HIGH, mean_offspring HIGH, cluster_scale SMALL | 160 | 4,069.7 | 1,505 – 8,892 |
| **Strong, diffuse clusters** | parent_intensity HIGH, mean_offspring HIGH, cluster_scale LARGE | 176 | 4,224.6 | 1,526 – 9,110 |
| **Weak clustering** | parent_intensity LOW, mean_offspring LOW, cluster_scale MODERATE | 176 | 48.5 | 9 – 135 |
