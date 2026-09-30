# results — the frozen numbers the paper is written from

```
summary.csv      every reported number, one row each (built by paper/scripts/summary.py; do not edit)
manifest.yaml    what was frozen: commit, config and bank hashes, date, known gaps
classify/ params/   NOT the paper run: the 5-family study from before the 8-family bank (stale, kept for history)
```

The paper run itself is `oneshot/results/default` (`paper/paper.yaml` `run`; also reachable as
`results_paper/pipeline`). `summary.csv` only reads its stored reports; nothing is rescored.

```bash
python paper/scripts/summary.py      # rebuilds summary.csv and paper/tables/s1_pipelines.tex, s2_models.tex
```

This directory is gitignored; commit these three files with `git add -f`, as for the other summaries.

## What was run

- **Data.** 8 families (Poisson, Thomas, nested Thomas, LGCP, Matérn II, ring, Matérn I, cell), 50 000 θ
  per family × 2 replicates. Split by θ: 37 000 / 1 000 / 12 000, so 192 000 test patterns.
- **Models.** 13 classifiers and 12 estimators per family, one training seed each: gradient-boosted
  trees and a linear model on classical summaries (L, F, G, J) and/or persistent-homology (PH) summaries;
  networks on the summary curves, on persistence images, on PersLay, and on curves and images fused.
- **Baseline.** Minimum contrast on Ripley's K, with penalised-contrast model selection, tuned on val.
- **End to end.** A pipeline (classifier, then the called family's estimator) is fitted on replicate 0
  and scored on replicate 1 with a kernel score. Skill is 1 when the fit scores as well as the true
  model and 0 when it is no better than a Poisson fit. Four sets of test clouds:

| set (`source`) | clouds | pipelines |
|---|---|---|
| `evaluation/heldout` | 480, stratified over every regime | best (7-model assignment of 25 Sep), classical |
| `evaluation/heldout_mincontrast` | the same 480 | ph, classical, mincontrast, oracle_ph, oracle_mincontrast |
| `evaluation/heldout_ph_ablation` | 1 080, strong-signal strata only | classical, ph, ph_estimators |
| `evaluation/heldout_ph_cell` | 1 000, cell with k ≥ 5 | classical, ph, ph_estimators |

## Headline numbers

| | value | row in `summary.csv` |
|---|---|---|
| 8-way accuracy, best classifier | 0.535 [0.532, 0.537] | `hgb_classical_ph6`, accuracy, all |
| … in regime (τ = 0.9) | 0.646 to 0.652 for the top four | accuracy, regime 0.9 |
| … minimum-contrast selection | 0.329 | `mincontrast (tuned)`, accuracy |
| Skill, trees on classical + PH (`ph`) | 0.92 | `pipeline:ph`, kernel_skill, heldout_mincontrast |
| Skill, same estimators, true family | 0.94 | `pipeline:oracle_ph` |
| Skill, classical features only | 0.91 | `pipeline:classical` |
| Skill, minimum contrast | 0.84 | `pipeline:mincontrast` |
| Cell (k ≥ 5) skill, classical → + PH | 0.92 → 1.00 | heldout_ph_cell |
| Cell estimation error, classical → + PH | 0.142 → 0.120 | `hgb_classical` / `hgb_classical_ph`, rmse_over_sd, cell |

Estimation error is RMSE(log θ) / s.d., averaged over a family's parameters; lower is better.

## Columns

| column | meaning |
|---|---|
| `method` | a model (`hgb_classical_ph`, `fusion_curves_alpha`, …), `mincontrast (tuned\|default)`, or `pipeline:<variant>` for end-to-end rows |
| `family` | the family, or `all` for a pooled number |
| `metric` | `accuracy`, `nll`, `recall`, `detected`, `rmse_over_sd`, `train_seconds`, `{kernel,dss}_{skill,regret,gain}` |
| `mean` | the point estimate on the test split (`subset` = `val`: the validation split) |
| `std`, `n_seeds` | over training seeds. Every unit has one seed, so `std` is empty and `n_seeds` is 1 |
| `lo`, `hi` | 95% bootstrap interval over test θ (compare rows only) |
| `se`, `n` | standard error over clouds and number of clouds (end-to-end rows only) |
| `subset` | `all`; `regime 0.5`, `regime 0.9` (detectable from Poisson with probability ≥ τ); a stratum; `structured`; `ended poisson\|family`; `identified True\|False` |
| `source` | the file the row was read from, relative to the run |

## Read before quoting

- **No seed variance.** Intervals cover test sampling only. Between two networks, a difference under
  about 0.005 in `rmse_over_sd` is within seed noise.
- **Two fused networks have no intervals.** `fusion_curves_dtm10` and `fusion_curves_alpha_dtm10` were
  trained after the last `compare`; their rows come from their own `report.json` and carry `mean` only.
- **`pipeline:best` is dated.** It was scored with the per-family assignment of 25 Sep (7 models),
  which is no longer the best on validation.
- **Per-family skill is noisy.** On the 480-cloud sets each family has 60 clouds. The same pipeline
  rescored on the same clouds moves by up to 0.06 per family (`pipeline:classical` appears in two sets).
  Quote overall skill, or the 1 000-cloud sets.
- **An empty skill** means the true model's gain over Poisson is not resolved (< 2 s.e.) on those clouds.
