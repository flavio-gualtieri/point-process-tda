# results — the frozen numbers the paper is written from

**Start with [`story.md`](story.md)**: every number the paper's story needs, section by section, as
tables with intervals. Each of those tables is also in `paper/tables/<name>.tex`, ready to `\input`.

```
story.md         the page to read: sections 3-6 and the appendix, generated; do not edit
summary.csv      every number, one row each: the single source story.md and the tables are built from
manifest.yaml    what was frozen: commit, sha256 of every config, bank manifest and output, known gaps
frozen/          copies of outputs made outside the repo (the score check; see manifest.yaml)
classify/ params/   NOT the paper run: the 5-family study from before the 8-family bank (stale, history only)
```

Rebuild everything from the frozen outputs (about 40 s, reads only; nothing is trained or simulated):

```bash
python paper/scripts/summary.py                                # summary.csv, story.md, paper/tables/s*_*.tex
CLOUDFORGER_RESULTS=results_paper python paper/scripts/make.py # F1-F5, T3-T10
```

This directory is gitignored; its files are committed with `git add -f`.

## The story, and where each part lives

The paper puts one network in front and uses every other model to show that all of them hit the same
ceiling. The headline is `fusion_curves_alpha_dtm10`: summary curves (L, F, G, J) plus persistence
images on the alpha and DTM filtrations, used as the classifier and as every family's estimator.

| section | claim | table | source |
|---|---|---|---|
| 3 Model | one network; controls laid out as input × learner | `s3_grid` | compare, predictions |
| 4 Ceiling | 13 classifiers within 0.488–0.535; averaging all gives 0.539 | `sA_models`, T4, T5 | compare, predictions |
| 5 Fidelity | skill 0.94 for network and trees alike, 0.85 for minimum contrast | `s5_fidelity`, `s5_pairs` | 480 clouds × 5 seeds |
| 6 Topology | PH cuts cell error 15% (trees) to 20% (network), lifts cell skill by 0.07–0.08 | `s6_topology` | compare; cell set (1000 clouds) |
| Appendix | the score punishes wrong parameters and wrong families | `sA_score_ladder` | wrong-model ladder |

Which models fill the grid, and which pipelines are scored, is set in `paper/paper.yaml` (`story`).
Change a model there and rerun `summary.py`: every table follows.

## What was run

- **Data.** 8 families (Poisson, Thomas, nested Thomas, LGCP, Matérn II, ring, Matérn I, cell), 50 000 θ
  per family × 2 replicates. Split by θ: 37 000 / 1 000 / 12 000, so 192 000 test patterns.
- **Models.** 13 classifiers and 12 estimators per family, one training seed each: gradient-boosted trees
  and a linear model on feature tables; networks on summary curves, persistence images, PersLay, and fused.
- **Baseline.** Minimum contrast on Ripley's K, with penalised-contrast model selection, tuned on val.
- **End to end.** A pipeline (classifier, then the called family's estimator) is fitted on replicate 0 and
  scored on replicate 1 with a kernel score. Skill is 1 when the fit scores as well as the true model and 0
  when it is no better than a Poisson fit.

| clouds | what | seeds | file |
|---|---|---|---|
| 480, every regime stratum | fusion, ph, curves, classical, best | 1 | `evaluation/networks` |
| the same 480 | mincontrast, oracle_ph, oracle_mincontrast, ph, classical | 1 | `evaluation/heldout_mincontrast` |
| the same 480 | all 7 above, kernel score only | **5 simulation seeds** | `frozen/scorecheck/repeat_seed*.csv` |
| 1 000, cell k ≥ 5 | estimators swapped, classifier `hgb_classical` fixed | 1 | `evaluation/networks_cell` |
| 2 100, 100 per stratum | the true family at wrong θ (the ladder) | 1 | `frozen/scorecheck/dose_seed0.csv` |
| 1 080 / 1 000 | the earlier trees-only PH ablation | 1 | `evaluation/heldout_ph_{ablation,cell}` |

`evaluation/…` is relative to the run, `oneshot/results/default`.

## summary.csv columns

| column | meaning |
|---|---|
| `method` | a model; `mincontrast (tuned\|default)`; `ensemble (13 classifiers)`; `pipeline:<variant>`; `wrong model: <kind>`; `oracle`; or `A - B` for a paired difference |
| `family` | the family, or `all` for a pooled number |
| `metric` | `accuracy`, `nll`, `ece`, `recall`, `detected`, `rmse_over_sd`, `train_seconds`, `{kernel,dss}_{skill,regret,gain}`, `kernel_mdg100` |
| `mean` | the point estimate; over several seeds, the mean of the per-seed values |
| `std`, `n_seeds`, `seeds` | spread over seeds of the kind `seeds` names: `training` (always 1 here) or `simulation` (5) |
| `lo`, `hi` | 95% bootstrap interval: over test θ for models, over clouds for end-to-end numbers |
| `se`, `n` | standard error over clouds and number of clouds (end-to-end rows) |
| `subset` | `all`, `val`; `regime 0.5`, `regime 0.9`; a stratum; `structured`; `ended poisson\|family`; `identified True\|False`; `wrong family`; a ladder tier `strong\|middle\|weak` |
| `source` | the file a row was read from: relative to the run, or to the repo for `results/frozen/` |

Several sources score the same pipeline (e.g. `pipeline:ph` in three sets). For the story, quote the
row whose `source` is `results/frozen/scorecheck/repeat_seed*.csv` (5 seeds); `story.md` already does.

## Read before quoting

- **Network and trees tie end to end.** Skill 0.941 vs 0.940, Δ +0.000 ± 0.019 over seeds. The 0.96 vs 0.92
  in `evaluation/networks` alone was one simulation seed. Say "matches", not "beats".
- **One training seed.** Intervals cover test sampling only. Between two networks, differences in
  `rmse_over_sd` under about 0.005 are within seed noise (retrained `nn_curves`: s.d. 0.0006–0.0044).
- **Per-family skill is not a finding** on the 480 clouds (60 per family): over simulation seeds it moves
  by 0.15–0.7 for Matérn I, Matérn II and cell. Use the 1 000-cloud cell set for cell.
- **Structured clouds sent to Poisson** carry a small gain that 5 seeds resolve (z 1.6–2.4): about 1% of the
  gain on identified clouds. "Costs nothing detectable at one seed; about 1% with five".
- **An empty skill** means the true model's gain over Poisson is not resolved (< 2 s.e.) on those clouds.
- **Skill is not parameter accuracy.** A truth perturbed by 0.25 s.d. still scores 0.86 in the strong
  strata (`sA_score_ladder`): pipelines at 0.94 can have RMSE/s.d. of 0.4–0.7.
