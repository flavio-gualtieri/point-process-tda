# paper — everything for the writing

```
main.tex              the paper (AISTATS template; aistats2026.sty)
paper.yaml            which results the figures and tables are built from -- the one file to switch after a rerun
scripts/              one script per display; `python paper/scripts/make.py [f1 t4 ...]` builds them all
figs/, tables/        generated (committed, so the paper compiles without re-running anything)
notes/                method write-ups: departure (delta-tilde), simulation, featurization, models, scores; outline.tex
```

The scripts only read a finished run (predictions, comparison and evaluation outputs, the bank's
manifests); none trains or scores anything. `paper.yaml` currently points at the paper run as it
stands (`oneshot/results/default`). After the rerun, point `run`, `evaluation`, `power` and
`ablation` at the new results and rebuild.

## Displays

| ID | What | Built by | From | Status |
|---|---|---|---|---|
| T1 | the 8 families: parameters, K, regime coordinate | hand, in `main.tex` | `src/cloudforger/simulation/families.py`, `pipeline/regime.py` | filled; re-check the K cells against the code |
| T2 | simulation design and data volume | hand, in `main.tex` | `configs/simulation.yaml`, `simulation/{families,bank,split}.py` | filled |
| F1 | example realizations, 2 per family | `f1_examples.py` | bank points + manifests, `compare/cutoffs.json` | generated |
| T3 | models: learner, input, training cost | `tables.py` → `t3_cost` (+ hand columns) | each unit's `report.json` | generated; # params and GPU model are not logged |
| F2 | pipeline schematic | hand (TikZ) | — | to draw |
| T8 | score power: d′ of true model vs CSR | `tables.py` → `t8_power` | `scripts/power.py` report | generated; the current run has 5 families, the rerun all 8 |
| T4 | classification by model × family | `tables.py` → `t4_classification` | `compare/report.json`, `mincontrast/report.json` | generated |
| F3 | confusion matrix of the frozen classifier | `f3_confusion.py` | `classify/<model>/predictions.npz` | generated |
| F4 | P(called Poisson) over each family's parameters, with boundaries | `f4_regime.py` | reference classifier predictions, `cutoffs.json` | generated; analytic curves not derived (only the hard core) |
| T5 | the fitted "looks like Poisson" boundary | `tables.py` → `t5_boundary` (fitted); near-closed forms by hand | `cutoffs.json` | fitted part generated; near-closed forms for the cluster families and LGCP not derived |
| T6 | fitted model vs Poisson, by regime stratum | `tables.py` → `t6_poisson_gap` | evaluation `main` | generated |
| T7 | main results | — | evaluation `mincontrast` and `main` | **blocked on the headline pipeline** (`ph` vs `best`) |
| F5 | detection, error and gain against u | `f5_main.py` | all classifiers / estimators, evaluation `main` | generated |
| T9 | estimation error per family × estimator | `tables.py` → `t9_estimation` | `compare/report.json`, `mincontrast/report.json` | generated |
| T10 | what PH adds | `tables.py` → `t10_ph` | `compare/report.json`, evaluations `ph_ablation`, `ph_cell` | generated |
| — | single-model ablation (appendix) | to add to `tables.py` | `<results>/ablation/compare/{classifiers,estimators}_by_model.csv` | after the ablation runs |

## Open decisions (they change what the scripts show)

- **Headline pipeline:** `ph` (hgb_classical_ph everywhere, the mincontrast evaluation) or `best`
  (per-family best on validation, evaluation `main`). Blocks T7; sets `headline_variant` for T6.
- **Reference classifier for the regime:** the config uses `hgb_classical`; the frozen headline
  classifier is `hgb_classical_ph`. Affects F4, T5, T6 (`configs/pipeline.yaml` `regime.classifier`).
- **Whether PH stays** in the paper (T10, and Section 1.2 of `main.tex`).
- **Figure style:** the reference data-viz palette (light, print): one blue ramp for magnitudes,
  at most three coloured series with every other model grey (F5), recessive axes. Colours live in
  `scripts/common.py`.
