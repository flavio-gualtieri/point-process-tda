# paper — everything for the writing

```
main.tex              the paper (AISTATS template; aistats2027.sty + fancyhdr.sty)
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
| F1 | one structured realization per family, one row | `f1_examples.py` | evaluation `main` clouds, bank points, `cutoffs.json` | generated |
| T3 | models: learner, input, training cost | `tables.py` → `t3_cost` (+ hand columns) | each unit's `report.json` | generated; # params and GPU model are not logged |
| F2 | pipeline schematic | `f2_pipeline.py` | — (hand-drawn, no run data) | generated; placeholder in `main.tex`, to be redrawn as one row |
| T8 | score power: d′ of true model vs CSR | `tables.py` → `t8_power` | `scripts/power.py` report | generated; the current run has 5 families, the rerun all 8 |
| T4 | classification by model × family | `tables.py` → `t4_classification` | `compare/report.json`, `mincontrast/report.json` | generated |
| F3 | confusion matrix of the frozen classifier, Table 1 order, mechanisms outlined | `f3_confusion.py` | `classify/<model>/predictions.npz` | generated |
| F4 | the boundary on one rescaled axis x: detection (L test, PH trees, ParamNet), ParamNet's calls, gain over CSR per stratum | `f4_regime.py` | classifier predictions, stored L curves, `cutoffs.json`, evaluation `main` | generated |
| S-regime | F4 per family (+ cell against k); P(not detected) over (nbar, coordinate) with boundaries | `s_regime.py` | as F4 | generated |
| T5 | the fitted "looks like Poisson" boundary | `tables.py` → `t5_boundary` (fitted); near-closed forms by hand | `cutoffs.json` | fitted part generated; near-closed forms for the cluster families and LGCP not derived |
| T6 | fitted model vs Poisson, by regime stratum | `tables.py` → `t6_poisson_gap` | evaluation `main` | generated |
| T7 | main results (Section 6, `tab:main`): identification beside fidelity, ParamNet vs minimum contrast | `tables.py` → `m_main` | evaluation `main`, `classify/*/predictions.npz`; paper config `main_table` | generated |
| — | what PH adds, relative change in error (Section 6, `tab:topology`) | `summary.py` → `m_topology` | `results/v2/summary.csv` (story `filtrations`) | generated |
| F5 | detection, error and gain against u | `f5_main.py` | all classifiers / estimators, evaluation `main` | generated; not in `main.tex` (superseded by F4 and S-regime) |
| T9 | estimation error per family × estimator | `tables.py` → `t9_estimation` | `compare/report.json`, `mincontrast/report.json` | generated |
| T10 | what PH adds | `tables.py` → `t10_ph` | `compare/report.json`, evaluations `ph_ablation`, `ph_cell` | generated |
| — | the L test on the clouds sent to Poisson (Section 6, prose) | `check_poisson_routed.py` | evaluation `main`, the bank's stored L curves, `configs/departure_tables.npz` | generated (`results/v2/poisson_routed.md`); run it with `PAPER_CONFIG`, not via `make.py` -- it writes prose numbers, not a display |
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
