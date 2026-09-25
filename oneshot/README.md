# oneshot — one classifier over every family, then per-family inference

Replaces the cascade's three-group first layer (see `cascade/README.md` for the history). One N-way
classifier covers all eight bank families; its P(poisson | x) **is** the CSR filter. A cloud called
poisson gets n̄̂ = n (the exact MLE); any other call goes to that family's estimator, trained on
**all** of the family's train clouds.

```
cloud ──► classifier (8-way) ──► poisson ─────────────► n̄̂ = n
                              └─► family f ──► estimator_f ──► θ̂
```

The regime map survives as an analysis: per family, where the reference classifier **detects** it
(does not call it poisson), as a monotone function of u = log(coordinate) + a·log(n̄). The cutoff is
a rule on θ alone, so every model is scored on the same in-regime clouds.

## Plug and play

Nothing about *which* models to use is in code. `configs/default.yaml` holds:

| key | what | add one by |
|---|---|---|
| `inputs` | feature tables for table learners (`classical`, `ph` of a filtration) | one entry; a new *source* = a loader in `inputs.py` `SOURCES` |
| `models` | learner + input: `hgb`, `linear` (tables, CPU) or `nn` (PHNet on curves / images / PersLay, GPU) | one entry; a new *learner* = a class in `learners.py` `LEARNERS` |
| `classify` | models trained as the 8-way classifier | listing the name |
| `estimate` | models trained as estimators, for every family but poisson | listing the name |
| `compare.estimators` | estimator assignments for the pipelines: a model for every family, `best` (lowest val error per family), or a mapping family → model | listing it |

Each (task, model[, family]) is an independent **unit** with a fixed output contract
(`core.py`: `predictions.npz` keyed by case_id). A unit that exists is skipped, so adding a model
and resubmitting trains only the new units, and `compare.py` re-scores everything against
everything, on the same test clouds, with paired bootstrap intervals.

A learner implements four methods over positional row indices (`learners.py` docstring):
`classify`, `crossfit` (or `None`), `regress`, `artifacts`.

## Running (SLURM; `submit.sh` only calls sbatch)

```bash
bash oneshot/submit.sh                                   # PH inputs -> untrained units (CPU + GPU arrays) -> compare
SKIP_PREPARE=1 bash oneshot/submit.sh                    # inputs already built
KIND=cpu bash oneshot/submit.sh                          # only table learners
CONFIG=oneshot/configs/smoke.yaml SKIP_PREPARE=1 bash oneshot/submit.sh   # every 250th theta, CPU only
python oneshot/train.py list --todo                      # what is left
sbatch oneshot/compare.sh                                # re-score at any point
```

`thin: k` in a config keeps every k-th θ (in every split) for quick runs.

## Files

| file | role |
|---|---|
| `core.py` | config, row table (8 families, cloudforger's θ split), targets, regime coordinates, prior weights, prediction I/O |
| `inputs.py` | feature-table sources |
| `learners.py` | `hgb`, `linear`, `nn` behind one interface |
| `prepare.py` | builds `ph` inputs (`cloudforger.vectorization.summaries`); `check` verifies every input exists |
| `train.py` | lists and trains units; per-unit `report.json` |
| `compare.py` | regime cutoffs, classifier / estimator / pipeline tables → `results/<name>/compare/summary.md` |

## Inputs it expects

- `data/bank/<family>/` for all eight families (`slurm/bank.sh`)
- `data/cascade/features/<family>.npz` (classical; `cascade/features.py --family F`)
- `data/featurization/<family>/<tag>/diagrams.npz` (`slurm/featurize_array.sh`) → `ph` inputs and PersLay
- `data/classical/<family>/<grid>/curves.npz` (`scripts/classical.py`) for curve networks

## Not yet here

- The end-to-end kernel score (`cascade/evaluate.py`) needs a per-cloud export (`family_hat`,
  `theta_hat`) from a pipeline; the pieces are in `compare.py` but the export is not written.
