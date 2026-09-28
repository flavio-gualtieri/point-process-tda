# Wrong family, right pattern — point-process fitting near complete spatial randomness

Code for the paper. Eight spatial point-process families are simulated on the unit square; one
classifier decides the family (its P(Poisson | x) is the CSR filter) and a per-family estimator
fits the parameters; each fit is then scored end to end by how well patterns simulated from it
reproduce an independent replicate (a kernel score on local configurations).

```
cloud ──► classifier (8-way) ──► poisson ─────────────► n̄̂ = n
                              └─► family f ──► estimator_f ──► θ̂ ──► simulate ──► kernel score vs held-out replicate
```

## Layout

```
configs/            every choice, as YAML: simulation, departure (+ fitted null tables), featurization,
                    classical, pipeline (models, regime, evaluation sets, min. contrast), ablation, scores;
                    smoke/ = small overrides for a quick end-to-end run
src/cloudforger/    the package
  simulation/         families, samplers, the bank (seeded, reproducible), train/val/test split
  departure/          CSR null tables and the departure statistic δ̃
  classical/          L, F, G, J summary functions; the classical feature table
  featurization/      persistence diagrams (Rips, alpha, DTM)
  vectorization/      persistence images, PersLay, PH summary vectors
  training/           the network (PHNet) and its data builders
  pipeline/           config, units, learners (trees, linear, networks), regime cutoffs
  scores/             kernel / Dawid–Sebastiani / energy scores, simulation from fits, skill
  baselines/          minimum contrast on Ripley's K
scripts/            one entry point per stage (below)
slurm/              run_all.sh (the whole chain), job.sh (the one worker), env.sh (cluster settings)
paper/              the paper, its figure/table scripts, and the method notes -- see paper/README.md
experimental/       work beside the paper (soft routing), reading a finished run
tests/              pytest suite, including an end-to-end run on a tiny fake bank
```

## Install

```bash
mamba env create -f environment.yml && mamba activate cloudforger     # or any env with pyproject's deps
pip install -e .
pytest tests/                                                         # ~10 min
```

## Reproduce

Everything, from simulation to end-to-end scores, is one chain of SLURM jobs. Data and results go
under two roots you name, so a run never writes into an earlier one:

```bash
SMOKE=1 bash slurm/run_all.sh                                             # minutes: every stage on a small bank
CLOUDFORGER_DATA=data_v2 CLOUDFORGER_RESULTS=results_v2 bash slurm/run_all.sh     # the paper run
FROM=train ... bash slurm/run_all.sh        # resume from a stage;  ONLY="compare endtoend" ...;  DRY=1 prints the jobs
python paper/scripts/make.py                # figures and tables (after pointing paper/paper.yaml at the results)
```

Every stage skips outputs that exist, so re-running after a failure only does what is missing.
Cluster specifics (environment, accounts, partitions) are all in `slurm/env.sh`.

| stage | script | writes |
|---|---|---|
| departure (optional; tables ship in `configs/`) | `departure.py` | `configs/departure_tables.npz` |
| simulate, merge | `simulate.py` | `<data>/bank/<family>/{points.npz, manifest.csv}` |
| relabel | `relabel.py` | δ̃ columns in every manifest |
| featurize, diagrams | `featurize.py` | `<data>/featurization/<family>/<filtration>/diagrams.npz` |
| curves | `classical.py` | `<data>/classical/<family>/<grid>/curves.npz` |
| tables | `tables.py` | `<data>/tables/{classical, ph_<filtration>}/<family>.npz` |
| train | `train.py` | `<results>/<run>/{classify, estimate/<family>}/<model>[/seed_<s>]/` |
| compare | `compare.py` | `<results>/<run>/compare/` — accuracy, estimation error, regime cutoffs |
| mincontrast | `mincontrast.py` | the classical baseline, as units the comparison and scores read |
| endtoend | `endtoend.py --set <name>` | `<results>/pipeline/evaluation/<name>/` — regret, gain, skill |
| power | `power.py` | `<results>/scores/power/` — can the score tell the true model from CSR? |

Runs (`<run>`) are `configs/pipeline.yaml` (the paper's pipeline) and `configs/ablation.yaml` (every
input representation as its own network, several seeds). A different config directory can override
any config file key by key: `CLOUDFORGER_CONFIGS=configs/smoke`.

The bank is reproducible from `configs/simulation.yaml`: every θ has its own seed stream, so any
θ can be regenerated alone and a strided bank (`stride`) is a subset of the full one. Across
machines, expect agreement up to floating-point rounding (last-digit differences in a few derived
parameters). `python scripts/simulate.py compare --other data` checks a regenerated bank against
another one.

## License

MIT — see [LICENSE](LICENSE).
