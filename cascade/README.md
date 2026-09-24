# cascade — classify → sub-classify → infer, scored by a Wasserstein energy score

An experiment built **on top of** ParamNet, not part of it.

```
                  stage 1                   stage 2 (per group, cascade-trained)      stage 3 (per family)
cloud ──►  poisson ─────────────────────────────────────────────────────────────►  nbar̂ = n  (MLE)
      ──►  clustered ──►  thomas | nested | lgcp | reject (→ poisson) ───────────►  θ̂ (log-scale regressors)
      ──►  repulsive ──►  matern2 | reject (→ poisson) ──────────────────────────►  θ̂
                                                                                      │
evaluation:  simulate from (familŷ, θ̂), the true model, and CSR  ──►  Wasserstein energy score per cloud
```

## Separability contract

- Everything lives in `cascade/` (code, configs, results) and `data/cascade/` (features, gitignored).
- `cascade/` **imports** `cloudforger` read-only: bank paths, the theta split, the summary
  functions, the samplers, the LGCP grid rule. It reads `data/bank/` and `results/classify/`,
  and writes nothing outside the two directories above.
- Nothing in `src/`, `scripts/`, `slurm/` or `configs/` imports or mentions `cascade/`.
  Deleting `cascade/` and `data/cascade/` removes it completely.
- Results go to `cascade/results/`, not `results/`, so `scripts/regimes.py` never picks them up.

## Running

```bash
python cascade/features.py --workers 16        # once: data/cascade/features/<family>.npz, whole bank
sbatch cascade/run.sh                          # stages 1-3, pipeline, evaluation for default.yaml
CONFIG=cascade/configs/<name>.yaml sbatch cascade/run.sh
FROM=stage3 CONFIG=... sbatch cascade/run.sh   # re-run from a stage onward
python cascade/stage1.py benchmark             # existing 5-way PH runs collapsed to 3 groups
```

Stage by stage: `stage1.py train`, `stage2.py`, `stage3.py`, `pipeline.py`, `evaluate.py`, each with
`--config`. Heavy fits belong on a compute node: the login node throttles a user to about one core.

## Where each choice lives

**Every modelling choice is a key in [`configs/default.yaml`](configs/default.yaml)**, commented
with why it was made and what the alternatives are. One config = one run: outputs go to
`cascade/results/<name>/<stage>/`, and each stage copies the config there as `config.yaml`. An
ablation is a copied config with one line changed and a new `name`. `configs/smoke.yaml` is the
same pipeline with fast models and tiny sizes, only for checking the code runs.

| file | role |
|---|---|
| `common.py` | config, data table, the prior (`balanced_groups`), models, theta-grouped cross-fitting |
| `features.py` | 111 classical columns: `L(r)−r` on the fixed axis, `L,F,G,J` on `u = r√n`, `log n` |
| `stage1.py` | poisson / clustered / repulsive; out-of-fold train predictions for cascade training |
| `stage1_nn.py`, `stage1_nn.sh` | the same stage with cloudforger's PHNet (PersLay) on a GPU, 3-class, prior-weighted CE |
| `regime.py` | stage 1's routing curve per family vs delta-tilde, and the `training: regime` pools |
| `stage2.py` | family within the routed group, trained on what stage 1 **routes** there (+ `reject`) |
| `stage3.py` | per-family parameter regressors on log targets; poisson is the MLE `nbar̂ = n` |
| `pipeline.py` | routing (shared by training and test time) and the assembled per-cloud output |
| `simulate.py` | (family, θ) → patterns with the bank's own samplers, same `n` conditioning |
| `wasserstein.py` | the evaluation metric — **its docstring documents every choice** |
| `evaluate.py` | energy score of fit / oracle / CSR on a sample of test clouds, regret and skill |

### Cascade training, concretely

Stage 1 is cross-fitted on the train split (folds grouped by theta). Its **out-of-fold**
predictions route the train clouds, and each stage-2 classifier trains on exactly the clouds routed
to its group. That includes the misrouted ones, mostly near-CSR poisson clouds, which get a `reject`
class. In-sample predictions would route almost every train cloud correctly, so stage 2 would never
see the misrouted inputs it meets at test time. Stage 2 is cross-fitted the same way, so stage 3
can train on the clouds the pipeline routes to each family. Sample weights are the prior's,
computed **before** routing, so the routed set keeps the mix stage 1 actually sends. The
near-CSR filtering from the original idea is therefore implicit: nothing is cut at a hand-picked
delta.

### Regime training (`training: regime`, configs `regime.yaml`, `nn.yaml`)

Instead of routing each train cloud, estimate on held-out data how often stage 1 routes clouds
of a family at a given delta-tilde to each group, and train downstream on **all** clouds past the
boundary where that reaches 50%. Reject candidates (near-CSR clouds, poisson, the other group)
enter stage 2 weighted by how often they would really arrive. With the HGB stage 1, the Thomas
stage-3 pool grows from 11k (cascade) to 38k. No out-of-fold predictions are needed, which is
what lets the neural stage 1 plug in.

### The metric, in one paragraph

Plain `W(observed, simulated)` is large even for the true model: two independent patterns of one
stationary process don't line up point by point. So the score is an **energy score** with W as
the distance between patterns: `ES = E W(S, x) − ½ E W(S, S′)`, i.e. distance to the model's
patterns minus the model's own spread. It is only compared across models on the **same** cloud:
`regret = ES(fit) − ES(oracle)` and `gain = ES(csr) − ES(oracle)`, pooled into
`skill = 1 − Σregret / Σgain` (1 = as good as the truth, 0 = no better than CSR). W itself is W₁
between the two patterns as normalized measures. It is computed exactly by assignment after thinning both
patterns to a common size, and thinning preserves the pair correlation. Caveats and how to change
each piece are in [`wasserstein.py`](wasserstein.py).

## Stage-1 results so far (test split, 120k clouds, balanced-groups prior)

| model | features | train clouds | balanced acc |
|---|---|---|---|
| logistic regression | classical, 111 cols | 70k (thetas < 7000) | **0.793** |
| best collapsed 5-way PH net (dtm_k10 PersLay H0+H1) | persistence | 370k | 0.774 ± 0.002 |
| alpha/rips PersLay H0+H1 | persistence | 370k | 0.757–0.760 |
| PH image H1 only (any filtration) | persistence | 370k | 0.59–0.64 |

The collapsed nets were trained for the 5-way family task, not this one, so this is not a
head-to-head of feature sets. It does show that stage 1 needs no neural network: a linear model
on classical features, with a fifth of the training data, already beats every PH arm. Full per-family and per-delta numbers: `cascade/results/benchmark_stage1.json`.
