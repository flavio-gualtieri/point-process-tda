# cascade — classify → sub-classify → infer, each piece trained on its own

An experiment built **on top of** ParamNet, not part of it.

```
                stage 1 (all clouds)     stage 2 (in-regime clouds)            stage 3 (in-regime clouds)
cloud ──►  poisson ─────────────────────────────────────────────────────────►  nbar̂ = n  (MLE)
      ──►  clustered ──►  thomas | nested | lgcp  ───────────────────────────►  θ̂ per family (log-MSE)
      ──►  repulsive ──►  matern2 (one family, no model) ────────────────────►  θ̂
```

Each component is trained **separately**, on bank clouds only, and never sees another component's
predictions. Stage 1 is the filter and trains on everything. Stages 2 and 3 train only on clouds
**in regime**: those where stage 1 reliably recognises the family's group, according to a frozen
rule on the bank's parameters. The pieces are put together only afterwards (`pipeline.py`), and the
end-to-end Wasserstein score (`evaluate.py`) is deferred.

## Separability contract

- Everything lives in `cascade/` (code, configs, results, logs) and `data/cascade/` (features, gitignored).
- `cascade/` **imports** `cloudforger` read-only: bank paths, the theta split, the summary
  functions, the samplers, the LGCP grid rule, and the network and its training loop. It reads
  `data/bank/`, `data/classical/`, `data/featurization/` and `results/classify/`, and writes nothing
  outside the two directories above.
- Nothing in `src/`, `scripts/`, `slurm/` or `configs/` imports or mentions `cascade/`.
- Results go to `cascade/results/`, not `results/`, so `scripts/regimes.py` never picks them up.

## Running (everything through SLURM; `run.sh` itself only calls `sbatch`)

```bash
bash cascade/run.sh                      # regime analysis -> components (hgb CPU + nn GPU) -> assemble
FROM=stage1 bash cascade/run.sh          # also redo features + stage 1 first
MODELS=hgb bash cascade/run.sh           # skip the GPU components
CONFIG=cascade/configs/<name>.yaml bash cascade/run.sh
CONFIG=cascade/configs/nn_curves.yaml sbatch cascade/stage1_nn.sh    # a network stage 1, for comparison
```

| script | where | what |
|---|---|---|
| `stage1.sh` | CPU | `features.py` (skipped when on disk) + `stage1.py train` |
| `regime_analysis.sh` | CPU | scores coordinates, writes `cutoffs.json` and the figures |
| `components.sh` | CPU array over τ | `components.py --model hgb`: stage 2 + stage 3 |
| `components_nn.sh` | GPU array over τ | `components.py --model nn` |
| `assemble.sh` | CPU | `pipeline.py`: components put together per τ, plus the sweep summary |

## Where each choice lives

**Every modelling choice is a key in [`configs/default.yaml`](configs/default.yaml)**, commented
with why it was made and the alternatives. `logreg.yaml` and `nn_curves.yaml` are stage-1-only
comparison runs. Old designs (cascade training, reject classes) are in `results/_archive/`.

| file | role |
|---|---|
| `common.py` | config, data table, the prior (`balanced_groups`), sklearn models, theta-grouped cross-fitting |
| `features.py` | 111 classical columns: `L(r)−r` on the fixed axis, `L,F,G,J` on `u = r√n`, `log n` |
| `stage1.py` | poisson / clustered / repulsive; out-of-fold train predictions; `benchmark`, paired `compare` |
| `stage1_nn.py` | the same stage with cloudforger's network |
| `regime.py` | physical coordinates (`DERIVED`), the cutoff fit, and `Cutoffs` (the frozen rule) |
| `regime_analysis.py` | which coordinate explains stage 1's routing best; exports `cutoffs.json` |
| `components.py` | stage 2 (clustered classifier) and stage 3 (per-family regressors), hgb or nn, one τ at a time |
| `nn.py` | cloudforger's PHNet on any subset of bank rows (classification or log-target regression) |
| `pipeline.py` | assembly per τ and model pair; `sweep/` summary table and figure |
| `simulate.py`, `wasserstein.py`, `evaluate.py` | the end-to-end Wasserstein energy score (deferred; so far it has no power) |

### The cutoff, concretely

From the `default` stage 1's out-of-fold routing on the train split (370k clouds, never test), each
family gets a rule on `u = log(x) + a·log(n̄)`: a coordinate x picked by the regime analysis
(Thomas: cluster overlap ω; nested: ω_inner; Matérn II and LGCP: ζ), corrected for n̄ with the
exponent `a` fitted by logistic regression. That makes it the 2-D (x, n̄) boundary written as one
number `x·n̄^a`. A monotone fit of P(routed to own group | u) then gives, for every τ in
`regime.taus`, the boundary where it reaches τ. A component at τ trains on the clouds past it,
and τ = 0 means no filtering. `components.py` scores every component twice on test: on in-regime clouds,
and on **all** clouds of its families. The second set is the same at every τ, so it is the fair
comparison across τ.

## Stage 1 (test split, 120k clouds, balanced-groups prior)

`python cascade/stage1.py compare --runs nn_curves logreg default nn` (paired, bootstrap over thetas):

| stage 1 | balanced acc |
|---|---|
| network, L+F+G+J curves (fixed grid) | 0.7961 |
| HGB, classical features (`default`) | 0.7957 (tie) |
| logistic regression, classical features | 0.7940 |
| network, PersLay dtm_k10 H0+H1 | 0.7737 |

Three different models tie at ~0.795, so that is close to what the task allows. `default` (HGB)
is used because it gives out-of-fold train predictions cheaply, and the cutoffs are fitted on those.
