# extensions — exploratory runs beside the paper version

**The paper version is `oneshot/` and `oneshot/results/default/`.** Nothing in this directory writes
there or changes what it contains. Everything here reads the paper run's stored outputs as files and
writes only under `extensions/results/` and `extensions/logs/`.

To keep it that way while `oneshot/` and `cascade/` are still being edited, nothing here imports
them: `common.py` reads the paper run's predictions directly and carries copies of the two helpers
it needs from `cascade/evaluate.py`. The only outside code is `cloudforger` (`src/`) and the scoring
package `cascade/scoring/`, which is self-contained by design. Jobs set `PYTHONDONTWRITEBYTECODE`, so
not even bytecode caches appear outside this directory.

| directory | question | run |
|---|---|---|
| `softroute/` | does simulating from the classifier's mixture over families beat its argmax? Scored with the paper's kernel score on the paper's own evaluation clouds, with the pipeline behind skill 0.93 frozen in `config.yaml` | `sbatch extensions/softroute/softroute.sh` |
| `learned/` | do learned summaries of the raw points (DeepSets, a GNN) beat the hand-built features? `smoke.yaml` is a 10%-of-the-bank smoke test | `bash extensions/learned/submit.sh` |

Each script's docstring states its method; each config states its choices.
