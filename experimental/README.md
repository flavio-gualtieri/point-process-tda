# experimental — work beside the paper version

Nothing here is part of the paper's pipeline, and nothing here writes outside `experimental/`:
each study reads a finished pipeline run (its stored predictions and evaluation clouds) and writes
only to its own `results/` (gitignored). It imports `cloudforger` like the scripts do, so it
always uses the paper's scoring code, never a copy.

## softroute — does simulating from the classifier's mixture beat its argmax?

The paper's pipeline routes each cloud to `argmax_f P(f | x)` and simulates from that family at its
estimate. `softroute/` scores mixtures `Σ_f w_f P_f(θ̂_f)` instead (`w = P(f | x)`, tempered, top-2,
uniform, and the true-family ceiling), with the paper's kernel score on the paper's own clouds. The
kernel score is quadratic in the mixture weights, so every weighting is scored exactly from one set
of simulations per family (see the module docstring), and `mixture_terms.npz` keeps what any other
weighting needs to be scored offline.

**Status.** One full run exists (1556 clouds, `extensions/results/softroute/default/`, from before
this directory moved): soft routing adds ~+0.01 skill over argmax on the strong-signal clouds, at
the edge of resolution. Not in the paper yet.

**To finish it:**

```bash
python experimental/softroute/softroute.py --limit 16 --workers 8          # smoke, minutes
sbatch -c 32 --mem 128G -t 12:00:00 -o logs/softroute_%j.out \
       slurm/job.sh python -u experimental/softroute/softroute.py          # full run
```

1. Point `paper_run` in `softroute/config.yaml` at the pipeline run to use (now: the paper run as it
   stands; after the rerun, `<results>/pipeline`, with the clouds renamed as the config notes).
2. Weightings to add go in `weightings()`; they score offline from `mixture_terms.npz` too.
3. If it goes in the paper, promote it: a pipeline evaluation option (`evaluation.sets`) in
   `configs/pipeline.yaml`, and the scoring loop into `cloudforger.scores`.
