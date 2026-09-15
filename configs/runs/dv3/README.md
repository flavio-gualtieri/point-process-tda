# DV3 runs: evaluation across regimes

Every config here sets `data.source: dv3`. Training uses `data/dv3/train/<group>/`
with the train/val split fixed at generation. Each method is then scored on
the three DV3 test products separately, and writes one per-pattern
`predictions_<set>.npz` for each:

| set | what it measures | regime axes |
|---|---|---|
| A | risk averaged over the prior (replaces the old random test slice) | δ̃ band × n̄ band |
| B | error, bias and s.d. at fixed θ (400 replicates per cell) | n̄ × scale × δ̃ |
| C | error and detection power along ladders down to exact CSR | δ̃ (+ CSR anchors) |

`results.json` keeps `test_loss` / `test_accuracy`, which now hold the set-A value,
and adds an `eval_sets` block with whole-set numbers for A, B and C.
`scripts/evaluate_regimes.py` produces the per-regime tables.

## Order

```bash
# 0. DV3 data exists (scripts/generation/dv3.py all). Diagrams for every set,
#    INCLUDING train (dv3_diagrams.py defaults to A B C only):
python scripts/processing/dv3_diagrams.py plan --n-jobs 20 --sets train A B C
#    ... run / merge as in its docstring, with the same --sets

# 1. Classification only: merge the families into data/dv3/<set>/_classify/
python scripts/processing/dv3_classification_bundle.py            # re-run after new diagram tags

# 2. Train. Per family (thomas | nested | matern2 | lgcp); run vihrs first,
#    because it builds the L(r) cache that fusion_k5 reads:
python scripts/train.py configs/runs/dv3/thomas/vihrs.yaml           --seed 9371
python scripts/train.py configs/runs/dv3/thomas/summstats_lfgj.yaml  --seed 9371
python scripts/train.py configs/runs/dv3/thomas/pi_multik.yaml       --seed 9371
python scripts/train.py configs/runs/dv3/thomas/fusion_k5.yaml       --seed 9371
python scripts/train.py configs/runs/dv3/thomas/mincontrast.yaml     # fits cached once, reused by all seeds
#    Classification: same pattern with configs/runs/dv3/classify/*.yaml
#
#    The classical summary-function baselines (vihrs CNN on L | F | G | L+F+G+J,
#    both F/G grids, both tasks: 7 arms x 5 groups x 10 seeds) run as one batch
#    and need no diagrams; the prep stage also builds the L cache fusion_k5 reads.
#    Parameter estimation goes first, classification is queued behind it:
bash slurm/dv3_classical_submit.sh                             # or: ... params  /  ... classify [JOBID]
python scripts/dv3_matrix.py                                   # refresh docs/status/experiment_matrix.xlsx
#
#    Persistence-feature cells (one vectorization x ONE DTM k x ONE homology dim:
#    <group>/ph_{pi,betti,landscape}.yaml templates). The helper computes the
#    missing TRAIN-set diagrams first, then runs the persistence-image cells
#    before Betti curves and landscapes:
bash slurm/dv3_ph_submit.sh                                    # k=5; or: ... pi  /  ... params;  DTM_K=10 bash ...

# 3. Classical CSR test on L, G, F (no training, no diagrams needed)
python scripts/classical_detection.py --jobs 16

# 4. Regime tables, paired seed-wise against a reference
python scripts/evaluate_regimes.py configs/runs/dv3/thomas/fusion_k5.yaml \
    --methods pi_multik vihrs_checkpointed vihrs_lfgj mincontrast mincontrast_g
python scripts/evaluate_regimes.py configs/runs/dv3/classify/pi_multik.yaml \
    --methods pi_multik vihrs vihrs_lfgj envelope_L envelope_LGF --name detect
```

Output goes to `results/<process>/_regimes/<name>/`: `summary.json`, one CSV per
(task, set), `paired_*.csv` (Wilcoxon, Holm-adjusted within each set/family),
and `figures/`.

## Reading the tables

- **params**: `loss` is the normalised loss L, computed with the train-split
  log+z statistics, so it is on the same scale for every method. `medae` is
  the median |error| in the same units. It stays readable when a classical fit
  diverges near CSR. On B and C also look at `bias_std` and `sd_std`
  separately, and at `rel_bias` / `log_rmse` of the natural parameter.
  `fail_rate` counts non-finite estimates. These are failures, not dropped rows.
- **classify**: every B cell and C rung holds a single family, so a cell's
  accuracy is that family's recall at that θ. `A/_all` is the balanced
  headline number.
- **detect**: `power` is P(score > threshold), where the threshold is set at 5%
  on the even replicates of the C CSR anchors (at matching n̄) and checked on
  the odd ones (`C/poisson/...` rows report that held-out size). Classifiers
  score 1 − P(poisson). The envelope tests score S, and also report
  `power_nominal` at their own S > 1 threshold.
- Rows marked `*` have fewer than 20 patterns.

## Methods wired for DV3

Wired: the pi_multik family (`pi_multik`, `_towers`, `_scaleconv`,
`_earlyfusion`, `_dimsplit`, including the `include_lfunc` / `include_fgfunc` /
`curve_channels` arms), `vihrs` (L, or L+F+G+J via `summary_channels`) for both
tasks, `betti_cnn` and `vec_multik` (e.g. landscapes) for both tasks, and
`mincontrast`, `mincontrast_g`, `mincontrast_nested`, `mincontrast_g_nested`. Any other method is refused with an explicit error
instead of silently falling back to a random split. To port one, copy the
`use_dv3` / `eval_paths` handling in `experiments/pi_multik/pi_multik.py`
and set `supports_dv3 = True`.
