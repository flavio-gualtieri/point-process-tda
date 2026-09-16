#!/bin/bash
# slurm/dv3_pi_logn_submit.sh -- submit the classification persistence-image
# cells that include the point count n(x) (slurm/dv3_pi_logn_train.sh).
#
# Two cells, one GPU each, 10 seeds per cell at 5 at a time:
#   task 0   H0+H1 + n(x)   -> results/dv3_classify/dtm_k<K>/pi_multik_h01_nocc_norm_logn
#   task 1   H0    + n(x)   -> results/dv3_classify/dtm_k<K>/pi_multik_h0_nocc_norm_logn
#
# Cells already on disk are skipped seed by seed, so a re-submit only runs what
# is missing. Not an sbatch script: run it on the login node from the repo root.
#
#   bash slurm/dv3_pi_logn_submit.sh          # both cells, DTM k=5
#   bash slurm/dv3_pi_logn_submit.sh h01      # only H0+H1 + n(x)
#   bash slurm/dv3_pi_logn_submit.sh h0       # only H0 + n(x)
#   DTM_K=10 bash slurm/dv3_pi_logn_submit.sh # another filtration, once its diagrams exist
#
# Both cells fit inside the sae per-user GPU cap, so they go in one array with
# no dependency between them.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

K="${DTM_K:-5}"
mode="${1:-all}"
case "$mode" in
  all) ARRAY=0-1 ;;
  h01) ARRAY=0 ;;
  h0)  ARRAY=1 ;;
  *) echo "usage: bash $0 [all|h01|h0]" >&2; exit 2 ;;
esac

# Fail on the login node rather than after the queue wait: without clouds.pkl
# the n(x) join cannot run, and that bundle is exactly what the earlier
# (no-n) classification cells did not need.
missing=()
for s in train A B C; do
  for p in "data/dv3/$s/_classify/dtm_k${K}/diagrams.pkl" "data/dv3/$s/_classify/clouds.pkl"; do
    [[ -f "$p" ]] || missing+=("$p")
  done
done
if (( ${#missing[@]} )); then
  printf 'missing input(s) -- not submitting:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  printf 'build the merged classification bundle with:\n' >&2
  printf '  python scripts/processing/dv3_classification_bundle.py\n' >&2
  exit 1
fi

job=$(DTM_K="$K" PAR=5 sbatch --parsable --array="$ARRAY" --export=ALL slurm/dv3_pi_logn_train.sh)
echo "classification + n(x): ${job} (tasks ${ARRAY}, DTM k=${K})" >&2
echo "watch:   squeue -u \$USER -n dv3_pi_logn" >&2
echo "logs:    logs/dv3_pi_logn_${job}_*.out" >&2
echo >&2
echo "when it finishes, score it against the stratified test set:" >&2
echo "  python scripts/evaluate_testset.py results/dv3_classify/dtm_k${K}/pi_multik_h01_nocc_norm_logn \\" >&2
echo "                                     results/dv3_classify/dtm_k${K}/pi_multik_h01_nocc_norm" >&2
echo "then fold it into the tables:" >&2
echo "  python scripts/rescore_results.py && python scripts/status_tables.py" >&2
