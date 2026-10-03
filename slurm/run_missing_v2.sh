#!/bin/bash
# The results main.tex still marks missing, for the parts the current code can already produce:
# the headline network WITHOUT the persistence images (nn_curves), as classifier, as estimator of
# every family, and end to end on the fidelity clouds.
#
#   bash slurm/run_missing_v2.sh          # submit the chain
#   DRY=1 bash slurm/run_missing_v2.sh    # print the sbatch calls, submit nothing
#
# Stages (-> waits on):
#   train_gpu     the 8 untrained nn_curves units (1 classifier + 7 estimators); plan.py picks the indices
#   compare       -> train_gpu    nn_curves into the classification / estimation tables (Table 2's rows)
#   e2e           -> compare      evaluation/fidelity_curves, 12 shards: fusion vs curves, same 1440 clouds
#   e2e_merge     -> e2e          the shards -> clouds.csv, report.json, summary.md
#
# Every stage skips work whose output exists, so re-running after a failure submits the same chain.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

export CLOUDFORGER_CONFIGS="$ROOT/configs/v2"
export CLOUDFORGER_DATA="$ROOT/data_v2"
export CLOUDFORGER_RESULTS="$ROOT/results_v2"

source slurm/env.sh
eval "$(python slurm/plan.py pipeline)"
RUN=$(basename "$CLOUDFORGER_RESULTS")
LOGS="logs/$RUN"
mkdir -p "$LOGS"
declare -A JOB

submit() {                                   # submit <key> "<keys it waits on>" <sbatch options...> -- <command...>
  local key=$1 deps=$2 opts=() ids=() d
  shift 2
  while [ "$1" != "--" ]; do opts+=("$1"); shift; done
  shift
  for d in $deps; do [ -n "${JOB[$d]:-}" ] && ids+=("${JOB[$d]}"); done
  local dep=()
  [ ${#ids[@]} -gt 0 ] && dep=("--dependency=afterok:$(IFS=:; echo "${ids[*]}")")
  local log="$LOGS/${key}_%j.out"
  [[ " ${opts[*]} " == *" --array="* ]] && log="$LOGS/${key}_%A_%a.out"
  local cmd=(sbatch --parsable -J "${RUN}_${key}" -o "$log" "${dep[@]}" "${opts[@]}" slurm/job.sh "$@")
  if [ -n "${DRY:-}" ]; then JOB[$key]="<$key>"; echo "${cmd[*]}"
  else JOB[$key]=$("${cmd[@]}"); printf '%-16s %s\n' "$key" "${JOB[$key]}"; fi
}
py="python -u"
single='THREADS=1'
SHARDS=12

echo "data $CLOUDFORGER_DATA  results $CLOUDFORGER_RESULTS  configs $CLOUDFORGER_CONFIGS"
echo "untrained gpu units: ${train_gpu_pipeline:-none}"

if [ -n "${train_gpu_pipeline:-}" ]; then
  submit train_curves "" $GPU_SBATCH -c 8 --mem 192G -t 12:00:00 --array="$train_gpu_pipeline"%4 -- \
    $py scripts/train.py --config pipeline.yaml unit --kind gpu --index '{task}'
fi
submit compare "train_curves" $CPU_SBATCH -c 4 --mem 48G -t 2:00:00 -- $py scripts/compare.py --config pipeline.yaml
submit e2e_curves "compare" $CPU_SBATCH -c 32 --mem 160G -t 12:00:00 --array=0-$((SHARDS - 1)) --export=ALL,$single -- \
  $py scripts/endtoend.py --set fidelity_curves --shard '{task}'/$SHARDS
submit e2e_curves_merge "e2e_curves" $CPU_SBATCH -c 4 --mem 32G -t 1:00:00 -- \
  $py scripts/endtoend.py --set fidelity_curves --merge
echo "logs: $LOGS/"
