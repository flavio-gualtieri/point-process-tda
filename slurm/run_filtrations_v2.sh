#!/bin/bash
# The three-point filtration contrast behind Section 4's "two complementary filtrations": the headline
# network on alpha alone, on DTM_10 alone, and on both (already trained). 16 new GPU units.
#
#   bash slurm/run_filtrations_v2.sh          # submit the chain
#   DRY=1 bash slurm/run_filtrations_v2.sh    # print the sbatch calls, submit nothing
#
# Parallel wherever the data allows:
#   prebuild_alpha, prebuild_dtm10   no dependency on each other -- both start at once. Each builds its
#                                    model's network input once (~30 min, ~80G) so the 16 training units
#                                    read a finished cache instead of 8 of them racing to build the same one.
#   train_filt   -> both prebuilds   array of all 16 units, UNTHROTTLED: 2 classifiers + 14 estimators
#                                    are mutually independent, so they run as wide as `sae` allows.
#   compare      -> train_filt       ~35s; rewrites compare/report.json, so e2e waits rather than racing it.
#   e2e_filt     -> compare          12 shards, unthrottled: both images vs alpha vs DTM_10 on the same
#                                    1440 clouds as `fidelity`, so every difference is paired.
#   e2e_merge    -> e2e_filt         shards -> clouds.csv, report.json, summary.md
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
  else JOB[$key]=$("${cmd[@]}"); printf '%-18s %s\n' "$key" "${JOB[$key]}"; fi
}
py="python -u"
single='THREADS=1'
SHARDS=12

echo "data $CLOUDFORGER_DATA  results $CLOUDFORGER_RESULTS  configs $CLOUDFORGER_CONFIGS"
echo "untrained gpu units: ${train_gpu_pipeline:-none}"
[ -n "${train_gpu_pipeline:-}" ] || { echo "nothing to train -- stop"; exit 0; }

for m in fusion_curves_alpha fusion_curves_dtm10; do
  submit prebuild_${m#fusion_curves_} "" $CPU_SBATCH -c 8 --mem 128G -t 4:00:00 -- \
    $py scripts/train.py --config pipeline.yaml prebuild --model $m
done
submit train_filt "prebuild_alpha prebuild_dtm10" $GPU_SBATCH -c 8 --mem 192G -t 12:00:00 \
  --array="$train_gpu_pipeline" -- \
  $py scripts/train.py --config pipeline.yaml unit --kind gpu --index '{task}'
submit compare "train_filt" $CPU_SBATCH -c 4 --mem 48G -t 2:00:00 -- $py scripts/compare.py --config pipeline.yaml
submit e2e_filt "compare" $CPU_SBATCH -c 32 --mem 160G -t 12:00:00 --array=0-$((SHARDS - 1)) --export=ALL,$single -- \
  $py scripts/endtoend.py --set fidelity_filtrations --shard '{task}'/$SHARDS
submit e2e_merge "e2e_filt" $CPU_SBATCH -c 4 --mem 32G -t 1:00:00 -- \
  $py scripts/endtoend.py --set fidelity_filtrations --merge
echo "logs: $LOGS/"
