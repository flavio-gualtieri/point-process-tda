#!/bin/bash
# The single-model ablation on the v2 bank: one network per input representation, as the 8-way classifier
# and as the estimator of every structured family, over 3 seeds. 40 inputs x 3 seeds x 8 tasks = 960 GPU units.
#
#   bash slurm/run_ablation_v2.sh          # submit the chain
#   DRY=1 bash slurm/run_ablation_v2.sh    # print the sbatch calls, submit nothing
#   MODELS="pi_alpha_h1 ..." bash slurm/run_ablation_v2.sh    # only these inputs
#
# Parallel per input, not per stage. Each input's cache is built once and its own 24 units wait only on
# that build, so training starts on the first finished cache instead of the last:
#   prebuild_<m>   no dependencies -- all 40 start at once (~12 min, ~65G at the fusion sizes)
#   train_<m>   -> prebuild_<m> only. 24 units, unthrottled.
#   compare     -> every train_<m>, plus the pipeline's own compare, whose cutoffs.json the
#                  ablation's regime subsets are frozen against (`regime.from: pipeline`).
#
# Resources are sized from what the comparable v2 units actually used, not from run_all.sh's defaults:
# the two-filtration training units peaked at 21-34G against a 192G request and finished in 4-35 min, and
# the prebuilds peaked at 54-65G in ~12 min. Asking for 64G/4h instead of 192G/24h lets four jobs share a
# node's memory and makes every unit backfillable, which is what gets 960 of them through a busy partition.
#
# Every stage skips work whose output exists, so re-running after a failure submits only what is missing.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

export CLOUDFORGER_CONFIGS="$ROOT/configs/v2"
export CLOUDFORGER_DATA="$ROOT/data_v2"
export CLOUDFORGER_RESULTS="$ROOT/results_v2"

source slurm/env.sh
RUN=$(basename "$CLOUDFORGER_RESULTS")
LOGS="logs/${RUN}_ablation"
mkdir -p "$LOGS"

# model -> the indices of its untrained units, from the config's own unit list
mapfile -t PLAN < <(python - <<'PY'
from collections import defaultdict
from cloudforger.paths import read_config
from cloudforger.pipeline.units import units, done
rc = read_config("ablation.yaml")
by = defaultdict(list)
for i, (task, model, fam, seed) in enumerate(units(rc, "gpu")):
    if not done(rc, task, model, fam, seed):
        by[model].append(str(i))
for m, idx in by.items():
    print(f"{m} {','.join(idx)}")
PY
)
[ ${#PLAN[@]} -gt 0 ] || { echo "nothing to train -- stop"; exit 0; }

if [ -n "${MODELS:-}" ]; then                      # keep only the named inputs
  keep=(); for row in "${PLAN[@]}"; do
    [[ " $MODELS " == *" ${row%% *} "* ]] && keep+=("$row")
  done
  PLAN=("${keep[@]}")
fi

echo "data $CLOUDFORGER_DATA  results $CLOUDFORGER_RESULTS  configs $CLOUDFORGER_CONFIGS"
echo "${#PLAN[@]} inputs, $(printf '%s\n' "${PLAN[@]}" | awk -F' ' '{n+=gsub(/,/,",")+1} END{print n}') units"

py="python -u"
train_ids=()

submit() {                                         # submit <key> "<dependency ids>" <opts...> -- <cmd...>
  local key=$1 deps=$2 opts=()
  shift 2
  while [ "$1" != "--" ]; do opts+=("$1"); shift; done
  shift
  local dep=()
  [ -n "$deps" ] && dep=("--dependency=afterok:$deps")
  local log="$LOGS/${key}_%j.out"
  [[ " ${opts[*]} " == *" --array="* ]] && log="$LOGS/${key}_%A_%a.out"
  local cmd=(sbatch --parsable -J "${RUN}_abl_${key}" -o "$log" "${dep[@]}" "${opts[@]}" slurm/job.sh "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}"; echo "<$key>"
  else "${cmd[@]}"; fi
}

for row in "${PLAN[@]}"; do
  m=${row%% *} idx=${row#* }
  pb=$(submit "prebuild_$m" "" $CPU_SBATCH -c 8 --mem 96G -t 2:00:00 -- \
    $py scripts/train.py --config ablation.yaml prebuild --model "$m" | tail -1)
  tr=$(submit "train_$m" "$pb" $GPU_SBATCH -c 8 --mem 64G -t 4:00:00 --array="$idx" -- \
    $py scripts/train.py --config ablation.yaml unit --kind gpu --index '{task}' | tail -1)
  train_ids+=("$tr")
  printf '%-26s prebuild %-10s train %-10s (%s units)\n' "$m" "$pb" "$tr" "$(($(grep -o ',' <<<"$idx" | wc -l) + 1))"
done

deps=$(IFS=:; echo "${train_ids[*]}")
submit compare "$deps" $CPU_SBATCH -c 4 --mem 64G -t 4:00:00 -- \
  $py scripts/compare.py --config ablation.yaml >/dev/null
echo "compare queued after all ${#train_ids[@]} training arrays"
echo "logs: $LOGS/"
