#!/bin/bash
# Re-simulation noise on skill: the `fidelity` set rescored at four more simulation seeds, so every
# number in Section 6 and Table 4 gets a seed s.d. beside its bootstrap-over-clouds interval.
#
#   bash slurm/run_rescore_v2.sh          # submit the chain
#   DRY=1 bash slurm/run_rescore_v2.sh    # print the sbatch calls, submit nothing
#   SEEDS="1 2" bash slurm/run_rescore_v2.sh   # only these seeds
#
# `fidelity` itself is seed 0 (configs/pipeline.yaml evaluation.seed) and is already scored, so these four
# complete the five. Each set holds `clouds` fixed (per_bin 60, clouds.seed 0) and moves only `seed`, so
# pick() returns the identical 1440 clouds and the five sets are paired cloud by cloud -- which is the
# whole point: the spread across seeds is re-simulation noise, not a different sample of clouds.
#
# CPU only. Nothing here touches results_v2/pipeline/compare or the trained units, so it is independent of
# the ablation's GPU work and of its compare (which writes results_v2/ablation/).
#
#   e2e_s<i>       12 shards, unthrottled; ~70 min each at the fidelity set's size
#   merge_s<i>  -> e2e_s<i>   that seed's shards -> clouds.csv, report.json, summary.md
#
# After every seed lands, point paper.yaml `repeat` at the five clouds.csv files (see
# paper/EXPERIMENTS_OUTSTANDING.md) and rerun paper/scripts/summary.py: summary.py's `seeded()` path then
# reports mean +/- s.d. over seeds and pools the seeds per cloud for the interval.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

export CLOUDFORGER_CONFIGS="$ROOT/configs/v2"
export CLOUDFORGER_DATA="$ROOT/data_v2"
export CLOUDFORGER_RESULTS="$ROOT/results_v2"

source slurm/env.sh
RUN=$(basename "$CLOUDFORGER_RESULTS")
LOGS="logs/${RUN}_rescore"
mkdir -p "$LOGS"

py="python -u"
single='THREADS=1'
SHARDS=12

submit() {                                         # submit <key> "<dependency ids>" <opts...> -- <cmd...>
  local key=$1 deps=$2 opts=()
  shift 2
  while [ "$1" != "--" ]; do opts+=("$1"); shift; done
  shift
  local dep=()
  [ -n "$deps" ] && dep=("--dependency=afterok:$deps")
  local log="$LOGS/${key}_%j.out"
  [[ " ${opts[*]} " == *" --array="* ]] && log="$LOGS/${key}_%A_%a.out"
  local cmd=(sbatch --parsable -J "${RUN}_rs_${key}" -o "$log" "${dep[@]}" "${opts[@]}" slurm/job.sh "$@")
  if [ -n "${DRY:-}" ]; then echo "${cmd[*]}" >&2; echo "<$key>"
  else "${cmd[@]}"; fi
}

echo "data $CLOUDFORGER_DATA  results $CLOUDFORGER_RESULTS  configs $CLOUDFORGER_CONFIGS"
for i in ${SEEDS:-1 2 3 4}; do
  set_name="fidelity_seed$i"
  sh=$(submit "e2e_s$i" "" $CPU_SBATCH -c 32 --mem 160G -t 12:00:00 \
    --array=0-$((SHARDS - 1)) --export=ALL,$single -- \
    $py scripts/endtoend.py --set "$set_name" --shard '{task}'/$SHARDS | tail -1)
  mg=$(submit "merge_s$i" "$sh" $CPU_SBATCH -c 4 --mem 32G -t 1:00:00 -- \
    $py scripts/endtoend.py --set "$set_name" --merge | tail -1)
  printf '%-18s shards %-10s merge %-10s (%d shards)\n' "$set_name" "$sh" "$mg" "$SHARDS"
done
echo "logs: $LOGS/"
