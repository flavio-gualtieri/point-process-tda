#!/bin/bash
# The whole paper run as one chain of SLURM jobs, each waiting on the jobs whose outputs it reads.
#
#   CLOUDFORGER_DATA=data_v2 CLOUDFORGER_RESULTS=results_v2 bash slurm/run_all.sh    # everything
#   SMOKE=1 bash slurm/run_all.sh                     # configs/smoke on a strided bank -> data_smoke/, results_smoke/
#   FROM=train ... bash slurm/run_all.sh              # skip the stages before `train` (their outputs exist)
#   ONLY="compare endtoend" ... bash slurm/run_all.sh # just these stages
#   DRY=1 ... bash slurm/run_all.sh                   # print the sbatch calls, submit nothing
#
# Runs (RUNS, default "pipeline ablation") are configs/<run>.yaml: the pipeline, and the single-model
# feature study. Only train and compare are per run; everything else is shared or the pipeline's.
#
# Every stage skips outputs that already exist, so re-running this after a failure submits the
# same chain and only the missing work is done. Data and results roots must be named explicitly
# (or SMOKE=1), so a run can never write into an earlier one by accident.
#
# Stages (-> waits on):
#   departure    CSR null tables; only when asked (ONLY / FROM), since the fitted tables ship in configs/
#   simulate     bank shards, one array task per (family, shard)
#   merge        -> simulate                   shards -> <data>/bank/<family>/
#   relabel      -> merge                      delta-tilde into every manifest
#   featurize    -> merge                      diagrams, one array task per (family, filtration)
#   diagrams     -> featurize                  merge the diagram shards
#   curves       -> merge                      L, F, G, J curves (network inputs)
#   tables       -> diagrams, curves           classical and PH tables, then check every model input
#   train        -> tables                     every untrained unit of each run: a CPU and a GPU array
#   compare      -> train                      per run; the ablation's also waits on the pipeline's cutoffs
#   mincontrast  -> merge, compare             clouds -> fit array -> assemble
#   endtoend     -> compare, mincontrast       evaluation sets main, ph_ablation, ph_cell, mincontrast
#   power        -> relabel                    power check of the scores (configs/scores.yaml)

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

STAGES=(departure simulate merge relabel featurize diagrams curves tables train compare mincontrast endtoend power)
RUNS=(${RUNS:-pipeline ablation})

if [ -n "${SMOKE:-}" ]; then
  export CLOUDFORGER_CONFIGS="$ROOT/configs/smoke"
  export CLOUDFORGER_DATA="${CLOUDFORGER_DATA:-$ROOT/data_smoke}"
  export CLOUDFORGER_RESULTS="${CLOUDFORGER_RESULTS:-$ROOT/results_smoke}"
fi
: "${CLOUDFORGER_DATA:?set CLOUDFORGER_DATA and CLOUDFORGER_RESULTS (a fresh root, e.g. data_v2 / results_v2) or SMOKE=1}"
: "${CLOUDFORGER_RESULTS:?set CLOUDFORGER_DATA and CLOUDFORGER_RESULTS (a fresh root, e.g. data_v2 / results_v2) or SMOKE=1}"
export CLOUDFORGER_DATA="$(realpath -m "$CLOUDFORGER_DATA")" CLOUDFORGER_RESULTS="$(realpath -m "$CLOUDFORGER_RESULTS")"

# Which stages run: ONLY, else FROM onwards, else everything but departure.
if [ -n "${ONLY:-}" ]; then
  SELECTED=($ONLY)
elif [ -n "${FROM:-}" ]; then
  SELECTED=(); on=
  for s in "${STAGES[@]}"; do [ "$s" = "$FROM" ] && on=1; [ -n "$on" ] && SELECTED+=("$s"); done
  [ ${#SELECTED[@]} -gt 0 ] || { echo "unknown stage FROM=$FROM" >&2; exit 2; }
else
  SELECTED=("${STAGES[@]:1}")
fi
selected() { [[ " ${SELECTED[*]} " == *" $1 "* ]]; }

source slurm/env.sh                                          # for the python call below
eval "$(python slurm/plan.py "${RUNS[@]}")"                  # array sizes: simulate featurize mincontrast train_*
RUN=$(basename "$CLOUDFORGER_RESULTS")
LOGS="logs/$RUN"
mkdir -p "$LOGS"
declare -A JOB

# submit <key> "<keys it waits on>" <sbatch options...> -- <command...>
submit() {
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
  if [ -n "${DRY:-}" ]; then
    JOB[$key]="<$key>"
    echo "${cmd[*]}"
  else
    JOB[$key]=$("${cmd[@]}")
    printf '%-14s %s\n' "$key" "${JOB[$key]}"
  fi
}
py="python -u"
single='THREADS=1'                                           # jobs that parallelize over processes

echo "data $CLOUDFORGER_DATA  results $CLOUDFORGER_RESULTS  configs ${CLOUDFORGER_CONFIGS:-configs}"
echo "stages: ${SELECTED[*]}"

if selected departure; then
  submit departure "" $CPU_SBATCH -c 16 --mem 32G -t 6:00:00 -- \
    bash -c "$py scripts/departure.py simulate --jobs 16 && $py scripts/departure.py fit && $py scripts/departure.py validate"
fi
if selected simulate; then
  submit simulate "departure" $SHORT_SBATCH -c 1 --mem 8G -t 1:00:00 --array=0-$((simulate - 1))%200 -- \
    $py scripts/simulate.py run --task '{task}'
fi
if selected merge; then
  submit merge "simulate" $SHORT_SBATCH -c 1 --mem 32G -t 1:00:00 -- $py scripts/simulate.py merge
fi
if selected relabel; then
  submit relabel "merge" $CPU_SBATCH -c 1 --mem 16G -t 4:00:00 -- $py scripts/relabel.py
fi
if selected featurize; then
  # A (family, filtration) task is every shard of it: about an hour at the paper bank's size, and a task
  # that times out blocks the chain, so a larger bank names more: FEATURIZE_SBATCH="-p compute" FEATURIZE_TIME=6:00:00
  submit featurize "merge" ${FEATURIZE_SBATCH:-$SHORT_SBATCH} -c 8 --mem 24G -t ${FEATURIZE_TIME:-1:00:00} --array=0-$((featurize - 1)) --export=ALL,$single -- \
    $py scripts/featurize.py run --task '{task}' --jobs 8
fi
if selected diagrams; then
  submit diagrams "featurize" $SHORT_SBATCH -c 1 --mem 32G -t 1:00:00 -- $py scripts/featurize.py merge
fi
if selected curves; then
  submit curves "merge" $CPU_SBATCH -c 1 --mem 16G -t 6:00:00 -- $py scripts/classical.py   # ~4h for 9 families
fi
if selected tables; then
  submit tables "diagrams curves" $CPU_SBATCH -c 16 --mem 64G -t 6:00:00 --export=ALL,$single -- \
    bash -c "$py scripts/tables.py classical && $py scripts/tables.py ph && $py scripts/tables.py check"
fi
if selected train; then
  for run in "${RUNS[@]}"; do
    for kind in cpu gpu; do
      var=train_${kind}_$run; idx=${!var}                    # unit indices without a report.json
      [ -n "$idx" ] || { echo "train_${kind}_$run: nothing to train"; continue; }
      if [ $kind = cpu ]; then res=($CPU_SBATCH -c 16 --mem 64G -t 12:00:00); else res=($GPU_SBATCH -c 8 --mem 192G -t 24:00:00); fi
      submit train_${kind}_$run "tables" "${res[@]}" --array="$idx" -- \
        $py scripts/train.py --config $run.yaml unit --kind $kind --index '{task}'
    done
  done
fi
if selected compare; then                                    # key `compare` = the pipeline's (cutoffs, `best`)
  for run in "${RUNS[@]}"; do
    key=compare; deps="train_cpu_$run train_gpu_$run"
    [ $run = pipeline ] || { key=compare_$run; deps="$deps compare"; }
    submit $key "$deps" $CPU_SBATCH -c 4 --mem 48G -t 2:00:00 -- $py scripts/compare.py --config $run.yaml
  done
fi
if selected mincontrast; then
  submit mc_clouds "merge" $CPU_SBATCH -c 16 --mem 64G -t 4:00:00 --export=ALL,$single -- $py scripts/mincontrast.py clouds
  submit mc_fit "mc_clouds" $CPU_SBATCH -c 16 --mem 16G -t 4:00:00 --array=0-$((mincontrast - 1)) --export=ALL,$single -- \
    $py scripts/mincontrast.py fit --task '{task}'
  submit mincontrast "mc_fit compare" $CPU_SBATCH -c 4 --mem 32G -t 2:00:00 -- $py scripts/mincontrast.py assemble
fi
if selected endtoend; then
  for set in main ph_ablation ph_cell mincontrast; do
    deps=compare; [ $set = mincontrast ] && deps="compare mincontrast"
    submit e2e_$set "$deps" $CPU_SBATCH -c 32 --mem 160G -t 12:00:00 --export=ALL,$single -- \
      $py scripts/endtoend.py --set $set
  done
fi
if selected power; then
  submit power "relabel" $CPU_SBATCH -c 16 --mem 160G -t 6:00:00 --export=ALL,$single -- $py scripts/power.py
fi
echo "logs: $LOGS/"
