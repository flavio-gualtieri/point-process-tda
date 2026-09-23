#!/bin/bash

# Submit the whole experiment matrix as one sbatch array per task, via slurm/train.sh.
#
#   bash slurm/matrix.sh            # print the sbatch commands, submit nothing
#   bash slurm/matrix.sh submit     # submit them
#
# One array per task rather than one array for everything, for two reasons:
#
#   MaxArraySize. The default matrix is 200 array tasks per task and six tasks, so a single array
#   would be 1200 -- over the 1001 most Slurm builds ship with, and the submission fails outright.
#
#   Memory. Classification holds all five families (500k patterns) and parameter estimation holds
#   one (100k), a 5x difference, and --mem is a property of the array, not of its tasks. Sizing one
#   array for the worst case would reserve 64G for 1000 tasks that need 16G and throttle the queue
#   for no reason. See the memory table in slurm/train.sh for where the figures come from.
#
# Every axis of slurm/train.sh passes straight through, so a narrowed matrix is a narrowed submit:
#
#   VARIANTS="perslay" CURVES="" bash slurm/matrix.sh submit    # just the PersLay arm
#   SEEDS="1 2 3" bash slurm/matrix.sh submit                   # a three-seed pilot
#
# Re-running after a split change needs no special handling: scripts/train.py retrains any seed
# whose run.json records a different cloudforger.simulation.split and skips the rest, so this is
# also the resume command.

set -euo pipefail

cd "$(dirname "$0")/.."

TASKS="${TASKS:-classify params:poisson params:thomas params:nested params:matern2 params:lgcp}"
THROTTLE="${THROTTLE:-20}"          # tasks of one array running at once
TIME="${TIME:-24:00:00}"
CLASSIFY_MEM="${CLASSIFY_MEM:-64G}"
PARAMS_MEM="${PARAMS_MEM:-16G}"

action="${1:-print}"
if [ "$action" != "print" ] && [ "$action" != "submit" ]; then
  echo "usage: bash slurm/matrix.sh [print|submit]" >&2
  exit 2
fi

total=0
for entry in $TASKS; do
  n=$(TASKS="$entry" bash slurm/train.sh | tail -1 | awk '{print $1}')
  [ "$n" -gt 0 ] || { echo "$entry: no runs, skipping" >&2; continue; }
  total=$(( total + n ))
  case "$entry" in
    classify) mem="$CLASSIFY_MEM" ;;
    *)        mem="$PARAMS_MEM" ;;
  esac
  cmd=(sbatch --job-name "train_${entry//:/_}" --mem "$mem" -t "$TIME"
       --array "0-$(( n - 1 ))%${THROTTLE}"
       --export "ALL,TASKS=$entry" slurm/train.sh)
  if [ "$action" = "submit" ]; then
    "${cmd[@]}"
  else
    printf '%s\n' "${cmd[*]}"
  fi
done

echo "# $total array tasks across $(echo $TASKS | wc -w | tr -d ' ') arrays" \
     "x $(bash -c 'echo ${SEED_CHUNK:-2}') seed(s) each"
[ "$action" = "print" ] && echo "# nothing submitted -- rerun with: bash slurm/matrix.sh submit"
