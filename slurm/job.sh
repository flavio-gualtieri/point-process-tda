#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1

# The one SLURM worker: activate the environment and run the command it is given, from the repo
# root. `{task}` in any argument becomes the array task id. Resources, dependencies and log files
# come from the sbatch call in slurm/run_all.sh.
#
#   sbatch -c 4 --mem 16G -t 1:00:00 slurm/job.sh python -u scripts/compare.py

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/env.sh

args=()
for a in "$@"; do
  args+=("${a//\{task\}/${SLURM_ARRAY_TASK_ID:-}}")
done
echo "$(hostname)  job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID}  threads $OMP_NUM_THREADS"
echo "data ${CLOUDFORGER_DATA:-data}  results ${CLOUDFORGER_RESULTS:-results}  configs ${CLOUDFORGER_CONFIGS:-configs}"
echo "+ ${args[*]}"
exec "${args[@]}"
