#!/bin/bash

#SBATCH -J cascade
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --output=cascade/logs/cascade_%j.out
#SBATCH --error=cascade/logs/cascade_%j.err

# The whole cascade for one config, stage by stage. Lives in cascade/ rather than slurm/ so the
# original project's job scripts never mention it.
#
#   sbatch cascade/run.sh                                           # cascade/configs/default.yaml
#   CONFIG=cascade/configs/<name>.yaml sbatch cascade/run.sh
#   FROM=stage3 CONFIG=... sbatch cascade/run.sh                     # re-run from a stage onward
#   UNTIL=stage1 CONFIG=... sbatch cascade/run.sh                    # stop after a stage
#
# A neural stage 1 (stage1.model: nn) needs a GPU, so it is its own job; chain the rest after it:
#   jid=$(CONFIG=cascade/configs/nn.yaml sbatch --parsable cascade/stage1_nn.sh)
#   FROM=stage2 CONFIG=cascade/configs/nn.yaml sbatch --dependency=afterok:$jid cascade/run.sh
#
# Features (cascade/features.py) are a prerequisite and are skipped when already on disk. The
# gradient-boosted fits are OpenMP-parallel, so they use every CPU asked for here; the login node
# throttles a user to about one core, which is why this is a job.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cascade/logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # the module sets 1, which serializes HGB

CONFIG="${CONFIG:-cascade/configs/default.yaml}"
FROM="${FROM:-features}"
UNTIL="${UNTIL:-evaluate}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"
STAGES=(features stage1 stage2 stage3 pipeline evaluate)

started=0
for stage in "${STAGES[@]}"; do
    [[ $stage == "$FROM" ]] && started=1
    (( started )) || continue
    echo "== $stage  ($(date +%T))"
    case $stage in
        features) python cascade/features.py --workers "$WORKERS" ;;
        stage1)   python cascade/stage1.py train --config "$CONFIG" ;;
        evaluate) python cascade/evaluate.py --config "$CONFIG" --workers "$WORKERS" ;;
        *)        python "cascade/$stage.py" --config "$CONFIG" ;;
    esac
    [[ $stage == "$UNTIL" ]] && break
done
