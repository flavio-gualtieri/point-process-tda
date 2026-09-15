#!/bin/bash

#SBATCH -J vihrs_topup
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 4
#SBATCH --cpus-per-gpu=4
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-14
#SBATCH --output=logs/vihrs_topup_%A_%a.out
#SBATCH --error=logs/vihrs_topup_%A_%a.err

# Extends the vihrs baseline from 5 seeds to 10, so the fusion sweep
# (slurm/fusion_*.sh, 5 arms x seeds 9371..9380) can be compared to it
# PAIRED rather than as two unpaired means.
#
# The thomas / nested_thomas vihrs configs already declare seeds 9371..9380 --
# only 9371..9375 were ever run. The classification config declares 5; the
# extra seeds are passed on the CLI, which overrides the config's seed list.
# Split parity is automatic: every method derives its split from
# train_val_test_indices(n, seed) on the same intersected seed set.
#
#   3 processes x 5 seeds (9376..9380) -> 15 array tasks
#   task t -> process = t / 5, seed = SEEDS[t % 5]
#
# No --run-tag: these land in the same default directories as the existing
# vihrs results, extending those arms in place rather than forking them.
#   results/{thomas,nested_thomas}/raw/vihrs_checkpointed/seed_<seed>/
#   results/classification/raw/vihrs/seed_<seed>/
# Resumable -- is_done() skips a seed that already has results.pt.
#
# Submit from the repo root:  sbatch slurm/vihrs_topup.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}

PROCS=(thomas nested_thomas classification)
CFGS=(configs/runs/thomas/thomas_vihrs.yaml
      configs/runs/nested_thomas/nested_thomas_vihrs.yaml
      configs/runs/classification/vihrs_lr_nx_k5k10k15.yaml)
N_PROCS=${#PROCS[@]}

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/vihrs_topup.sh)}"
if (( t >= N_PROCS * N_SEEDS )); then
  echo "task $t >= $((N_PROCS * N_SEEDS)) -- nothing to do (check --array span)"; exit 0
fi
pi=$(( t / N_SEEDS ))
seed="${SEEDS[$(( t % N_SEEDS ))]}"
proc="${PROCS[$pi]}"
cfg="${CFGS[$pi]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  vihrs  process=${proc}  seed=${seed}"
echo "  config=${cfg}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py "$cfg" --seed "$seed"
