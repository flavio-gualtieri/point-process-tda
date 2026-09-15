#!/bin/bash

#SBATCH -J vihrs_all
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-24
#SBATCH --output=logs/vihrs_all_%A_%a.out
#SBATCH --error=logs/vihrs_all_%A_%a.err

# ARM 3 of 3 -- vihrs baseline (L(r)-r + n(x) 1D-CNN, Vihrs 2022), 5 seeds,
# every process EXCEPT the Vihrs LGCP-family pair (lgcp, lgcp_strauss).
#
#   5 processes  (thomas, matern_cluster, nested_thomas, aniso_thomas, strauss)
# x 5 seeds      (9371..9375)
#   = 25 array tasks, one GPU per task.
#
# vihrs reads clouds.pkl directly and computes its own summary statistics
# -- no filtration / diagrams / featurize step. Each task runs that
# PROCESS'S OWN vihrs config; target set (and so head output dim +
# standardization columns) varies by process -- 3 params for thomas /
# matern_cluster / strauss, 5 for nested_thomas / aniso_thomas:
#
#   thomas          configs/runs/thomas/thomas_vihrs.yaml                   (3)  parent_intensity, mean_offspring, cluster_scale
#   matern_cluster  configs/runs/matern_cluster/matern_cluster_vihrs.yaml   (3)  parent_intensity, mean_offspring, cluster_radius
#   nested_thomas   configs/runs/nested_thomas/nested_thomas_vihrs.yaml     (5)  + meta_offspring, meta_cluster_scale
#   aniso_thomas    configs/runs/aniso_thomas/aniso_thomas_vihrs.yaml       (5)  + cluster_aspect, cluster_theta
#   strauss         configs/runs/strauss/strauss_vihrs.yaml                 (3)  beta, gamma, radius
#
# All five configs set checkpoint_best: true -> subdir "vihrs_checkpointed"
# (run_vihrs_method in scripts/train.py), so results land under
#   results/<process>/raw/vihrs_checkpointed/seed_<seed>/
# No --run-tag: this is the reference vihrs result.
#
# NOTE: nested_thomas_vihrs.yaml's process.design block predates the Sept
# cloud regen and no longer matches data/nested_thomas/clouds.pkl -- but
# the vihrs path in train.py only reads clouds.pkl + target_label_names,
# never the design block (that is generate.py's), so this is cosmetic and
# the run is correct. target_label_names there is the same 5-entry list
# nested_thomas/pi_multik.yaml uses.
#
# Resumable: is_done() skips a finished seed. Rerun specific failures with
#   sbatch --array=<comma,ids> slurm/vihrs_all.sh
#
# ASSUMES (per process): data/<process>/clouds.pkl and adversarial_clouds.pkl.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/vihrs_all.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

PROCESSES=(thomas          matern_cluster   nested_thomas   aniso_thomas    strauss)
CONFIGS=(
  configs/runs/thomas/thomas_vihrs.yaml
  configs/runs/matern_cluster/matern_cluster_vihrs.yaml
  configs/runs/nested_thomas/nested_thomas_vihrs.yaml
  configs/runs/aniso_thomas/aniso_thomas_vihrs.yaml
  configs/runs/strauss/strauss_vihrs.yaml
)
NPARAMS=(3 3 5 5 3)          # estimated-parameter count per process (doc only)
SEEDS=(9371 9372 9373 9374 9375)

NSEED=${#SEEDS[@]}                              # 5
NTASKS=$(( ${#PROCESSES[@]} * NSEED ))          # 25 -- must equal the --array span (0-24)

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/vihrs_all.sh)}"
if (( t >= NTASKS )); then
  echo "task $t >= $NTASKS -- nothing to do (check --array span)"; exit 0
fi

p=$(( t / NSEED ))
s=$(( t % NSEED ))
proc="${PROCESSES[$p]}"
cfg="${CONFIGS[$p]}"
seed="${SEEDS[$s]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  Array task ${t}  ->  process=${proc} (${NPARAMS[$p]} params)  seed=${seed}  [vihrs]"
echo "  config=${cfg}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py "$cfg" --seed "$seed"
