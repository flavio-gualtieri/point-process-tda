#!/bin/bash

#SBATCH -J pi_multik_h0h1_all
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 02:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-24
#SBATCH --output=logs/pi_multik_h0h1_all_%A_%a.out
#SBATCH --error=logs/pi_multik_h0h1_all_%A_%a.err

# ARM 2 of 3 -- [PI | H0+H1 | DTM{5,10,15} | 5 seeds], every process
# EXCEPT the Vihrs LGCP-family pair (lgcp, lgcp_strauss).
#
#   5 processes  (thomas, matern_cluster, nested_thomas, aniso_thomas, strauss)
# x 5 seeds      (9371..9375, first 5 of each config's 10-seed list)
#   = 25 array tasks, one GPU per task.
#
# This is the reference pi_multik run: NO --run-tag, so it owns the default
#   results/<process>/dtm_k5+10+15/pi_multik/seed_<seed>/
# path that ARM 1 (pi_multik_h0_all.sh, --run-tag h0only) and evaluate.py
# compare against. Each task runs that PROCESS'S OWN config -- filename and
# target set both vary by process:
#
#   process         config                                                   #params  targets
#   thomas          thomas/thomas_pi_multik_k5k10k15.yaml                       3      parent_intensity, mean_offspring, cluster_scale
#   matern_cluster  matern_cluster/matern_cluster_pi_multik_k5k10k15.yaml       3      parent_intensity, mean_offspring, cluster_radius
#   nested_thomas   nested_thomas/pi_multik.yaml                                5      + meta_offspring, meta_cluster_scale
#   aniso_thomas    aniso_thomas/aniso_thomas_pi_multik_k5k10k15.yaml           5      + cluster_aspect, cluster_theta
#   strauss         strauss/strauss_pi_multik_k5k10k15.yaml                     3      beta, gamma, radius
#
# The head output dim (and standardization column count) follows each
# config's target_label_names, so the 3- vs 5-parameter split is handled
# by pointing every task at the right config -- never a shared template.
# Resources are sized for the worst case (5-param head, 6 PI channels =
# H0+H1 x 3 k, heaviest clouds = nested_thomas): 88 GB, 2 h is generous.
#
# Resumable: is_done() skips a seed whose results.pt already exists, so a
# plain re-submit only fills gaps. Rerun specific failures with
#   sbatch --array=<comma,ids> slurm/pi_multik_h0h1_all.sh
#
# ASSUMES, for every process above:
#   data/<process>/clouds.pkl, adversarial_clouds.pkl
#   data/<process>/dtm_k{5,10,15}/{,adversarial_}diagrams.pkl
# (clouds via slurm/generate_*.sh; diagrams via slurm/diagrams_compute*.sh
# -> slurm/diagrams_merge*.sh). pi_multik reads the per-k diagrams.pkl
# directly and calibrates persistence images per seed on that seed's train
# split -- no scripts/featurize.py step needed.
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/pi_multik_h0h1_all.sh

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
  configs/runs/thomas/thomas_pi_multik_k5k10k15.yaml
  configs/runs/matern_cluster/matern_cluster_pi_multik_k5k10k15.yaml
  configs/runs/nested_thomas/pi_multik.yaml
  configs/runs/aniso_thomas/aniso_thomas_pi_multik_k5k10k15.yaml
  configs/runs/strauss/strauss_pi_multik_k5k10k15.yaml
)
NPARAMS=(3 3 5 5 3)          # estimated-parameter count per process (doc only)
SEEDS=(9371 9372 9373 9374 9375)

NSEED=${#SEEDS[@]}                              # 5
NTASKS=$(( ${#PROCESSES[@]} * NSEED ))          # 25 -- must equal the --array span (0-24)

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/pi_multik_h0h1_all.sh)}"
if (( t >= NTASKS )); then
  echo "task $t >= $NTASKS -- nothing to do (check --array span)"; exit 0
fi

p=$(( t / NSEED ))
s=$(( t % NSEED ))
proc="${PROCESSES[$p]}"
cfg="${CONFIGS[$p]}"
seed="${SEEDS[$s]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  Array task ${t}  ->  process=${proc} (${NPARAMS[$p]} params)  seed=${seed}"
echo "  config=${cfg}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train.py "$cfg" --seed "$seed"
