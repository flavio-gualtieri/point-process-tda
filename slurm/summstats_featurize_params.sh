#!/bin/bash

#SBATCH -J summ_feat_params
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=24G
#SBATCH --array=0-4
#SBATCH --output=logs/summ_feat_params_%A_%a.out
#SBATCH --error=logs/summ_feat_params_%A_%a.err

# STAGE 1 of 2 for the PARAMETER-ESTIMATION F/G experiment. CPU ONLY.
#
# Builds the F/G/J feature caches for the three parameter-estimation
# processes, mirroring slurm/summstats_featurize.sh (which did the same for
# the classification bundle). The classification path and the parameter path
# use DIFFERENT extractors and different cache files -- the parameter one also
# carries the regression `targets` matrix -- so the classification caches
# cannot be reused here.
#
#   task 0 -> thomas           (7000 + 1000 clouds)
#   task 1 -> matern_cluster   (7000 + 1000)
#   task 2 -> nested_thomas    (7000 + 1000)
#   task 3 -> lgcp             (7000 + 1000; targets mu, sigma2, s)
#   task 4 -> lgcp_strauss     (7000 + 1000; targets mu, sigma2, s, gamma, radius)
#
# Each writes, at fg_r_max = 0.25 (the grid that won the classification
# baseline sweep):
#   data/<process>/clouds.summ_fg0250_cache.npz
#   data/<process>/adversarial_clouds.summ_fg0250_cache.npz
# All four channels (L, F, G, J) go into one file, so every arm of STAGE 2 --
# the vihrs L+F+G baseline AND the pi_multik PH (+) L (+) F (+) G arms --
# shares one featurization. The historical L-only caches
# (clouds.lfunc_cache.npz) are NOT touched, so the published vihrs and
# fusion numbers keep their exact inputs.
#
# WHY A SEPARATE JOB: the cache is split-independent, so all 150 training
# tasks in STAGE 2 share it. Cold, they would each recompute it (correct --
# the write is now atomic, tmp + os.replace -- but ~150x wasted work on GPU
# nodes). One CPU task per process makes STAGE 2 start training in seconds.
#
# COST: ~8000 clouds each, F and G computed by sorting rather than by
# broadcasting an (N, m) mask, so the pass is dominated by the pre-existing
# O(n^2) L(r) computation. Expect a few minutes per task; 1 h is ample.
#
# NOTE these are the same clouds strauss/aniso_thomas would use, but neither
# is included: strauss's pi_multik training is a bimodal initialization
# lottery (see slurm/strauss_restarts.sh) and any fusion arm run there before
# that is fixed is uninterpretable, and aniso_thomas's target set still
# contains cluster_theta, which is at chance for every method.
#
# Submit from the repo root, and WAIT for it before STAGE 2:
#   fid=$(sbatch --parsable slurm/summstats_featurize_params.sh)
#   sbatch --dependency=afterok:$fid slurm/fgfusion_params.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

PROCESSES=(thomas matern_cluster nested_thomas lgcp lgcp_strauss)
CONFIGS=(
  configs/runs/thomas/thomas_vihrs.yaml
  configs/runs/matern_cluster/matern_cluster_vihrs.yaml
  configs/runs/nested_thomas/nested_thomas_vihrs.yaml
  configs/runs/lgcp/lgcp_vihrs.yaml
  configs/runs/lgcp_strauss/lgcp_strauss_vihrs.yaml
)
# Tasks 3-4 (lgcp, lgcp_strauss) were added after 0-2 had already run; for
# just the new ones:   sbatch --array=3-4 slurm/summstats_featurize_params.sh
# These need only clouds.pkl -- NO persistence diagrams -- so they can run
# immediately, in parallel with slurm/diagrams_compute_lgcp.sh. LGCP clouds
# reach 2344 points, but L is O(n^2) and F/G are O(n log n), so this stays
# cheap; the 24 GB below is ample.
FG_R_MAX=0.25

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/summstats_featurize_params.sh)}"
if (( t >= ${#PROCESSES[@]} )); then
  echo "task $t >= ${#PROCESSES[@]} -- nothing to do (check --array span)"; exit 0
fi
proc="${PROCESSES[$t]}"
cfg="${CONFIGS[$t]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  process=${proc}  fg_r_max=${FG_R_MAX}"
echo "  config=${cfg}"

# One seed, one epoch: prepare_data runs before any training, so the cache is
# written on the way in and the throwaway model costs essentially nothing.
# --run-tag keeps it out of the results tree STAGE 2 writes.
python -u scripts/train.py "$cfg" --seed 9371 --force \
    --run-tag "summ__cachewarm_params" \
    --set "method.params.summary_channels=[L,F,G]" \
    --set "method.params.fg_r_max=${FG_R_MAX}" \
    --set "method.params.n_epochs=1" \
    --set "method.params.skip_mincontrast=true"

echo
echo "cache files now present for ${proc}:"
ls -la "data/${proc}"/*summ_fg*_cache.npz
