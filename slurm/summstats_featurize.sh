#!/bin/bash

#SBATCH -J summ_featurize
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-1
#SBATCH --output=logs/summ_featurize_%A_%a.out
#SBATCH --error=logs/summ_featurize_%A_%a.err

# STAGE 1 of 2 -- build the F/G/J feature cache. CPU ONLY, no GPU requested.
#
# WHY A SEPARATE JOB. The feature cache is split-independent (whole file, file
# order), so all 60 training tasks in STAGE 2 share it. If they were launched
# cold, all 60 would recompute the same 40k-cloud featurization in parallel:
# correct (the cache write is atomic -- tmp file + os.replace -- so they
# cannot tear each other's file) but ~60x wasted work, and on GPU nodes.
# Building it once here on a CPU node costs one task and makes STAGE 2 tasks
# start training in seconds.
#
#   task 0 -> fg_r_max = 0.25   (the literal same radius grid as Ripley's L)
#   task 1 -> fg_r_max = 0.08   (F and G saturate long before 0.25; see below)
#
# Each task builds BOTH caches for its fg_r_max (35k train/test + 5k
# adversarial clouds):
#   data/classification/clouds.summ_fg<NNNN>_classify_cache.npz
#   data/classification/adversarial_clouds.summ_fg<NNNN>_classify_cache.npz
# where NNNN = round(fg_r_max * 1000). All four channels (L, F, G, J) go into
# one file, so every channel-subset arm of STAGE 2 shares one featurization.
# The historical L-only cache (clouds.lfunc_classify_cache.npz) is NOT
# touched, so the published vihrs numbers keep their exact inputs.
#
# WHY TWO fg_r_max VALUES. F and G are CDFs that reach 1 far below Ripley's
# geometric r_max = 0.25 -- mean nearest-neighbour distance is ~0.018-0.041
# at these intensities -- so on the shared 0.25 grid most of their 513
# samples sit saturated at 1 and carry no gradient. fg_r_max puts F/G/J on
# their own grid of the SAME length, so shortening it raises their resolution
# without changing the CNN's input shape. 0.25 is the zero-tuning mirror;
# 0.08 is the informative range. Running both and picking on VALIDATION
# accuracy is what makes this a strong baseline rather than a strawman.
#
# COST. F and G are computed by sorting, not by broadcasting an (N, m) mask
# (see summstats._reduced_sample_cdf), so each is O(N log N + m log N) and
# the four-channel pass is dominated by the pre-existing O(n^2) L(r)
# computation -- expect roughly 2-4x the ~2 min the L-only cache took, i.e.
# well under 15 min per task. 32 GB and 1 h are generous.
#
# Trigger a rebuild (e.g. after changing an estimator) by deleting the .npz
# files, or by adding --set method.params.recompute_features=true below.
#
# ASSUMES:
#   data/classification/clouds.pkl, adversarial_clouds.pkl
#   data/classification/dtm_k{5,10,15}/{,adversarial_}diagrams.pkl  (seeds/labels only)
#
# Submit from the point-process-tda repo root, and WAIT for it before STAGE 2:
#   sbatch slurm/summstats_featurize.sh
#   sbatch --dependency=afterok:<jobid> slurm/summstats_classify.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CFG=configs/runs/classification/summstats_kfgj.yaml
FG_R_MAX=(0.25 0.08)

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/summstats_featurize.sh)}"
if (( t >= ${#FG_R_MAX[@]} )); then
  echo "task $t >= ${#FG_R_MAX[@]} -- nothing to do (check --array span)"; exit 0
fi
fg="${FG_R_MAX[$t]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  building F/G/J cache at fg_r_max=${fg}"

# One seed is enough: prepare_data_classify runs before any training, so the
# cache is written on the way in. --run-tag keeps this throwaway model out of
# the results tree that STAGE 2 writes; the cache is the actual product.
# n_epochs=1 so we pay the featurization and essentially nothing else.
python -u scripts/train.py "$CFG" --seed 9371 --force \
    --run-tag "summ__cachewarm_fg${fg}" \
    --set "method.params.summary_channels=[L,F,G,J]" \
    --set "method.params.fg_r_max=${fg}" \
    --set "method.params.n_epochs=1"

echo
echo "cache files now present for fg_r_max=${fg}:"
ls -la data/classification/*summ_fg*_classify_cache.npz
