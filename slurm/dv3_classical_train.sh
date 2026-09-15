#!/bin/bash

#SBATCH -J dv3_cl_train
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH --array=0-34
#SBATCH --output=logs/dv3_cl_train_%A_%a.out
#SBATCH --error=logs/dv3_cl_train_%A_%a.err

# STAGE 2 of 2 -- DV3 classical baselines: the vihrs 1-D CNN (Vihrs 2022) on
# the standard summary functions, for parameter estimation (4 families) and
# 5-way classification, 10 seeds each, scored on DV3 sets A, B and C.
# Needs NO persistence diagrams -- runnable while the DTM diagrams compute.
#
#   arm  tag        channels + n(x)  F/G/J grid   results subdir
#   0    L          L(r)-r           --           vihrs_checkpointed (params) / vihrs (classify)
#   1    LFGJ       L, F, G, J       0.25         vihrs_lfgj      the union baseline to beat
#   2    LFGJ_r080  L, F, G, J       0.08         vihrs_lfgj_r080
#   3    G          G                0.25         vihrs_g
#   4    F          F                0.25         vihrs_f
#   5    G_r080     G                0.08         vihrs_g_r080
#   6    F_r080     F                0.08         vihrs_f_r080
#
#   tasks 0-27   PARAMETER ESTIMATION (inference): t = arm * 4 + family,
#                family 0 thomas  1 nested  2 matern2  3 lgcp
#                (arm-major, so L and L+F+G+J -- tasks 0-7 -- start first)
#   tasks 28-34  CLASSIFICATION: t = 28 + arm
#
# Inference goes first. slurm/dv3_classical_submit.sh submits 0-27 as soon as
# the params caches exist and 28-34 only after every inference task has ended,
# so classification never takes one of the 12 GPUs inference could use.
#
# Configs: configs/runs/dv3/<group>/vihrs.yaml (arm L) and
# configs/runs/dv3/<group>/summstats_lfgj.yaml (every other arm, with the
# channels / grid / results subdir overridden below). Same network, epoch cap
# and early stopping for every arm; only the input channels differ. Arms 1-2,
# 3-5 and 4-6 differ only in the F/G/J radius grid: F and G saturate far
# below r = 0.25 at DV3 intensities, so 0.08 is the informative range and
# 0.25 the zero-tuning mirror of L's grid. Report the one that wins on
# VALIDATION (scripts/dv3_matrix.py marks it), never on A/B/C.
#
# Output: results/dv3_<group>/raw/<subdir>/seed_<seed>/ with results.json
# (eval_sets block) and predictions_{A,B,C}.npz for scripts/evaluate_regimes.py.
# Then refresh the matrix sheet:  python scripts/dv3_matrix.py
#
# PACKING. The CNN is tiny (a 1-D conv stack on 1-4 x 513 inputs), so one run
# leaves an A100 mostly idle, and the sae QOS caps a user at 12 GPUs. Each
# task therefore runs several seeds at once on its one GPU: an inference task
# all 10 (one wave, 1 CPU thread each; a family's train+eval features are
# ~0.3 GB), a classification task 5 at a time (2 waves, 2 threads; ~5 GB of
# features and clouds per run). Override with PAR=<n> in the environment.
# Per-seed logs: logs/dv3_classical/.
# Timing (sacct, DV2 analogues): vihrs params 4-9 min/run alone, vihrs
# classify 1-5 min/run; DV3 has ~1.4x the training rows and also scores
# A/B/C, and packed runs share a GPU, so expect ~20-40 min per task.
# 2 h leaves room for that; a re-submit only re-runs unfinished seeds.
#
# REQUIRES STAGE 1 (slurm/dv3_classical_prep.sh): the preflight hard-fails if
# a needed feature cache is missing rather than letting 10 runs rebuild it.
# Submit the whole chain with:  bash slurm/dv3_classical_submit.sh
#
# Resumable: train.py skips a seed whose results.pt exists, so a plain
# re-submit (or --array=<ids>) only fills gaps. A task exits non-zero if any
# of its seeds has no results.pt at the end, naming the seed logs to read.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs logs/dv3_classical

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
FAMILY_LIST=(thomas nested matern2 lgcp)

# Keep in step with NEURAL_ARMS in scripts/dv3_matrix.py.
ARM_TAG=(L    LFGJ        LFGJ_r080       G        F        G_r080        F_r080)
ARM_CH=(""    "[L,F,G,J]" "[L,F,G,J]"     "[G]"    "[F]"    "[G]"         "[F]")
ARM_FG=(""    0.25        0.08            0.25     0.25     0.08          0.08)
ARM_SUB=(""   vihrs_lfgj  vihrs_lfgj_r080 vihrs_g  vihrs_f  vihrs_g_r080  vihrs_f_r080)
N_ARMS=${#ARM_TAG[@]}
N_FAM=${#FAMILY_LIST[@]}
N_PARAMS=$(( N_ARMS * N_FAM ))   # 28

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/dv3_classical_train.sh)}"
if (( t < N_PARAMS )); then
  arm=$(( t / N_FAM ))
  group="${FAMILY_LIST[$(( t % N_FAM ))]}"
  process="dv3_${group}"; datadir="$group"; cache_suffix=_cache.npz
  PAR="${PAR:-10}"
elif (( t < N_PARAMS + N_ARMS )); then
  arm=$(( t - N_PARAMS ))
  group=classify
  process=dv3_classify; datadir=_classify; cache_suffix=_classify_cache.npz
  PAR="${PAR:-5}"
else
  echo "task $t >= $((N_PARAMS + N_ARMS)) -- nothing to do (check --array span)"; exit 0
fi
tag="${ARM_TAG[$arm]}"

threads=$(( ${SLURM_CPUS_PER_TASK:-$PAR} / PAR )); (( threads >= 1 )) || threads=1
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads

if [[ "$tag" == "L" ]]; then
  cfg="configs/runs/dv3/${group}/vihrs.yaml"
  if [[ "$group" == "classify" ]]; then subdir=vihrs; else subdir=vihrs_checkpointed; fi
  cache="clouds.lfunc${cache_suffix}"
  overrides=(--set "method.params.results_subdir=${subdir}")
else
  cfg="configs/runs/dv3/${group}/summstats_lfgj.yaml"
  subdir="${ARM_SUB[$arm]}"
  fg="${ARM_FG[$arm]}"
  cache="clouds.summ_fg$(printf '%04d' "$(awk "BEGIN{print int(${fg} * 1000 + 0.5)}")")${cache_suffix}"
  overrides=(--set "method.params.summary_channels=${ARM_CH[$arm]}"
             --set "method.params.fg_r_max=${fg}"
             --set "method.params.results_subdir=${subdir}")
fi
resdir="results/${process}/raw/${subdir}"

echo "Host: $(hostname)  job ${SLURM_ARRAY_JOB_ID:-unset}_${t}  ->  arm=${tag}  group=${group}"
echo "  config=${cfg}  ${overrides[*]}"
echo "  results -> ${resdir}/seed_<seed>/   cache=${cache}   ${PAR} seeds at a time, ${threads} thread(s) each"
echo "  GPU: ${SLURM_JOB_GPUS:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Preflight: a dead interpreter (the 2026-09-07 GPFS purge reaped cloud-env's
# stdlib and killed 76 array tasks in <1 s), no GPU, or a missing stage-1 cache.
CACHE_ENV="$cache" DATADIR_ENV="$datadir" python - <<'PYEOF'
import os, sys
from pathlib import Path
import torch
from cloudforger.baselines import summstats, vihrs  # noqa: F401

bad = [] if torch.cuda.is_available() else ["no CUDA device visible"]
for s in ("train", "A", "B", "C"):
    p = Path("data/dv3") / s / os.environ["DATADIR_ENV"] / os.environ["CACHE_ENV"]
    if not p.exists():
        bad.append(f"{p}: MISSING -- run slurm/dv3_classical_prep.sh first")
if bad:
    print("preflight FAILED:", *bad, sep="\n  "); sys.exit(1)
print(f"preflight OK (torch {torch.__version__}, {torch.cuda.get_device_name(0)})")
PYEOF

run_seed() {
  local seed="$1"
  local log="logs/dv3_classical/${SLURM_ARRAY_JOB_ID:-local}_${t}_${tag}_${group}_s${seed}.log"
  echo "  [$(date +%H:%M:%S)] seed ${seed} start -> ${log}"
  python -u scripts/train.py "$cfg" --seed "$seed" "${overrides[@]}" > "$log" 2>&1 || true
  echo "  [$(date +%H:%M:%S)] seed ${seed} end"
}

for seed in "${SEEDS[@]}"; do
  if [[ -f "${resdir}/seed_${seed}/results.pt" ]]; then
    echo "  seed ${seed}: done already, skipping"; continue
  fi
  while (( $(jobs -rp | wc -l) >= PAR )); do wait -n || true; done
  run_seed "$seed" &
  sleep 3   # stagger interpreter start-up / CUDA context creation
done
wait || true

# train.py catches a failed seed and exits 0, so judge by the product.
failed=()
for seed in "${SEEDS[@]}"; do
  [[ -f "${resdir}/seed_${seed}/results.pt" ]] || failed+=("$seed")
done
if (( ${#failed[@]} )); then
  echo "FAILED seeds (${#failed[@]}/${#SEEDS[@]}): ${failed[*]} -- see logs/dv3_classical/${SLURM_ARRAY_JOB_ID:-local}_${t}_*"
  exit 1
fi
echo "all ${#SEEDS[@]} seeds done: ${resdir}"
