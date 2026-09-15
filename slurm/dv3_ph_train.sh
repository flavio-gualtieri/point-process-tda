#!/bin/bash

#SBATCH -J dv3_ph_train
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -t 04:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/dv3_ph_train_%A_%a.out
#SBATCH --error=logs/dv3_ph_train_%A_%a.err

# DV3 PERSISTENCE-FEATURE CELLS of the experiment matrix: one vectorization,
# ONE filtration (DTM k = $DTM_K, default 5) and ONE homology dimension per
# cell, 10 seeds each, scored on DV3 sets A, B and C. No fusion across k or
# dimensions, no L/F/G columns -- only the topology (+ n(x) for parameters).
#
#   vec  config (configs/runs/dv3/<group>/)  method       results subdir (under dtm_k<K>/)
#   0    ph_pi.yaml                          pi_multik    pi_multik_h<d>
#   1    ph_betti.yaml                       betti_cnn    betti_cnn_<d>
#   2    ph_landscape.yaml                   vec_multik   vec_multik_landscape_native_h<d>
#
#   tasks 0-23   PARAMETER ESTIMATION: t = vec*8 + dim*4 + family
#                family 0 thomas  1 nested  2 matern2  3 lgcp; dim 0 = H0, 1 = H1
#                (vec-major, so the persistence-image cells, tasks 0-7, start first)
#   tasks 24-29  CLASSIFICATION: t = 24 + vec*2 + dim
#
# One array job covers one k: DTM_K=10 / 15 give the other filtrations
# (results under dtm_k10/, dtm_k15/), once those diagrams exist.
# Submit through slurm/dv3_ph_submit.sh, which computes the missing train-set
# diagrams first and runs classification only after parameter estimation.
#
# PACKING. Several seeds share the GPU: PAR=5 for parameter tasks (2 waves),
# PAR=4 for classification (3 waves; ~8 GB of diagrams/images per run over
# 166k clouds). Per-seed logs: logs/dv3_ph/. Timing (sacct, DV2 single-scale
# pi_multik ~8 min/run alone; 600-epoch classify ~50 min): expect ~1 h per
# parameter task; give classification tasks --time=10:00:00 (the helper does).
#
# Resumable: train.py skips a seed whose results.pt exists; a task exits
# non-zero if any seed has no results.pt at the end, naming the seed logs.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs logs/dv3_ph

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

DTM_K="${DTM_K:-5}"
SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
FAMILY_LIST=(thomas nested matern2 lgcp)
VEC_LIST=(pi betti landscape)
N_FAM=${#FAMILY_LIST[@]}
N_PARAMS=$(( ${#VEC_LIST[@]} * 2 * N_FAM ))   # 24

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/dv3_ph_train.sh)}"
if (( t < N_PARAMS )); then
  vec="${VEC_LIST[$(( t / (2 * N_FAM) ))]}"
  dim=$(( (t % (2 * N_FAM)) / N_FAM ))
  group="${FAMILY_LIST[$(( t % N_FAM ))]}"
  process="dv3_${group}"; datadir="$group"
  PAR="${PAR:-5}"
elif (( t < N_PARAMS + 2 * ${#VEC_LIST[@]} )); then
  vec="${VEC_LIST[$(( (t - N_PARAMS) / 2 ))]}"
  dim=$(( (t - N_PARAMS) % 2 ))
  group=classify
  process=dv3_classify; datadir=_classify
  PAR="${PAR:-4}"
else
  echo "task $t out of range -- nothing to do (check --array span)"; exit 0
fi

threads=$(( ${SLURM_CPUS_PER_TASK:-$PAR} / PAR )); (( threads >= 1 )) || threads=1
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads

cfg="configs/runs/dv3/${group}/ph_${vec}.yaml"
overrides=(--set "filtration=[{name: dtm, params: {k: ${DTM_K}, q: 2.0, maxdim: 1}}]")
case "$vec" in
  pi)        subdir="pi_multik_h${dim}"
             overrides+=(--set "method.params.homology_dims=[${dim}]" --set "method.params.results_subdir=${subdir}") ;;
  betti)     subdir="betti_cnn_${dim}"
             overrides+=(--set "method.params.hom_dim=${dim}") ;;
  landscape) subdir="vec_multik_landscape_native_h${dim}"
             overrides+=(--set "method.params.homology_dims=[${dim}]" --set "method.params.results_subdir=${subdir}") ;;
esac
resdir="results/${process}/dtm_k${DTM_K}/${subdir}"

echo "Host: $(hostname)  job ${SLURM_ARRAY_JOB_ID:-unset}_${t}  ->  ${vec} H${dim} DTM k=${DTM_K}  group=${group}"
echo "  config=${cfg}"
echo "  results -> ${resdir}/seed_<seed>/   ${PAR} seeds at a time, ${threads} thread(s) each"
echo "  GPU: ${SLURM_JOB_GPUS:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Preflight: dead interpreter (2026-09-07 GPFS purge), no GPU, or diagrams
# missing for any set -- fail in seconds, not after the queue wait.
TAG_ENV="dtm_k${DTM_K}" DATADIR_ENV="$datadir" python - <<'PYEOF'
import os, sys
from pathlib import Path
import torch
from cloudforger.experiments import betti_cnn  # noqa: F401
from cloudforger.experiments.pi_multik import pi_multik, vectorized_multik  # noqa: F401

bad = [] if torch.cuda.is_available() else ["no CUDA device visible"]
for s in ("train", "A", "B", "C"):
    p = Path("data/dv3") / s / os.environ["DATADIR_ENV"] / os.environ["TAG_ENV"] / "diagrams.pkl"
    if not p.exists():
        bad.append(f"{p}: MISSING")
if bad:
    print("preflight FAILED:", *bad, sep="\n  "); sys.exit(1)
print(f"preflight OK (torch {torch.__version__}, {torch.cuda.get_device_name(0)})")
PYEOF

run_seed() {
  local seed="$1"
  local log="logs/dv3_ph/${SLURM_ARRAY_JOB_ID:-local}_${t}_${vec}_h${dim}_k${DTM_K}_${group}_s${seed}.log"
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
  echo "FAILED seeds (${#failed[@]}/${#SEEDS[@]}): ${failed[*]} -- see logs/dv3_ph/${SLURM_ARRAY_JOB_ID:-local}_${t}_*"
  exit 1
fi
echo "all ${#SEEDS[@]} seeds done: ${resdir}"
