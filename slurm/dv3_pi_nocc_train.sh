#!/bin/bash

#SBATCH -J dv3_pi_nocc
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -t 01:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/dv3_pi_nocc_%A_%a.out
#SBATCH --error=logs/dv3_pi_nocc_%A_%a.err

# PERSISTENCE-IMAGE cells WITHOUT CoordConv, with and without input
# normalization -- the fix for the collapsed H1 runs (pi_multik_h1: H1 pixels
# ~10x below H0's, swamped by the two constant coordinate channels, so the
# network ignored its input). One DTM filtration (k = $DTM_K, default 5),
# 10 seeds per cell, scored on DV3 sets A, B, C.
#
#   cell  dim  coordconv  pi_normalize  results subdir (results/<process>/dtm_k<K>/)
#   0     H0   off        channel       pi_multik_h0_nocc_norm
#   1     H1   off        channel       pi_multik_h1_nocc_norm
#   2     H0   off        none          pi_multik_h0_nocc   (isolates CoordConv vs the
#                                                          existing pi_multik_h0 run)
#   3     H0+H1 off       channel       pi_multik_h01_nocc_norm  (both dimensions as two
#                                                          channels of one image, each
#                                                          normalized separately)
#   pi_normalize=channel: per-(k, dim) pixel z-score fit on the TRAIN rows only
#   and applied frozen to validation and A/B/C (pi_multik.fit_pi_norm).
#
#   tasks 0-15   PARAMETER ESTIMATION: t = cell*4 + family
#                family 0 thomas  1 nested  2 matern2  3 lgcp
#   tasks 16-19  CLASSIFICATION: t = 16 + cell
#   (cells 0-2 were first submitted under the 3-cell layout, classification then
#   at 12-14; their results are on disk and are skipped on a re-submit)
#
# Everything else matches configs/runs/dv3/<group>/ph_pi.yaml (same network,
# epochs, early stopping; n(x) on for parameters, off for classification).
# Needs the dtm_k<K> diagrams for train/A/B/C (and the _classify bundle),
# which exist for k=5.
#
# Submit through slurm/dv3_pi_nocc_submit.sh (parameters first, then
# classification with --time=03:00:00). Timing from the matching pi_multik_h0
# jobs: parameter tasks 7-15 min with all 10 seeds on one GPU; classification
# 56 min with 5 at a time, 24 GB peak RSS.
#
# Resumable: train.py skips a seed whose results.pt exists; a task exits
# non-zero if any seed has no results.pt at the end, naming the seed logs.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs logs/dv3_pi_nocc

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

DTM_K="${DTM_K:-5}"
SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
FAMILY_LIST=(thomas nested matern2 lgcp)

CELL_TAG=(h0_nocc_norm  h1_nocc_norm  h0_nocc  h01_nocc_norm)
CELL_DIMS=("0"          "1"           "0"      "0,1")
CELL_NORM=(channel      channel       none     channel)
N_CELLS=${#CELL_TAG[@]}
N_FAM=${#FAMILY_LIST[@]}
N_PARAMS=$(( N_CELLS * N_FAM ))   # 16

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/dv3_pi_nocc_train.sh)}"
if (( t < N_PARAMS )); then
  cell=$(( t / N_FAM ))
  group="${FAMILY_LIST[$(( t % N_FAM ))]}"
  process="dv3_${group}"; datadir="$group"
  PAR="${PAR:-10}"
elif (( t < N_PARAMS + N_CELLS )); then
  cell=$(( t - N_PARAMS ))
  group=classify
  process=dv3_classify; datadir=_classify
  PAR="${PAR:-5}"
else
  echo "task $t out of range -- nothing to do (check --array span)"; exit 0
fi
dims="${CELL_DIMS[$cell]}"
subdir="pi_multik_${CELL_TAG[$cell]}"

threads=$(( ${SLURM_CPUS_PER_TASK:-$PAR} / PAR )); (( threads >= 1 )) || threads=1
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads

cfg="configs/runs/dv3/${group}/ph_pi.yaml"
overrides=(
  --set "filtration=[{name: dtm, params: {k: ${DTM_K}, q: 2.0, maxdim: 1}}]"
  --set "method.params.homology_dims=[${dims}]"
  --set "method.params.coordconv=false"
  --set "method.params.pi_normalize=${CELL_NORM[$cell]}"
  --set "method.params.results_subdir=${subdir}"
)
resdir="results/${process}/dtm_k${DTM_K}/${subdir}"

echo "Host: $(hostname)  job ${SLURM_ARRAY_JOB_ID:-unset}_${t}  ->  ${subdir} (H[${dims}], coordconv off, pi_normalize=${CELL_NORM[$cell]})  DTM k=${DTM_K}  group=${group}"
echo "  config=${cfg}"
echo "  results -> ${resdir}/seed_<seed>/   ${PAR} seeds at a time, ${threads} thread(s) each"
echo "  GPU: ${SLURM_JOB_GPUS:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Preflight: dead interpreter (2026-09-07 GPFS purge), no GPU, missing diagrams,
# or a code tree without pi_normalize -- fail in seconds, not after the queue wait.
TAG_ENV="dtm_k${DTM_K}" DATADIR_ENV="$datadir" python - <<'PYEOF'
import os, sys
from pathlib import Path
import torch
from cloudforger.experiments.pi_multik import pi_multik

bad = [] if torch.cuda.is_available() else ["no CUDA device visible"]
if not hasattr(pi_multik, "fit_pi_norm"):
    bad.append("pi_multik.fit_pi_norm missing -- this code tree predates pi_normalize")
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
  local log="logs/dv3_pi_nocc/${SLURM_ARRAY_JOB_ID:-local}_${t}_${CELL_TAG[$cell]}_k${DTM_K}_${group}_s${seed}.log"
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
  echo "FAILED seeds (${#failed[@]}/${#SEEDS[@]}): ${failed[*]} -- see logs/dv3_pi_nocc/${SLURM_ARRAY_JOB_ID:-local}_${t}_*"
  exit 1
fi
echo "all ${#SEEDS[@]} seeds done: ${resdir}"
