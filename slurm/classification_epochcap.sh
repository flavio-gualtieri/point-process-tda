#!/bin/bash

#SBATCH -J cls_cap600
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 6
#SBATCH --cpus-per-gpu=6
#SBATCH -t 01:30:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-59
#SBATCH --output=logs/cls_cap600_%A_%a.out
#SBATCH --error=logs/cls_cap600_%A_%a.err

# CLASSIFICATION EPOCH-CAP RERUN -- n_epochs 200 -> 600, patience 30 -> 60.
#
# WHY. Every configs/runs/classification/*.yaml sets n_epochs: 200 (the
# parameter configs use 500). The arms reported in TODAY.MD 3 as "collapsed"
# are exactly the arms that hit that cap while still improving:
#
#   logs/fusion_classification_25847700_12.out  (raw8, seed 9373, acc 0.8002)
#     epoch  75 | train 0.4646 | val 0.4580 | train_acc 0.7667 | val_acc 0.7707
#     epoch 200 | train 0.4206 | val 0.4164 | train_acc 0.7960 | val_acc 0.8023
#   -- val CE strictly monotone decreasing to the cap, val_acc ABOVE train_acc
#      throughout, and no early-stopping line. That is truncation, not the
#      overfitting TODAY.MD attributes it to.
#
# Meanwhile the CONVERGED raw8 seeds average 0.8861 -- above the paper's
# headline pca2 number (0.8797) -- and clean pca4 beats pca2 on 7 of the 9
# seeds where both converge. If that survives the lifted cap, the "the CE
# head overfits the noise directions" mechanism is wrong and the headline
# classification number goes up.
#
#   arm idx  run-tag             include_lfunc  lfunc_pca  collapse rate @200ep
#   0        cap600__Loff        0              0          0/10  (early-stops at 150, val worsening)
#   1        cap600__raw8        8              0          7/10
#   2        cap600__pca2        16             2          0/10  (hits cap, already flat)
#   3        cap600__pca3        16             3          3/10
#   4        cap600__pca4        16             4          1/10
#   5        cap600__vihrs       -- the vihrs baseline, SAME cap (see below)
#
# pca3 is included on purpose: raw8 7/10, pca3 3/10, pca4 1/10, pca2 0/10 is
# a dose-response in collapse rate against PCA truncation. If lifting the cap
# flattens that curve, conditioning is the mechanism and the evidence is much
# stronger than the two extreme arms alone.
#
# FAIRNESS. The vihrs classification baseline is ALSO capped at 200
# (configs/runs/classification/vihrs_lr_nx_k5k10k15.yaml), so it is rerun at
# the same 600/60. It previously early-stopped at epoch 63 with best
# checkpoint at 33, so it is expected NOT to move -- but the comparison is
# only defensible if both sides get the same budget. Do not report a lifted
# PH number against the 200-epoch vihrs number.
#
#   6 arms x 10 seeds (9371..9380) -> 60 array tasks, one GPU each, all
#   independent. No % throttle: this is ~60 x ~14 min of GPU, run it wide.
#
# NON-DESTRUCTIVE: every arm writes under --run-tag cap600__*, i.e.
#   results/classification/dtm_k5/pi_multik/_runs/cap600__<arm>/seed_<seed>/
#   results/classification/raw/vihrs/_runs/cap600__vihrs/seed_<seed>/
# The existing fusion__* / default results (the paper's current numbers) are
# untouched, so the comparison is cap600__X vs fusion__X at the same seed.
#
# Resumable: is_done() skips a seed whose results.pt already exists, so a
# plain re-submit only fills gaps. Rerun specific failures with
#   sbatch --array=<comma,ids> slurm/classification_epochcap.sh
#
# ASSUMES (all already on disk, nothing to regenerate):
#   data/classification/{clouds,adversarial_clouds}.pkl
#   data/classification/dtm_k5/{,adversarial_}diagrams.pkl
#   data/classification/{clouds,adversarial_clouds}.lfunc_classify_cache.npz
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/classification_epochcap.sh
# Aggregate with:
#   python scripts/collect_fusion_results.py

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

PI_CFG=configs/runs/classification/fusion_k5.yaml
VIHRS_CFG=configs/runs/classification/vihrs_lr_nx_k5k10k15.yaml

SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}

# arm 5 (vihrs) has no include_lfunc/lfunc_pca -- the "-" entries are never read.
ARM_TAG=(Loff raw8 pca2 pca3 pca4 vihrs)
ARM_NL=(0    8    16   16   16   -)
ARM_PCA=(0   0    2    3    4    -)
N_ARMS=${#ARM_TAG[@]}

N_EPOCHS=600
PATIENCE=60

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/classification_epochcap.sh)}"
if (( t >= N_ARMS * N_SEEDS )); then
  echo "task $t >= $((N_ARMS * N_SEEDS)) -- nothing to do (check --array span)"; exit 0
fi
arm=$(( t / N_SEEDS ))
seed="${SEEDS[$(( t % N_SEEDS ))]}"
tag="${ARM_TAG[$arm]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  arm=${tag}  seed=${seed}"
echo "  n_epochs=${N_EPOCHS}  early_stopping_patience=${PATIENCE}  run-tag=cap600__${tag}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Preflight: catches BOTH failure modes that have actually bitten this
# project -- a dead interpreter (the 2026-09-07 GPFS purge reaped cloud-env's
# stdlib, so every array task died in <1 s before any repo code ran) and a
# missing input bundle. Deliberately existence-only, not a full pickle.load:
# the 35k-cloud bundle is read by training anyway, where a corrupt file fails
# just as loudly, and 60 redundant full loads is pure waste.
python - <<'PYEOF'
import sys
from pathlib import Path
import torch  # noqa: F401  -- import is the interpreter/env check
import cloudforger  # noqa: F401
req = [
    "data/classification/clouds.pkl",
    "data/classification/adversarial_clouds.pkl",
    "data/classification/dtm_k5/diagrams.pkl",
    "data/classification/dtm_k5/adversarial_diagrams.pkl",
    "data/classification/clouds.lfunc_classify_cache.npz",
    "data/classification/adversarial_clouds.lfunc_classify_cache.npz",
]
bad = [f"{f}: MISSING" for f in req if not Path(f).exists()]
bad += [f"{f}: EMPTY" for f in req if Path(f).exists() and Path(f).stat().st_size == 0]
if bad:
    print("preflight FAILED:", *bad, sep="\n  "); sys.exit(1)
print(f"preflight OK (torch {torch.__version__}, cuda={torch.cuda.is_available()})")
PYEOF

if [[ "$tag" == "vihrs" ]]; then
  python -u scripts/train.py "$VIHRS_CFG" --seed "$seed" --run-tag "cap600__${tag}" \
      --set "method.params.n_epochs=${N_EPOCHS}" \
      --set "method.params.early_stopping_patience=${PATIENCE}"
else
  nl="${ARM_NL[$arm]}"
  pca="${ARM_PCA[$arm]}"
  python -u scripts/train.py "$PI_CFG" --seed "$seed" --run-tag "cap600__${tag}" \
      --set "method.params.include_lfunc=${nl}" \
      --set "method.params.lfunc_pca=${pca}" \
      --set "method.params.n_epochs=${N_EPOCHS}" \
      --set "method.params.early_stopping_patience=${PATIENCE}"
fi
