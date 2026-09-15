#!/bin/bash

#SBATCH -J strauss_restarts
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH --array=0-69
#SBATCH --output=logs/strauss_restarts_%A_%a.out
#SBATCH --error=logs/strauss_restarts_%A_%a.err

# STRAUSS RESTART SELECTION -- fused DTM k=5,10,15, 10 seeds x 6 restarts,
# plus the vihrs topup from 5 to 10 seeds so the comparison is paired.
#
# WHY. The reported "PH 0.558 +/- 0.18, 3/5 seeds fail to converge, 3x worse
# than vihrs" is an artifact of an initialization lottery, not a feature or
# filtration failure. From logs/pi_multik_h0h1_all_25097048_2{0..4}.out and
# logs/pi_multik_h0_all_25097049_2{0..4}.out:
#
#   seed   H0+H1     H0 only
#   9371   0.4463    0.6873
#   9372   0.6990    0.2538
#   9373   0.6769    0.2341
#   9374   0.7183    0.7181
#   9375   0.2500    0.7009
#
# Every failing run sits at train loss 0.702-0.708, FLAT, until early
# stopping; every escaping run is already at train ~0.455 by epoch 25. The
# SAME seed escapes under one feature set and stalls under the other, so it
# is not a bad split -- it is decided by the initialization in the first ~25
# epochs. The plateau's per-target signature (beta 0.49 / gamma 0.87 /
# radius 0.75) is "learned the intensity channel, never learned the
# interaction". The three cleanly-escaped runs average 0.2460 against vihrs
# 0.1921 -- 28% behind, not 190%.
#
# MORE EPOCHS WILL NOT FIX THIS. The Strauss plateau is flat (train
# 0.7069 -> 0.7023 over 100 epochs before early stopping). That is the
# opposite of the classification pathology in slurm/classification_epochcap.sh,
# where val is monotone improving into the cap. Two pathologies, two fixes:
# restarts here, epoch budget there. Do not swap them.
#
# HOW. Each task is one (seed, restart). init_offset shifts ONLY the value
# passed to prepare_device() in experiments/pi_multik/pi_multik.py, i.e. the
# global torch RNG that drives weight init, dropout and batch order.
# train_val_test_indices(n, seed) draws from its own local
# torch.Generator(seed), so the train/val/test partition is byte-identical
# across restarts of a seed. Restarts are therefore paired to the same split,
# and selecting among them on VALIDATION loss is honest -- test is never
# consulted. init_offset=0 reproduces the existing default-path run exactly,
# so r0 doubles as a free reproducibility check against
# results/strauss/dtm_k5+10+15/pi_multik/seed_937{1..5}/.
#
#   Tasks  0-59: 10 seeds x 6 restarts (init_offset 0..5), pi_multik fused-k
#                -> results/strauss/dtm_k5+10+15/pi_multik/_runs/restart__r<j>/seed_<seed>/
#   Tasks 60-69: 10 seeds, vihrs baseline (topup; 9371..9375 already exist
#                and are skipped by is_done() in <1 s)
#                -> results/strauss/raw/vihrs_checkpointed/seed_<seed>/
#
# With a 4/10 per-restart escape rate, 6 restarts gives
# 1 - 0.6^6 = 95.3% probability that each seed escapes at least once.
#
# WHY 10 SEEDS FOR VIHRS: strauss/matern_cluster/aniso_thomas all have vihrs
# at n=5 only. The exact two-sided Wilcoxon signed-rank floor at n=5 is
# p = 2/2^5 = 0.0625, so NO paired claim on Strauss can reach p < 0.05 until
# both sides are at 10. strauss_vihrs.yaml already lists all 10 seeds.
#
# PREDICTION (state it before reading the result): selected-restart PH lands
# at ~0.246 +/- 0.01 with 0/10 failures, against vihrs 0.192 -- with the whole
# residual gap in gamma (0.297 vs 0.186), the interaction parameter a
# second-order summary measures most directly.
# KILL CRITERION: if selection lands at ~0.246 and the follow-up L(r) fusion
# arm does not improve gamma by >= 15%, stop -- the features have a real
# ceiling on Gibbs interaction and the paper says so.
#
# Resumable: is_done() skips any (seed, restart) whose results.pt exists.
#   sbatch --array=<comma,ids> slurm/strauss_restarts.sh
#
# ASSUMES (all already on disk, nothing to regenerate):
#   data/strauss/{clouds,adversarial_clouds}.pkl
#   data/strauss/dtm_k{5,10,15}/{,adversarial_}diagrams.pkl
#
# REQUIRES the init_offset knob in experiments/pi_multik/pi_multik.py.
# Verify before submitting:  grep -n init_offset src/cloudforger/experiments/pi_multik/pi_multik.py
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/strauss_restarts.sh
# then, once every task has finished:
#   python scripts/collect_restarts.py

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

PI_CFG=configs/runs/strauss/strauss_pi_multik_k5k10k15.yaml
VIHRS_CFG=configs/runs/strauss/strauss_vihrs.yaml

SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}
N_RESTARTS=6                                # init_offset 0..5
N_PI=$(( N_SEEDS * N_RESTARTS ))            # 60
NTASKS=$(( N_PI + N_SEEDS ))                # 70 -- must equal the --array span (0-69)

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/strauss_restarts.sh)}"
if (( t >= NTASKS )); then
  echo "task $t >= $NTASKS -- nothing to do (check --array span)"; exit 0
fi

echo "Host: $(hostname)"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Same existence-only preflight as slurm/classification_epochcap.sh: the
# torch/cloudforger imports catch a purged env before a GPU slot is burned
# (jobs 25811867/75/76/77 lost all 76 tasks that way on 2026-09-07).
python - <<'PYEOF'
import sys
from pathlib import Path
import torch  # noqa: F401
import cloudforger  # noqa: F401
req = ["data/strauss/clouds.pkl", "data/strauss/adversarial_clouds.pkl"]
req += [f"data/strauss/dtm_k{k}/{a}diagrams.pkl"
        for k in (5, 10, 15) for a in ("", "adversarial_")]
bad = [f"{f}: MISSING" for f in req if not Path(f).exists()]
bad += [f"{f}: EMPTY" for f in req if Path(f).exists() and Path(f).stat().st_size == 0]
if bad:
    print("preflight FAILED:", *bad, sep="\n  "); sys.exit(1)
print(f"preflight OK (torch {torch.__version__}, cuda={torch.cuda.is_available()})")
PYEOF

if (( t < N_PI )); then
  # pi_multik restart arm. Seed varies fastest so that a partial array still
  # covers whole restart rounds rather than a few seeds at every restart.
  r=$(( t / N_SEEDS ))
  seed="${SEEDS[$(( t % N_SEEDS ))]}"
  echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  pi_multik  seed=${seed}  init_offset=${r}  run-tag=restart__r${r}"
  echo "  config=${PI_CFG}"
  python -u scripts/train.py "$PI_CFG" --seed "$seed" --run-tag "restart__r${r}" \
      --set "method.params.init_offset=${r}"
else
  seed="${SEEDS[$(( t - N_PI ))]}"
  echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  vihrs topup  seed=${seed}  (no run-tag; is_done skips 9371..9375)"
  echo "  config=${VIHRS_CFG}"
  python -u scripts/train.py "$VIHRS_CFG" --seed "$seed"
fi
