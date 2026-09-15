#!/bin/bash

#SBATCH -J summ_classify
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 6
#SBATCH --cpus-per-gpu=6
#SBATCH -t 01:30:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-59
#SBATCH --output=logs/summ_classify_%A_%a.out
#SBATCH --error=logs/summ_classify_%A_%a.err

# STAGE 2 of 2 -- THE STRONG CLASSICAL BASELINE for process classification.
#
# WHY THIS EXISTS. The current headline is PH+L(r) 0.8806 vs vihrs 0.8379 on
# 4-way family classification. vihrs consumes exactly ONE summary function
# (Ripley's L(r)-r) plus n(x), so a reviewer can read that margin as "L(r) is
# an impoverished feature set" rather than "topology carries information the
# standard summaries do not". The textbook answer for MODEL DISCRIMINATION is
# K/L together with F (empty space), G (nearest neighbour) and
# J = (1-G)/(1-F). This job runs that union through the SAME 1-D CNN, on the
# SAME clouds, seeds, split and epoch budget.
#
# This can kill the framing, which is exactly why it runs first. If PH keeps a
# clear margin over L+F+G+J, the classification result is much harder to
# dismiss. If the margin collapses, the honest paper is "PH matches the union
# of the standard summaries without their stationarity and edge-correction
# assumptions" -- worth knowing now rather than in review.
#
#   arm idx  run-tag            channels     note
#   0        summ__L            L            REGRESSION CHECK: reproduces the
#                                            published vihrs baseline through
#                                            the new code path (in_channels=1)
#   1        summ__FG           F,G          the two new functions alone
#   2        summ__LFG          L,F,G        union without the derived ratio
#   3        summ__LFGJ         L,F,G,J      THE BASELINE TO BEAT
#   4        summ__LFGJ_fg08    L,F,G,J      same, F/G/J on a 0.08 radius grid
#   5        summ__J            J            J alone -- the classical
#                                            discrimination statistic by itself
#
#   6 arms x 10 seeds (9371..9380) -> 60 array tasks, all independent.
#
# ARM 0 IS THE CONTROL, NOT FILLER. The vihrs edits are default-preserving
# (VihrsCNN gained in_channels=1; the per-channel z-score reduces to the old
# global one for a single channel), but "should be identical" is not evidence.
# Arm 0 must land on the published 0.8379 +/- 0.0031 -- within run-to-run
# nondeterminism, which is NOT negligible here: the 200-vs-600 epoch reruns
# moved individual vihrs seeds by up to 0.011 with no config change that
# bound. Compare the ARM MEAN over 10 seeds, not per-seed values.
#
# ARM 4 EXISTS SO THE BASELINE IS NOT HANDICAPPED. F and G saturate at 1 far
# below Ripley's r_max = 0.25, so on the shared grid most of their 513 samples
# carry no signal. fg_r_max puts F/G/J on their own grid of the same length.
# 0.25 is the zero-tuning mirror of L's grid; 0.08 is the informative range.
# Report whichever wins on VALIDATION accuracy as "the" baseline -- picking on
# test would be exactly the cherry-picking this job exists to rule out.
#
# COMPARE AGAINST: results/classification/dtm_k5/pi_multik/_runs/cap600__pca2/
# (0.8806 +/- 0.0028, n=10) and cap600__Loff (0.8729, PH with no L at all).
# Those ran at n_epochs=600 / patience=60, matched here.
#
# REQUIRES STAGE 1 (slurm/summstats_featurize.sh) to have finished: the
# preflight below hard-fails if the F/G/J cache is missing rather than letting
# 60 GPU tasks each rebuild it. Chain them with
#   fid=$(sbatch --parsable slurm/summstats_featurize.sh)
#   sbatch --dependency=afterok:$fid slurm/summstats_classify.sh
#
# Resumable: is_done() skips a seed whose results.pt exists.
#   sbatch --array=<comma,ids> slurm/summstats_classify.sh
#
# Aggregate with:
#   python scripts/collect_fusion_results.py     (PH arms)
#   for f in results/classification/raw/vihrs/_runs/summ__*/seed_*/results.json; \
#     do jq -r '[.config.run_tag,.seed,.test_accuracy]|@tsv' $f; done | sort

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CFG=configs/runs/classification/summstats_kfgj.yaml
SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}

ARM_TAG=(L        FG        LFG         LFGJ          LFGJ_fg08     J)
ARM_CH=("[L]"    "[F,G]"   "[L,F,G]"   "[L,F,G,J]"   "[L,F,G,J]"   "[J]")
ARM_FG=(0.25      0.25      0.25        0.25          0.08          0.25)
N_ARMS=${#ARM_TAG[@]}

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/summstats_classify.sh)}"
if (( t >= N_ARMS * N_SEEDS )); then
  echo "task $t >= $((N_ARMS * N_SEEDS)) -- nothing to do (check --array span)"; exit 0
fi
arm=$(( t / N_SEEDS ))
seed="${SEEDS[$(( t % N_SEEDS ))]}"
tag="${ARM_TAG[$arm]}"
ch="${ARM_CH[$arm]}"
fg="${ARM_FG[$arm]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  arm=${tag}  seed=${seed}"
echo "  summary_channels=${ch}  fg_r_max=${fg}  run-tag=summ__${tag}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Preflight: dead interpreter (the 2026-09-07 GPFS purge reaped cloud-env's
# stdlib and killed 76 array tasks in <1 s), missing cloud/diagram bundles,
# and -- unique to this job -- a missing STAGE 1 cache. Arm 0 is exempt: it
# uses the historical L-only cache, not the F/G/J one.
ARM_TAG_ENV="$tag" FG_ENV="$fg" python - <<'PYEOF'
import os, sys
from pathlib import Path
import torch  # noqa: F401  -- the interpreter/env check
from cloudforger.baselines import summstats  # noqa: F401  -- the new module imports

req = ["data/classification/clouds.pkl", "data/classification/adversarial_clouds.pkl"]
req += [f"data/classification/dtm_k{k}/{a}diagrams.pkl"
        for k in (5, 10, 15) for a in ("", "adversarial_")]
bad = [f"{f}: MISSING" for f in req if not Path(f).exists()]

if os.environ["ARM_TAG_ENV"] != "L":
    stem = f"summ_fg{round(float(os.environ['FG_ENV']) * 1000):04d}"
    for name in (f"clouds.{stem}_classify_cache.npz",
                 f"adversarial_clouds.{stem}_classify_cache.npz"):
        p = Path("data/classification") / name
        if not p.exists():
            bad.append(f"{p}: MISSING -- run slurm/summstats_featurize.sh first")
if bad:
    print("preflight FAILED:", *bad, sep="\n  "); sys.exit(1)
print(f"preflight OK (torch {torch.__version__}, cuda={torch.cuda.is_available()})")
PYEOF

python -u scripts/train.py "$CFG" --seed "$seed" --run-tag "summ__${tag}" \
    --set "method.params.summary_channels=${ch}" \
    --set "method.params.fg_r_max=${fg}"
