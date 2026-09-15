#!/bin/bash

#SBATCH -J twobranch_classify
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 6
#SBATCH --cpus-per-gpu=6
#SBATCH -t 01:30:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-79
#SBATCH --output=logs/twobranch_classify_%A_%a.out
#SBATCH --error=logs/twobranch_classify_%A_%a.err

# THE GATE: does the persistence-image branch add anything on top of the full
# L + F + G curves, when both sit in ONE model? Classification.
#
# WHY THIS REPLACES THE fgfus__ COMPARISON. The previous "PH adds to the
# union" evidence (slurm/fgfusion_classify.sh: PH (+) L (+) F/G 0.8896 vs
# vihrs L+F+G 0.8839, 8/10 seeds, p ~ 0.15) confounds features with
# architecture: PH got 24 COARSE scalar samples of L/F/G through an MLP side
# vector, while vihrs got three FULL 513-point curves through a 1-D CNN. This
# job puts both in a single two-branch model and switches the PH branch off
# and on, so the difference is the PH branch and nothing else:
#
#   curve branch   CurveEncoder = the vihrs conv trunk (Conv1d(C,64,7)-Pool5-
#                  Conv1d-Pool5-Conv1d, EXACTLY VihrsCNN's) over [L, F, G] at
#                  all 513 radii, then Linear -> 64 so it enters the head at
#                  the same width as the PI embedding
#   PH branch      the pi_multik CoordConv persistence-image encoder, DTM k=5
#   head           the usual MLP over [PI emb | curve emb | log n(x)]
#
#   arm        use_pi   curve_channels   run-tag
#   curves     false    [L, F, G]        tb__curves__r<j>     PH branch OFF
#   PHcurves   true     [L, F, G]        tb__PHcurves__r<j>   PH branch ON
#
#   2 arms x 4 restarts (init_offset 0..3) x 10 seeds -> 80 tasks.
#   Ordered so tasks 0-19 are restart 0 of BOTH arms: a complete
#   single-run comparison exists as soon as the first 20 finish.
#
# RESTARTS ARE PART OF THE PROTOCOL, not a rescue. This classifier's training
# landscape is bimodal (an initialization lottery, not overfitting -- see
# slurm/classification_epochcap.sh), so each (arm, seed) gets 4 restarts that
# share the seed's exact split (init_offset shifts only the global torch RNG;
# train_val_test_indices uses its own generator), and scripts/
# collect_twobranch.py keeps the restart with the lowest VALIDATION loss for
# BOTH arms before pairing. Test is never used to choose.
#
# PRE-STATED READ-OUT (decide before looking):
#   * PH branch helps  <=>  PHcurves - curves >= ~0.005 with p < 0.05.
#     Then the classification half of the paper uses PH + L + F + G.
#   * Otherwise the PH contribution is the diagnostic finding only
#     ("DTM persistence mostly re-encodes nearest-neighbour / empty-space
#     structure"), and the headline drops PH.
#
# REFERENCE (n=10, 600 epochs, no restarts): vihrs L+F+G 0.8839,
# PH (+) L (+) F/G coarse 0.8896, PH only 0.8729. The `curves` arm should land
# near vihrs L+F+G -- it is the same conv trunk on the same curves -- and is a
# sanity check on the new branch: if it is far off, look there first.
#
# ASSUMES (all on disk; nothing to regenerate):
#   data/classification/{,adversarial_}clouds.pkl
#   data/classification/dtm_k5/{,adversarial_}diagrams.pkl  (labels/seeds; PI for PHcurves)
#   data/classification/{,adversarial_}clouds.summ_fg0250_classify_cache.npz  (L, F, G curves)
#
# Resumable (is_done() skips finished seeds):
#   sbatch --array=<comma,ids> slurm/twobranch_classify.sh
# Read out:
#   python scripts/collect_twobranch.py --processes classification

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

CFG=configs/runs/classification/fusion_k5.yaml
SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}
ARM_TAG=(curves PHcurves)
ARM_USE_PI=(false true)
N_ARMS=${#ARM_TAG[@]}
N_RESTARTS=4

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/twobranch_classify.sh)}"
NTASKS=$(( N_RESTARTS * N_ARMS * N_SEEDS ))   # 80 -- must equal the --array span
if (( t >= NTASKS )); then
  echo "task $t >= $NTASKS -- nothing to do (check --array span)"; exit 0
fi
r=$(( t / (N_ARMS * N_SEEDS) ))
rem=$(( t % (N_ARMS * N_SEEDS) ))
arm=$(( rem / N_SEEDS ))
seed="${SEEDS[$(( rem % N_SEEDS ))]}"
tag="${ARM_TAG[$arm]}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  arm=${tag}  seed=${seed}  restart=${r}"
echo "  use_pi=${ARM_USE_PI[$arm]}  curve_channels=[L,F,G]  run-tag=tb__${tag}__r${r}"
echo "  Assigned GPU: ${SLURM_JOB_GPUS:-unset}"

python - <<'PYEOF'
import sys
from pathlib import Path
import torch  # noqa: F401  -- interpreter/env check (cloud-env lives on purgeable scratch)
from cloudforger.experiments.pi_multik.pi_multik import CurveEncoder  # noqa: F401
req = [
    "data/classification/clouds.pkl",
    "data/classification/adversarial_clouds.pkl",
    "data/classification/dtm_k5/diagrams.pkl",
    "data/classification/dtm_k5/adversarial_diagrams.pkl",
    "data/classification/clouds.summ_fg0250_classify_cache.npz",
    "data/classification/adversarial_clouds.summ_fg0250_classify_cache.npz",
]
bad = [f"{f}: MISSING" for f in req if not Path(f).exists()]
if bad:
    print("preflight FAILED:", *bad, "(curve caches: slurm/summstats_featurize.sh)", sep="\n  ")
    sys.exit(1)
print(f"preflight OK (torch {torch.__version__}, cuda={torch.cuda.is_available()})")
PYEOF

# No L / F / G scalar columns: in this model the summary functions enter ONLY
# through the curve branch, so the two arms differ in the PH branch alone.
#
# include_log_n=true overrides fusion_k5.yaml's `false` (which was held fixed
# there only so the old L-encoding sweep varied nothing else). vihrs L+F+G --
# the reference the `curves` arm is checked against -- feeds n(x) too, so
# with it on the curves arm is that baseline in all but its head. Both arms
# get it, so the PH ablation itself is unaffected. NOTE this makes the
# two-branch numbers not directly comparable to the earlier cap600__/fgfus__
# PH arms, which ran WITHOUT n(x); compare within this job.
python -u scripts/train.py "$CFG" --seed "$seed" --run-tag "tb__${tag}__r${r}" \
    --set "method.params.include_log_n=true" \
    --set "method.params.use_pi=${ARM_USE_PI[$arm]}" \
    --set "method.params.curve_channels=[L,F,G]" \
    --set "method.params.fg_r_max=0.25" \
    --set "method.params.include_lfunc=0" \
    --set "method.params.lfunc_pca=0" \
    --set "method.params.include_fgfunc=0" \
    --set "method.params.init_offset=${r}" \
    --set "method.params.n_epochs=600" \
    --set "method.params.early_stopping_patience=60"
