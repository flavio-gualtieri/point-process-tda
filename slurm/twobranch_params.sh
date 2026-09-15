#!/bin/bash

#SBATCH -J twobranch_params
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 6
#SBATCH --cpus-per-gpu=6
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=14G
#SBATCH --gres=gpu:1
#SBATCH --array=0-239
#SBATCH --output=logs/twobranch_params_%A_%a.out
#SBATCH --error=logs/twobranch_params_%A_%a.err

# The same PH-branch ablation as slurm/twobranch_classify.sh, for PARAMETER
# ESTIMATION -- one two-branch model, PH branch off vs on, curve branch
# [L, F, G] at full resolution in both.
#
# WHY RUN IT when fgp__ already said "L + F + G wins parameter estimation":
# that comparison had the same features-vs-architecture confound as the
# classification one (coarse 24-column MLP side vector for PH, full curves
# through a CNN for vihrs), AND the PH (+) F/G arms ran with the per-column F/G
# scaling bug now fixed in build_extra. This is the clean version. The
# expectation is still that the PH branch adds little here -- F and G carried
# exactly the density/count signal PH used to contribute -- and the
# per-target table in scripts/collect_twobranch.py is where to check that.
#
#   processes  thomas, matern_cluster, nested_thomas  (k5 fusion configs)
#   arms       curves (use_pi false)  /  PHcurves (use_pi true)
#   restarts   init_offset 0..3, selected on VALIDATION loss per seed
#
#   3 processes x 4 restarts x 2 arms x 10 seeds -> 240 tasks (~1-3 min each).
#   Ordered process-major, then restart, so each process's restart-0
#   comparison (20 tasks) is complete early.
#
# lgcp / lgcp_strauss are NOT here yet: they need k=5 diagrams
# (slurm/diagrams_{compute,merge}_lgcp.sh), their summary caches
# (slurm/summstats_featurize_params.sh --array=3-4) and a fusion_k5.yaml each.
#
# REFERENCE (n=10, k5, no restarts): vihrs L+F+G thomas 0.1065 /
# matern 0.1142 / nested 0.1983. The `curves` arm should sit near these.
#
# Resumable: sbatch --array=<comma,ids> slurm/twobranch_params.sh
# Read out:
#   python scripts/collect_twobranch.py --processes thomas matern_cluster nested_thomas

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

PROCESSES=(thomas matern_cluster nested_thomas)
SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)
N_SEEDS=${#SEEDS[@]}
ARM_TAG=(curves PHcurves)
ARM_USE_PI=(false true)
N_ARMS=${#ARM_TAG[@]}
N_RESTARTS=4

PER_PROC=$(( N_RESTARTS * N_ARMS * N_SEEDS ))   # 80
NTASKS=$(( ${#PROCESSES[@]} * PER_PROC ))       # 240 -- must equal the --array span

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/twobranch_params.sh)}"
if (( t >= NTASKS )); then
  echo "task $t >= $NTASKS -- nothing to do (check --array span)"; exit 0
fi
proc="${PROCESSES[$(( t / PER_PROC ))]}"
rem=$(( t % PER_PROC ))
r=$(( rem / (N_ARMS * N_SEEDS) ))
rem2=$(( rem % (N_ARMS * N_SEEDS) ))
arm=$(( rem2 / N_SEEDS ))
seed="${SEEDS[$(( rem2 % N_SEEDS ))]}"
tag="${ARM_TAG[$arm]}"
cfg="configs/runs/${proc}/fusion_k5.yaml"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  task ${t}  ->  process=${proc}  arm=${tag}  seed=${seed}  restart=${r}"
echo "  config=${cfg}  run-tag=tb__${tag}__r${r}  Assigned GPU: ${SLURM_JOB_GPUS:-unset}"

PROC_ENV="$proc" python - <<'PYEOF'
import os, sys
from pathlib import Path
import torch  # noqa: F401
from cloudforger.experiments.pi_multik.pi_multik import CurveEncoder  # noqa: F401
p = os.environ["PROC_ENV"]
req = [f"data/{p}/clouds.pkl", f"data/{p}/adversarial_clouds.pkl",
       f"data/{p}/dtm_k5/diagrams.pkl", f"data/{p}/dtm_k5/adversarial_diagrams.pkl",
       f"data/{p}/clouds.summ_fg0250_cache.npz", f"data/{p}/adversarial_clouds.summ_fg0250_cache.npz",
       f"configs/runs/{p}/fusion_k5.yaml"]
bad = [f"{f}: MISSING" for f in req if not Path(f).exists()]
if bad:
    print("preflight FAILED:", *bad, "(curve caches: slurm/summstats_featurize_params.sh)", sep="\n  ")
    sys.exit(1)
print(f"preflight OK (torch {torch.__version__}, cuda={torch.cuda.is_available()})")
PYEOF

python -u scripts/train.py "$cfg" --seed "$seed" --run-tag "tb__${tag}__r${r}" \
    --set "method.params.use_pi=${ARM_USE_PI[$arm]}" \
    --set "method.params.curve_channels=[L,F,G]" \
    --set "method.params.fg_r_max=0.25" \
    --set "method.params.include_lfunc=0" \
    --set "method.params.include_fgfunc=0" \
    --set "method.params.init_offset=${r}"
