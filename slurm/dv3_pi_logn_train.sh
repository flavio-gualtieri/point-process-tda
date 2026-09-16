#!/bin/bash

#SBATCH -J dv3_pi_logn
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -t 03:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/dv3_pi_logn_%A_%a.out
#SBATCH --error=logs/dv3_pi_logn_%A_%a.err

# CLASSIFICATION persistence-image cells WITH the point count n(x).
#
# WHY. Every classification PI run so far set include_log_n=false, as a
# "topology-only" baseline, while every classical (vihrs) model it is compared
# against always receives n(x) -- there is no switch to turn it off in
# baselines/vihrs.py. That was harmless under the old design, but the model
# parameters are defined in units of the mean point spacing (s = sigma *
# sqrt(nbar), tau = R * sqrt(nbar), ...) while a persistence diagram is in
# absolute units, so without n(x) the network cannot convert one to the other
# at all. n(x) is NOT a label leak here: the generator draws nbar log-uniformly
# and identically for every family (configs/generation/dv3.yaml, `nbar`), so
# the point count carries no class information on its own.
#
# The parameter-estimation runs already get n(x) (pi_multik defaults
# include_log_n to `not is_classify`), so only classification is re-run here.
#
#   cell  dims   include_log_n  results subdir (results/dv3_classify/dtm_k<K>/)
#   0     H0+H1  true           pi_multik_h01_nocc_norm_logn
#   1     H0     true           pi_multik_h0_nocc_norm_logn
#
# Each is the exact twin of an existing cell (pi_multik_h01_nocc_norm,
# pi_multik_h0_nocc_norm) with n(x) as the only difference, so the pair
# isolates what the point count is worth. coordconv off and
# pi_normalize=channel in both, matching those runs.
#
# Everything else matches configs/runs/dv3/classify/ph_pi.yaml (same network,
# 600 epochs, early stopping at 60). 10 seeds per cell.
#
# NEEDS, for train/A/B/C under data/dv3/<set>/_classify/:
#   dtm_k<K>/diagrams.pkl   the merged per-k diagram bundle
#   clouds.pkl              the merged cloud bundle -- n(x) is joined from it
#                           by seed, and it is NOT needed when include_log_n
#                           is false, which is why these runs need it and the
#                           earlier ones did not. Build with
#                           scripts/processing/dv3_classification_bundle.py.
#
# Submit through slurm/dv3_pi_logn_submit.sh.
# Timing: the matching no-n classification cells took ~56 min with 5 seeds at
# a time and 24 GB peak RSS, hence PAR=5 and -t 03:00:00.
#
# Resumable: train.py skips a seed whose results.pt exists; a task exits
# non-zero if any seed has no results.pt at the end, naming the seed logs.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs logs/dv3_pi_logn

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

DTM_K="${DTM_K:-5}"
SEEDS=(9371 9372 9373 9374 9375 9376 9377 9378 9379 9380)

CELL_TAG=(h01_nocc_norm_logn  h0_nocc_norm_logn)
CELL_DIMS=("0,1"              "0")
N_CELLS=${#CELL_TAG[@]}

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/dv3_pi_logn_train.sh)}"
if (( t >= N_CELLS )); then
  echo "task $t out of range (0-$(( N_CELLS - 1 ))) -- nothing to do"; exit 0
fi
cell=$t
dims="${CELL_DIMS[$cell]}"
subdir="pi_multik_${CELL_TAG[$cell]}"
group=classify
datadir=_classify
process=dv3_classify
PAR="${PAR:-5}"

threads=$(( ${SLURM_CPUS_PER_TASK:-$PAR} / PAR )); (( threads >= 1 )) || threads=1
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads

cfg="configs/runs/dv3/${group}/ph_pi.yaml"
overrides=(
  --set "filtration=[{name: dtm, params: {k: ${DTM_K}, q: 2.0, maxdim: 1}}]"
  --set "method.params.homology_dims=[${dims}]"
  --set "method.params.include_log_n=true"
  --set "method.params.coordconv=false"
  --set "method.params.pi_normalize=channel"
  --set "method.params.results_subdir=${subdir}"
)
resdir="results/${process}/dtm_k${DTM_K}/${subdir}"

echo "Host: $(hostname)  job ${SLURM_ARRAY_JOB_ID:-unset}_${t}  ->  ${subdir} (H[${dims}], n(x) ON, coordconv off, pi_normalize=channel)  DTM k=${DTM_K}"
echo "  config=${cfg}"
echo "  results -> ${resdir}/seed_<seed>/   ${PAR} seeds at a time, ${threads} thread(s) each"
echo "  GPU: ${SLURM_JOB_GPUS:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Preflight. Beyond the usual checks (live interpreter, GPU, diagrams present)
# this one actually PERFORMS the n(x) join on the train split, because that is
# the code path these runs turn on for the first time: the classification
# diagram bundles carry globally-unique seeds while clouds.pkl has shipped with
# either seed convention, and _classification_n_points reconciles them. A
# mismatch there raises KeyError -- far better to see it in seconds here than
# three hours into the array.
TAG_ENV="dtm_k${DTM_K}" DATADIR_ENV="$datadir" python - <<'PYEOF'
import os, sys
from pathlib import Path
import numpy as np
import torch
from cloudforger.experiments.pi_multik import pi_multik

tag, datadir = os.environ["TAG_ENV"], os.environ["DATADIR_ENV"]
bad = [] if torch.cuda.is_available() else ["no CUDA device visible"]
if not hasattr(pi_multik, "fit_pi_norm"):
    bad.append("pi_multik.fit_pi_norm missing -- this code tree predates pi_normalize")
if not hasattr(pi_multik, "_classification_n_points"):
    bad.append("pi_multik._classification_n_points missing -- this code tree cannot join n(x) "
               "for classification")

for s in ("train", "A", "B", "C"):
    base = Path("data/dv3") / s / datadir
    for p in (base / tag / "diagrams.pkl", base / "clouds.pkl"):
        if not p.exists():
            bad.append(f"{p}: MISSING")

if not bad:
    # The join itself, on the split the run will train from.
    try:
        base = Path("data/dv3/train") / datadir
        bundle = pi_multik._load_pickle(base / tag / "diagrams.pkl")
        seeds = np.asarray(bundle["seeds"] if isinstance(bundle, dict) and "seeds" in bundle
                           else [d["seed"] for d in bundle["diagrams"]])
        n = pi_multik._classification_n_points(base / "clouds.pkl", bundle, seeds)
        if len(n) != len(seeds):
            bad.append(f"n(x) join returned {len(n)} rows for {len(seeds)} diagrams")
        elif not np.isfinite(n).all() or (n <= 0).any():
            bad.append("n(x) join produced non-positive or non-finite counts")
        else:
            print(f"n(x) join OK on train/{datadir}: {len(n)} clouds, "
                  f"n in [{int(n.min())}, {int(n.max())}], median {int(np.median(n))}")
    except Exception as exc:  # noqa: BLE001 -- any failure here is a hard stop
        bad.append(f"n(x) join FAILED: {exc!r}")

if bad:
    print("preflight FAILED:", *bad, sep="\n  "); sys.exit(1)
print(f"preflight OK (torch {torch.__version__}, {torch.cuda.get_device_name(0)})")
PYEOF

run_seed() {
  local seed="$1"
  local log="logs/dv3_pi_logn/${SLURM_ARRAY_JOB_ID:-local}_${t}_${CELL_TAG[$cell]}_k${DTM_K}_s${seed}.log"
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
  echo "FAILED seeds (${#failed[@]}/${#SEEDS[@]}): ${failed[*]} -- see logs/dv3_pi_logn/${SLURM_ARRAY_JOB_ID:-local}_${t}_*"
  exit 1
fi
echo "all ${#SEEDS[@]} seeds done: ${resdir}"
echo
echo "next, on the login node:"
echo "  python scripts/evaluate_testset.py ${resdir} \\"
echo "         results/${process}/dtm_k${DTM_K}/pi_multik_${CELL_TAG[$cell]%_logn}"
echo "  (the same cell without n(x) -- the pair isolates what the point count buys)"
