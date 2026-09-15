#!/bin/bash

#SBATCH -J dv3_cl_prep
#SBATCH -p compute,computeshort
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/dv3_cl_prep_%A_%a.out
#SBATCH --error=logs/dv3_cl_prep_%A_%a.err

# STAGE 1 of 2 -- DV3 classical baselines: summary-function feature caches.
# CPU ONLY. No diagrams needed, so this (and stage 2) runs while the DTM
# diagrams are still being computed.
#
# Normally submitted by slurm/dv3_classical_submit.sh, which puts parameter
# estimation (inference) on the critical path and classification after it.
#
# WHY A SEPARATE STAGE. Every vihrs run featurizes train + A + B + C before
# training, and the caches are split-independent (whole file, file order), so
# the 350 GPU runs of stage 2 share them. Built here once on CPUs, every GPU
# task starts training in seconds instead of 10 runs each burning GPU time on
# the same O(n^2) L(r) pass.
#
#   tasks 0-11  PARAMETER ESTIMATION -- one (pass, family) each, t = pass*4 + family
#               family  0 thomas  1 nested  2 matern2  3 lgcp
#               pass    0 clouds.lfunc_cache.npz        (L only: the vihrs L arm AND
#                                                         the PH fusion configs' include_lfunc)
#                       1 clouds.summ_fg0250_cache.npz  (L, F, G, J on the 0.25 F/G grid)
#                       2 clouds.summ_fg0080_cache.npz  (L, F, G, J on the 0.08 F/G grid)
#               next to data/dv3/{train,A,B,C}/<family>/clouds.pkl. ~41k clouds
#               per task; the three passes write different files, so they run
#               in parallel. Expect a few minutes each on 8 CPUs.
#   task 12     CLASSIFICATION -- merges the 5 families into data/dv3/<set>/_classify/
#               (dv3_classification_bundle.py --clouds-only; its pickle write is
#               not atomic, hence one task), then all three passes
#               (clouds.{lfunc,summ_fg0250,summ_fg0080}_classify_cache.npz) over
#               ~166k clouds. Submit it with MORE CPUs (~20-30 min at 16):
#                 sbatch --array=12 --cpus-per-task=16 --mem=64G slurm/dv3_classical_prep.sh
#
# The fg caches hold ALL FOUR channels, so every channel-subset arm of stage 2
# (L+F+G+J, G, F) shares one pass per F/G/J grid.
#
# HOW. scripts/train.py with method.params.prepare_only=true runs vihrs's
# data preparation (resolve paths, featurize, write caches) and stops before
# training: no model, no results dir, no ledger row. feature_jobs fans the
# per-cloud curves out to a pool of SLURM_CPUS_PER_TASK processes
# (deterministic per cloud, so the caches equal a serial build).
#
# The 1 h limit keeps every task eligible for BOTH compute and computeshort
# (separate per-user QOS caps; dv3_dtm5 fills the compute one). Each (set,
# group, pass) cache is written atomically as it completes, so a timed-out
# task resumes where it stopped on re-submit (sbatch --array=<ids> ...).

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

FAMILY_LIST=(thomas nested matern2 lgcp)
PASS_FG=("" 0.25 0.08)                    # "" = the L-only pass
PASS_CACHE=(lfunc summ_fg0250 summ_fg0080)
SETS=(train A B C)
JOBS="${SLURM_CPUS_PER_TASK:-8}"
N_PARAMS=$(( ${#PASS_FG[@]} * ${#FAMILY_LIST[@]} ))   # 12

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/dv3_classical_prep.sh)}"
if (( t < N_PARAMS )); then
  group="${FAMILY_LIST[$(( t % ${#FAMILY_LIST[@]} ))]}"
  passes=( $(( t / ${#FAMILY_LIST[@]} )) )
  CFG_L=configs/runs/dv3/${group}/vihrs.yaml
  CFG_U=configs/runs/dv3/${group}/summstats_lfgj.yaml
  suffix=_cache.npz
elif (( t == N_PARAMS )); then
  group=_classify
  passes=(0 1 2)
  CFG_L=configs/runs/dv3/classify/vihrs.yaml
  CFG_U=configs/runs/dv3/classify/summstats_lfgj.yaml
  suffix=_classify_cache.npz
else
  echo "task $t > ${N_PARAMS} -- nothing to do (check --array span)"; exit 0
fi

echo "Host: $(hostname)  partition ${SLURM_JOB_PARTITION:-?}  job ${SLURM_JOB_ID:-unset}  task ${t}  ->  group=${group}  passes=${passes[*]}  feature_jobs=${JOBS}"

if [[ "$group" == "_classify" ]]; then
  echo; echo "== merging the per-family clouds into data/dv3/<set>/_classify/ (skipped if up to date)"
  python -u scripts/processing/dv3_classification_bundle.py --clouds-only --sets "${SETS[@]}"
fi

prep() {  # prep <config> [extra --set ...]
  local cfg="$1"; shift
  echo; echo "== prepare_only: ${cfg} $*"
  python -u scripts/train.py "$cfg" --seed 9371 \
      --set method.params.prepare_only=true \
      --set "method.params.feature_jobs=${JOBS}" "$@"
}

CACHES=()
for k in "${passes[@]}"; do
  if [[ -z "${PASS_FG[$k]}" ]]; then
    prep "$CFG_L"
  else
    prep "$CFG_U" --set "method.params.fg_r_max=${PASS_FG[$k]}"
  fi
  CACHES+=("clouds.${PASS_CACHE[$k]}${suffix}")
done

# train.py reports a failed seed and still exits 0, so check the product.
missing=0
echo; echo "== caches for ${group}"
for s in "${SETS[@]}"; do
  for c in "${CACHES[@]}"; do
    f="data/dv3/${s}/${group}/${c}"
    if [[ -f "$f" ]]; then
      echo "  ok       $f  ($(du -h "$f" | cut -f1))"
    else
      echo "  MISSING  $f"; missing=1
    fi
  done
done
if (( missing )); then
  echo "prep FAILED for ${group}: see the traceback above"; exit 1
fi
echo "prep OK for ${group}"
