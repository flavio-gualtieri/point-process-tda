#!/bin/bash

#SBATCH -J diagrams_compute_lgcp
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=48G
#SBATCH -t 04:00:00
#SBATCH --array=0-959%600
#SBATCH --output=logs/diagrams_compute_lgcp_%A_%a.out
#SBATCH --error=logs/diagrams_compute_lgcp_%A_%a.err

# STAGE 1 of 2 -- persistence diagrams for lgcp and lgcp_strauss.
# Same shard -> stitch pipeline as slurm/diagrams_compute_strauss_aniso.sh
# (scripts/processing/diagrams_shard.py); the clouds already exist in
# data/{lgcp,lgcp_strauss}/{,adversarial_}clouds.pkl.
#
#   4 filtrations  (dtm_k5, dtm_k10, dtm_k15, rips)
# x 2 processes    (lgcp, lgcp_strauss)
# x NSHARDS = 120  shards each
#   = 960 array tasks, 1 CPU each, <= 600 running at once.
#
# ORDERED FILTRATION-MAJOR, dtm_k5 FIRST: tasks 0-239 are dtm_k5 for both
# processes. Every current experiment (fusion, two-branch, classification)
# uses k=5 only, so under the %600 throttle the k5 bundles finish first and
# can be merged (slurm/diagrams_merge_lgcp.sh --array=0-1) before k10/k15/rips
# are done. k10/k15/rips are only needed for the fused-k configs and for
# rebuilding the classification bundle with every filtration tag.
#
# ---------------------------------------------------------------------------
# MEMORY -- READ BEFORE SUBMITTING. These clouds are bigger than anything
# featurized so far:
#
#   process        mean pts   max pts   (for reference: matern_cluster max 1291,
#   lgcp             421       2344      aniso_thomas 1147, strauss 881)
#   lgcp_strauss     272       1407
#
# DTM-Rips at maxdim=1 with thresh=None builds the FULL 2-skeleton, whose
# size grows as n^3 / 6. Matern clouds at <=1291 points already needed
# 48-96 GB for their worst tasks; (2344 / 1291)^3 ~ 6x that puts the single
# largest LGCP diagram somewhere around 150-300 GB -- an estimate, not a
# measurement. So:
#
#   * 120 shards (not 80) keeps each task at ~66 diagrams, so GUDHI's
#     ~80 MB/diagram leak stays ~5 GB and each task's peak is set by its
#     largest cloud rather than by accumulated leak.
#   * EVERY TASK FIRST PRINTS ITS SHARD'S LARGEST CLOUD (the preamble below),
#     so an OUT_OF_MEMORY task's log tells you exactly how big a node it needs.
#   * 48 GB is the first-pass tier. Resubmit failures in a higher tier:
#       sacct -j <jobid> --format=JobID%20,State,MaxRSS -X | grep -E "OUT_OF_ME|TIMEOUT"
#       sbatch --array=<ids> --mem-per-cpu=128G -t 08:00:00 slurm/diagrams_compute_lgcp.sh
#     and, for any shard whose preamble reports a cloud above ~1800 points,
#     straight to the largest node you have (256 GB+).
#   * If some clouds are infeasible on any node, do NOT drop them silently --
#     the merge refuses a truncated bundle, and removing the densest LGCP
#     clouds would censor the high-intensity end of the design. Options are
#     (a) edge collapse before expansion (gudhi SimplexTree.collapse_edges,
#     exact for flag complexes -- would need an equivalence check against the
#     current diagrams on a few hundred existing clouds before use), or
#     (b) regenerating LGCP with an intensity cap in the design constraints.
# rips tasks are seconds (ripser never materializes the complex).
# ---------------------------------------------------------------------------
#
# Resumable: a shard whose output exists is skipped, so a plain re-submit only
# fills gaps.
#
# Submit from the repo root, then once every task of a filtration is done:
#   sbatch slurm/diagrams_compute_lgcp.sh
#   sbatch slurm/diagrams_merge_lgcp.sh            # all 8 (process, filtration)
#   sbatch --array=0-1 slurm/diagrams_merge_lgcp.sh   # just dtm_k5, both processes

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

# One thread per task -- GUDHI's DTM backend is internally multi-threaded.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

FILTRATIONS=(dtm_k5 dtm_k10 dtm_k15 rips)   # k5 first -- see header
PROCESSES=(lgcp lgcp_strauss)
NSHARDS=120                                 # MUST match slurm/diagrams_merge_lgcp.sh

NPROC=${#PROCESSES[@]}
PER_FILT=$(( NPROC * NSHARDS ))                     # 240
NTASKS=$(( ${#FILTRATIONS[@]} * PER_FILT ))         # 960 -- must equal the --array span

t="${SLURM_ARRAY_TASK_ID:?run this as an array job (sbatch slurm/diagrams_compute_lgcp.sh)}"
if (( t >= NTASKS )); then
  echo "task $t >= $NTASKS -- nothing to do (check --array span vs NSHARDS)"; exit 0
fi
filt="${FILTRATIONS[$(( t / PER_FILT ))]}"
rem=$(( t % PER_FILT ))
proc="${PROCESSES[$(( rem / NSHARDS ))]}"
shard=$(( rem % NSHARDS ))

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID:-unset}  Array task ${t}  ->  process=${proc}  filtration=${filt}  shard ${shard}/${NSHARDS}"

# Preamble: size of the largest cloud this task will see, per split, using
# the SAME contiguous partition diagrams_shard.py uses. Separate process, so
# the loaded clouds are freed before any persistence starts.
PROC_ENV="$proc" SHARD_ENV="$shard" NSHARDS_ENV="$NSHARDS" python - <<'PYEOF'
import os, pickle, sys
sys.path.insert(0, "scripts/processing")
from diagrams_shard import shard_bounds
proc, shard, total = os.environ["PROC_ENV"], int(os.environ["SHARD_ENV"]), int(os.environ["NSHARDS_ENV"])
for split in ("clouds", "adversarial_clouds"):
    with open(f"data/{proc}/{split}.pkl", "rb") as fh:
        clouds = pickle.load(fh)
    lo, hi = shard_bounds(len(clouds), shard, total)
    sizes = [int(c.get("n_points", len(c["points"]))) for c in clouds[lo:hi]]
    big = sum(s > 1300 for s in sizes)
    print(f"  [{split}] shard {shard}: {hi - lo} clouds, largest {max(sizes) if sizes else 0} pts, "
          f"{big} above 1300 pts" + ("   <-- high-memory shard" if sizes and max(sizes) > 1300 else ""))
PYEOF

python -u scripts/processing/diagrams_shard.py compute \
  --process "$proc" \
  --filtration "$filt" \
  --n-shards "$NSHARDS" \
  --shard-index "$shard"
