# Cluster settings for every SLURM job: the one file to edit on another cluster.
#
# Sourced from the repo root by slurm/job.sh (activates the environment) and by slurm/run_all.sh
# (passes the *_SBATCH flags to sbatch). Each can also be overridden from the calling shell.
#
# GPU access here: the `compute` partition has no GPUs, and `gpu` only admits other accounts, so
# GPU jobs go to `sae` under pilot_sae_gpu, which is not the default account and must be named.
# `computeshort` shares `compute`'s nodes with a 1-hour limit and a much shorter queue.

CLOUDFORGER_ENV="${CLOUDFORGER_ENV:-/gpfs/scratch/qp252676/globus/envs/cloud-env}"
CPU_SBATCH="${CPU_SBATCH:--p compute}"
SHORT_SBATCH="${SHORT_SBATCH:--p computeshort}"
GPU_SBATCH="${GPU_SBATCH:--A pilot_sae_gpu -p sae --gres=gpu:1}"

# Threads per process: THREADS if the job sets it (1 for jobs that parallelize over processes),
# else every CPU of the job. Read before the module load, which resets OMP_NUM_THREADS to 1.
threads="${THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

mkdir -p logs
module load miniforge
set +u                                               # conda's activate scripts read unset variables
mamba activate "$CLOUDFORGER_ENV"
set -u

export OMP_NUM_THREADS="$threads"
