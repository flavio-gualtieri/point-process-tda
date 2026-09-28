# Cluster settings for every SLURM job: the one file to edit on another cluster.
#
# Sourced from the repo root by every worker in slurm/ (activates the environment) and by
# slurm/run_all.sh (passes CPU_SBATCH / GPU_SBATCH to sbatch, where they override the workers'
# #SBATCH defaults). Each can also be overridden from the calling shell.
#
# GPU access here: the `compute` partition has no GPUs, and `gpu` only admits other accounts, so
# GPU jobs go to `sae` under pilot_sae_gpu, which is not the default account and must be named.

CLOUDFORGER_ENV="${CLOUDFORGER_ENV:-/gpfs/scratch/qp252676/globus/envs/cloud-env}"
CPU_SBATCH="${CPU_SBATCH:--p compute}"
GPU_SBATCH="${GPU_SBATCH:--A pilot_sae_gpu -p sae --gres=gpu:1}"

mkdir -p logs
module load miniforge
set +u                                               # conda's activate scripts read unset variables
mamba activate "$CLOUDFORGER_ENV"
set -u

# The module sets 1, which serializes the tree learners; jobs that parallelize over processes
# instead export OMP_NUM_THREADS=1 after sourcing this.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
