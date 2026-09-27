#!/bin/bash
# Every unfinished unit of a learned-summary config as one GPU array, then the summary.
#
#   bash extensions/learned/submit.sh                                   # smoke.yaml
#   CONFIG=extensions/learned/<name>.yaml bash extensions/learned/submit.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p extensions/logs
export CONFIG="${CONFIG:-extensions/learned/smoke.yaml}" PYTHONDONTWRITEBYTECODE=1
PY="${PY:-/gpfs/scratch/qp252676/globus/envs/cloud-env/bin/python}"
idx=$($PY extensions/learned/learned.py --config "$CONFIG" list | grep -v "(done)" | awk '{print $1}' | paste -sd, -)
if [[ -z "$idx" ]]; then echo "nothing to train"; exit 0; fi
id=$(sbatch --parsable --export=ALL --array="$idx" extensions/learned/unit.sh)
echo "units [$idx]: $id"
sid=$(sbatch --parsable --export=ALL --dependency=afterany:$id -J ext_learned_sum -p compute -c 1 --mem=4G -t 00:10:00 \
      --output=extensions/logs/learned_summary_%j.out \
      --wrap "module load miniforge; mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env; python extensions/learned/learned.py --config $CONFIG summarize")
echo "summary: $sid"
