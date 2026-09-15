#!/bin/bash

#SBATCH -J nb_generation
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH --output=logs/nb_generation_%j.out
#SBATCH --error=logs/nb_generation_%j.err

# Headless smoke test of notebooks/cloud_generation_walkthrough.ipynb:
# executes every code cell in order in one namespace (cloud-env has no
# nbconvert/ipykernel), times each cell, stops at the first failure naming
# the cell, and saves every figure the notebook would show to
#   notebooks/out/generation_smoke/fig_NN.png
#
# Submit from the point-process-tda repo root:
#   sbatch slurm/notebook_generation_smoke.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg
mkdir -p logs

module load miniforge
set +u
mamba activate /gpfs/scratch/qp252676/globus/envs/cloud-env
set -u

python - <<'EOF'
import json, time, traceback
from pathlib import Path
import matplotlib.pyplot as plt

nb = json.loads(Path("notebooks/cloud_generation_walkthrough.ipynb").read_text())
out = Path("notebooks/out/generation_smoke"); out.mkdir(parents=True, exist_ok=True)
n_fig = [0]

def save_show(*args, **kwargs):
    for num in plt.get_fignums():
        n_fig[0] += 1
        plt.figure(num).savefig(out / f"fig_{n_fig[0]:02d}.png", dpi=80, bbox_inches="tight")
    plt.close("all")

plt.show = save_show
ns = {"__name__": "__main__"}
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
t_all = time.time()
for k, cell in enumerate(code_cells):
    src = "".join(cell["source"])
    t0 = time.time()
    print(f"\n===== cell {k + 1}/{len(code_cells)}: {src.strip().splitlines()[0][:80]!r}", flush=True)
    try:
        exec(compile(src, f"<cell {k + 1}>", "exec"), ns)
    except Exception:
        traceback.print_exc()
        raise SystemExit(f"FAILED at code cell {k + 1}")
    print(f"----- cell {k + 1} ok ({time.time() - t0:.1f}s, figures so far: {n_fig[0]})", flush=True)
print(f"\nALL {len(code_cells)} CODE CELLS OK in {time.time() - t_all:.0f}s; {n_fig[0]} figures -> {out}")
EOF
