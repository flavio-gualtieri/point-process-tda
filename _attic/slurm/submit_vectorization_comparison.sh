#!/bin/bash
# Submits the "alternative vectorizations" experiment grid for the
# writeup's Vectorization Comparisons section (ssec:exp-summaries): native
# (shared+concat) arm for every vectorization, the architecture-agnostic
# MLP control for every vectorization (+ STAT, MLP-only by construction),
# a homology-dimension ablation restricted to LS and BC (PI's own H0/H1
# arms already exist from the calibrated-experiments batch -- see
# pi_multik_h0only_*.sh/pi_multik_h1only_*.sh), and the silhouette
# channel ablation. thomas and nested_thomas both, 5 seeds each, at the
# calibration each vectorization's own config already carries (PI:
# resolution=64/sigma_pixels=0.5/coverage=0.95; landscape/silhouette:
# G=128/coverage=0.95; betti: grid_size=128/coverage=0.95, range_pad=1.1).
#
# Reuses, rather than duplicates, whatever native-path infra already
# existed before this batch (all 10-seed, a superset of the 5 requested --
# submit with --array=0-4 appended if you want to cap them):
#   PI native:  pi_multik_headline_thomas.sh / pi_multik_baseline_complete.sh
#   LS native:  vec_multik_landscape_thomas.sh / _nested_thomas.sh
#   SIL-0..3 native: vec_multik_silhouette_{thomas,nested_thomas}[_p0/_p2/_p3].sh
#   BC native:  betti_multik_thomas.sh / _nested_thomas.sh (already
#               include_euler=false, i.e. already "BC" as defined for this
#               comparison -- see thomas_betti_multik_k5k10k15.yaml's header)
#   PI mlp:     vec_multik_pi_mlp_thomas.sh / _nested_thomas.sh (previous batch)
# None of those 12 are submitted below -- only the genuinely new arms are.
#
# 28 GPU array jobs below. NOT run automatically by anything -- run it
# yourself from the point-process-tda repo root:
#   bash slurm/submit_vectorization_comparison.sh
# or submit the sbatch files individually.

set -euo pipefail
cd "$(dirname "$0")/.."

# --- Native (shared+concat): only EC is new (PI/LS/SIL-0..3/BC above already exist) ---
sbatch slurm/betti_multik_euler_only_thomas.sh
sbatch slurm/betti_multik_euler_only_nested_thomas.sh

# --- Architecture-agnostic MLP control (PI already exists, see above) ---
sbatch slurm/vec_multik_landscape_mlp_thomas.sh
sbatch slurm/vec_multik_landscape_mlp_nested_thomas.sh
sbatch slurm/vec_multik_silhouette_mlp_p0_thomas.sh
sbatch slurm/vec_multik_silhouette_mlp_p0_nested_thomas.sh
sbatch slurm/vec_multik_silhouette_mlp_p1_thomas.sh
sbatch slurm/vec_multik_silhouette_mlp_p1_nested_thomas.sh
sbatch slurm/vec_multik_silhouette_mlp_p2_thomas.sh
sbatch slurm/vec_multik_silhouette_mlp_p2_nested_thomas.sh
sbatch slurm/vec_multik_silhouette_mlp_p3_thomas.sh
sbatch slurm/vec_multik_silhouette_mlp_p3_nested_thomas.sh
sbatch slurm/vec_multik_betti_mlp_bc_thomas.sh
sbatch slurm/vec_multik_betti_mlp_bc_nested_thomas.sh
sbatch slurm/vec_multik_betti_mlp_ec_thomas.sh
sbatch slurm/vec_multik_betti_mlp_ec_nested_thomas.sh
sbatch slurm/vec_multik_persistence_statistics_mlp_thomas.sh
sbatch slurm/vec_multik_persistence_statistics_mlp_nested_thomas.sh

# --- Homology-dimension ablation: LS + BC only (PI's own H0/H1 already exist) ---
sbatch slurm/vec_multik_landscape_h0only_thomas.sh
sbatch slurm/vec_multik_landscape_h0only_nested_thomas.sh
sbatch slurm/vec_multik_landscape_h1only_thomas.sh
sbatch slurm/vec_multik_landscape_h1only_nested_thomas.sh
sbatch slurm/betti_multik_h0only_thomas.sh
sbatch slurm/betti_multik_h0only_nested_thomas.sh
sbatch slurm/betti_multik_h1only_thomas.sh
sbatch slurm/betti_multik_h1only_nested_thomas.sh

# --- Silhouette channel ablation: SIL-1, normalized channel only ---
sbatch slurm/vec_multik_silhouette_normonly_thomas.sh
sbatch slurm/vec_multik_silhouette_normonly_nested_thomas.sh
