#!/bin/bash
# Submits the full writeup calibration-sweep block for SIL-1/BC/LS + the PI
# confirmation pass, both processes (thomas, nested_thomas), all FUSED DTM
# k=5,10,15 -- results land under the fresh results/ tree (results_exp/ is
# the old, renamed-aside run). See each slurm/*.sh below for its own
# grid/results-path details. NOT run automatically by anything -- run it
# yourself from the point-process-tda repo root:
#   bash slurm/submit_writeup_calib_sweep.sh
# or submit the sbatch files individually, in any order (no job
# dependencies between them -- they only share already-generated
# data/{thomas,nested_thomas}/... inputs, never each other's results).
#
# 16 sbatch scripts total:
#   SIL-1 G-sweep      (G in {64,128,256} x 3 seeds, x2 processes)
#   BC    G-sweep      (grid_size in {64,128,256} x 3 seeds, x2 processes)
#   BC    q-sweep      (pd_calibration_coverage in {0.95,0.99,1.00} x 3 seeds, x2 processes)
#   Inherited-q fairness check (LS/SIL-1/BC x q in {0.95,0.99} x 3 seeds, x2 processes)
#   PI confirmation pass (encoder_mode in {shared,independent}, fusion_mode=concat,
#                          sigma_pixels=0.5 (the pi_multik calib sweep's chosen value)
#                          x 5 seeds, x2 processes)
#
# ASSUMES data/{thomas,nested_thomas}/clouds.pkl and
# data/{thomas,nested_thomas}/dtm_k{5,10,15}/diagrams.pkl already exist for
# both processes -- see each script's own header for the one-off
# generate.py/featurize.py commands if they don't.

set -euo pipefail
cd "$(dirname "$0")/.."

# SIL-1 grid-resolution sweep
sbatch slurm/vec_multik_silhouette_calib_G_thomas.sh
sbatch slurm/vec_multik_silhouette_calib_G_nested_thomas.sh

# BC grid-resolution sweep
sbatch slurm/betti_multik_calib_G_thomas.sh
sbatch slurm/betti_multik_calib_G_nested_thomas.sh

# BC calibration-coverage sweep
sbatch slurm/betti_multik_calib_q_thomas.sh
sbatch slurm/betti_multik_calib_q_nested_thomas.sh

# Inherited-q fairness check (LS, SIL-1, BC)
sbatch slurm/vec_multik_landscape_fairq_thomas.sh
sbatch slurm/vec_multik_landscape_fairq_nested_thomas.sh
sbatch slurm/vec_multik_silhouette_fairq_thomas.sh
sbatch slurm/vec_multik_silhouette_fairq_nested_thomas.sh
sbatch slurm/betti_multik_fairq_thomas.sh
sbatch slurm/betti_multik_fairq_nested_thomas.sh

# PI confirmation pass (shared+concat, independent+concat)
sbatch slurm/pi_multik_confirm_shared_thomas.sh
sbatch slurm/pi_multik_confirm_independent_thomas.sh
sbatch slurm/pi_multik_confirm_shared_nested_thomas.sh
sbatch slurm/pi_multik_confirm_independent_nested_thomas.sh
