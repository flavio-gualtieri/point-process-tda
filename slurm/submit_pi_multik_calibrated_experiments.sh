#!/bin/bash
# Submits the full "newly-calibrated" pi_multik experiment grid requested
# for the writeup's Experiments section (headline + every ablation/control
# arm), thomas and nested_thomas both, at the calibration sweep's chosen
# defaults (resolution=64, sigma_pixels=0.5, pd_calibration_coverage=0.95,
# pad=1.05 -- see configs/runs/{thomas,nested_thomas}/thomas_pi_multik_k5k10k15.yaml
# and pi_multik.yaml, both already updated).
#
# Covers every arm EXCEPT:
#   - [PI | H0+H1 | DTM{5} / DTM{10} / DTM{15} | 5] (single-scale) -- already
#     covered by the existing, 10-seed pi_multik_singlescale_{thomas,
#     nested_thomas}_k{5,10,15}.sh scripts (a superset of the requested 5
#     seeds; their configs already picked up the new calibration
#     automatically). Submit those 6 directly, or with --array=0-4 appended
#     at submission time to cap at 5 seeds.
#   - [PI | H0+H1 | Rips | 5] -- BLOCKED. plain Rips gives every H0 feature
#     birth=0, which build_calibrated_imager's axis_bounds still rejects
#     outright ("H0 birth axis is degenerate; use a 1-D vectorizer" --
#     src/cloudforger/calibration/diagram_calibration.py). The writeup's
#     SS4.4 already describes a "fall back to a 1-D vectorization" resolution
#     for this case, but that fallback is NOT implemented in code -- it's
#     aspirational prose only. Submitting the existing
#     pi_multik_rips_{thomas,nested_thomas}.sh scripts as-is will just
#     reproduce that crash on every task (that's their documented, current
#     behavior). Needs a decision (implement the 1-D fallback -- a real
#     architecture change, since pi_multik's multi-channel tensor assumes
#     every channel is the same G x G raster -- vs. run H1-only vs. accept
#     the crash as the result) before a working Rips arm can be submitted.
#   - [... | encoder_mode x fusion_mode, 6-way] on nested_thomas -- already
#     covered by the existing pi_multik_sweep_{shared,independent}_{concat,
#     conv_avg,conv_flatten}.sh scripts (run-tag encsweep_*), whose config
#     (pi_multik.yaml) already picked up the new calibration automatically.
#
# 21 GPU array jobs below. NOT run automatically by anything -- run it
# yourself from the point-process-tda repo root:
#   bash slurm/submit_pi_multik_calibrated_experiments.sh
# or submit the sbatch files individually.

set -euo pipefail
cd "$(dirname "$0")/.."

# Headline: [PI | H0+H1 | DTM{5,10,15} | 10]
sbatch slurm/pi_multik_headline_thomas.sh
# nested_thomas headline: sbatch slurm/pi_multik_baseline_complete.sh (pre-existing)

# Homology-dimension ablation: [PI | H0 or H1 | DTM{5,10,15} | 5]
sbatch slurm/pi_multik_h0only_thomas.sh
sbatch slurm/pi_multik_h0only_nested_thomas.sh
sbatch slurm/pi_multik_h1only_thomas.sh
sbatch slurm/pi_multik_h1only_nested_thomas.sh

# CoordConv ablation (SS6.3): [... | CoordConv channels removed]
sbatch slurm/pi_multik_coordconv_removed_thomas.sh
sbatch slurm/pi_multik_coordconv_removed_nested_thomas.sh

# encoder_mode x fusion_mode 6-way, thomas (nested_thomas: pre-existing, see above)
sbatch slurm/pi_multik_sweep_shared_concat_thomas.sh
sbatch slurm/pi_multik_sweep_shared_conv_avg_thomas.sh
sbatch slurm/pi_multik_sweep_shared_conv_flatten_thomas.sh
sbatch slurm/pi_multik_sweep_independent_concat_thomas.sh
sbatch slurm/pi_multik_sweep_independent_conv_avg_thomas.sh
sbatch slurm/pi_multik_sweep_independent_conv_flatten_thomas.sh

# Architecture-agnostic control: [... | encoder_path = mlp]
sbatch slurm/vec_multik_pi_mlp_thomas.sh
sbatch slurm/vec_multik_pi_mlp_nested_thomas.sh

# log N(x) ablation + its matched "log N only" baseline -- submit both of
# each pair together, never one without the other
sbatch slurm/pi_multik_logn_removed_thomas.sh
sbatch slurm/pi_multik_logn_removed_nested_thomas.sh
sbatch slurm/logn_only_thomas.sh
sbatch slurm/logn_only_nested_thomas.sh

# Leakage sanity check: [... | shuffled labels], expect L~=1
sbatch slurm/pi_multik_shuffled_labels_thomas.sh
sbatch slurm/pi_multik_shuffled_labels_nested_thomas.sh
