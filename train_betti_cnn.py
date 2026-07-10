"""Trains betti_cnn_0 and betti_cnn_1 (Betti curves + Conv1D encoder + n(x),
see src/cloudforger/nn/experiments/betti_cnn.py) across all 10 canonical
seeds. Run directly and watch progress in your terminal:

    python train_betti_cnn.py

Results land in results/params/2d/thomas/betti_cnn_<dim>/seed_<seed>/,
same layout as every other method, so models/evaluate_seeds.py picks it up
by adding "betti_cnn_0 betti_cnn_1" to its --features list.

Already-finished (seed, dim) pairs are skipped, so if you stop this partway
through (Ctrl-C) you can just re-run it to pick up where it left off.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.nn.experiments import build_experiment

SEEDS = [
    56245, 189089, 2342344, 9278394, 91873097,
    908308920, 235498734453, 928374129038471,
    974924729845723, 9267492783429472,
]

DATA_DIR = ROOT / "data" / "params" / "2d" / "thomas"
RESULTS_DIR = ROOT / "results" / "params" / "2d" / "thomas"

CFG_BASE = {
    "task": "params", "process": "thomas",
    "batch_size": 32, "n_epochs": 500, "lr": 0.001, "embedding_dim": 128,
}

for hom_dim in (0, 1):
    method = f"betti_cnn_{hom_dim}"
    for seed in SEEDS:
        output_dir = RESULTS_DIR / method / f"seed_{seed}"

        if (output_dir / "results.pt").exists():
            print(f"\n{method} | seed {seed}: already done, skipping.")
            continue

        print(f"\n{'=' * 80}\n{method} | seed {seed}\n{'=' * 80}")
        cfg = {**CFG_BASE, "method": method, "seed": seed}
        experiment = build_experiment(cfg)
        experiment.run(
            DATA_DIR / "betti.pkl",
            output_dir,
            adversarial_path=DATA_DIR / "adversarial_betti.pkl",
        )

print("\nAll seeds done. Compare with:")
print(
    "  python models/evaluate_seeds.py --models-root results/params/2d/thomas "
    "--features betti_0 betti_1 betti_cnn_0 betti_cnn_1"
)
