# scripts/runners/params/train_new_pi.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cloudforger.nn.experiments import build_experiment

""" SEEDS = [
    56245, 189089, 2342344, 9278394, 91873097,
    908308920, 235498734453, 928374129038471,
    974924729845723, 9267492783429472,
] """

SEEDS = [
    56245,
]

DATA_DIR = ROOT / "data" / "params" / "2d" / "thomas"
RESULTS_DIR = ROOT / "results" / "params" / "2d" / "thomas"

CFG_BASE = {
    "task": "params", "process": "thomas",
    "batch_size": 32, "n_epochs": 500, "lr": 0.001, "embedding_dim": 128,
}
hom_dim = 0

method = f"pi_{hom_dim}"
for seed in SEEDS:
    output_dir = RESULTS_DIR / method / f"seed_{seed}"

    if (output_dir / "results.pt").exists():
        print(f"\n{method} | seed {seed}: already done, skipping.")
        continue

    print(f"\n{'=' * 80}\n{method} | seed {seed}\n{'=' * 80}")
    cfg = {**CFG_BASE, "method": method, "seed": seed}
    experiment = build_experiment(cfg)
    experiment.run(
        DATA_DIR / "images.pkl",
        output_dir,
        adversarial_path=DATA_DIR / "adversarial_images.pkl",
    )

print("\nAll seeds done. Compare with:")