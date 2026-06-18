# scripts/processing/compute_features.py
"""Stage 2 (statistics branch): clouds -> correlation features.
Reads {dim}d/clouds.pkl, writes {dim}d/features.pkl for each ambient dimension.
Rerun if a statistic's config (n_samples, grid_size) changes."""

import pickle
from pathlib import Path
from cloudforger.stats.pair_dist import PairDistanceCDF
from cloudforger.core.features import CorrelationFeatures

CONFIG = {
    "dims": list(range(2, 18)),
    "data_dir": Path("data"),
}

# The statistics to compute. Add TripleAngleCDF here later — the rest
# of the script picks it up with no other changes.
STATISTICS = [
    PairDistanceCDF(n_samples=5000, grid_size=64),
]

def compute_features_for_dim(dim: int, data_dir: Path) -> None:
    clouds_path = data_dir / f"{dim}d" / "clouds.pkl"
    out_path    = data_dir / f"{dim}d" / "features.pkl"

    with open(clouds_path, "rb") as f:
        bundle = pickle.load(f)

    clouds   = bundle["clouds"]
    features = []

    print(f"  Computing features for dim={dim} ({len(clouds)} clouds) ...")
    for i, cloud in enumerate(clouds):
        vectors = {stat.name: stat.compute(cloud) for stat in STATISTICS}
        features.append(CorrelationFeatures(
            features=vectors,
            generator_name=cloud.generator_name,
            generator_params=cloud.generator_params,
            seed=cloud.seed,
            statistic_params={stat.name: stat.params for stat in STATISTICS},
        ))
        print(f"\r    {i + 1}/{len(clouds)}", end="", flush=True)
    print()

    with open(out_path, "wb") as f:
        pickle.dump({
            "features":        features,
            "labels":          bundle["labels"],
            "label_names":     bundle["label_names"],
            "statistic_params": {s.name: s.params for s in STATISTICS},
            "config":          bundle["config"],   # carry through from clouds
        }, f)

    print(f"  Saved features for {len(features)} clouds "
          f"({[s.name for s in STATISTICS]}) → {out_path}")

def main():
    for dim in CONFIG["dims"]:
        compute_features_for_dim(dim, CONFIG["data_dir"])
    print("\nDone.")

if __name__ == "__main__":
    main()