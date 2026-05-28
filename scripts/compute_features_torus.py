# scripts/compute_features_torus.py
import pickle
from pathlib import Path

from cloudforger.stats.pair_dist import PairDistanceCDF
from cloudforger.core.features import CorrelationFeatures

DATA_DIR = Path("data")

# The statistics to compute. Add TripleAngleCDF here later — the rest
# of the script picks it up with no other changes.
STATISTICS = [
    PairDistanceCDF(n_samples=5000, grid_size=64),
]


def main():
    with open(DATA_DIR / "clouds_torus.pkl", "rb") as f:
        bundle = pickle.load(f)

    clouds = bundle["clouds"]

    features = []
    for cloud in clouds:
        # One vector per statistic, keyed by the statistic's name.
        vectors = {stat.name: stat.compute(cloud) for stat in STATISTICS}
        features.append(CorrelationFeatures(
            features=vectors,
            generator_name=cloud.generator_name,
            generator_params=cloud.generator_params,
            seed=cloud.seed,
            statistic_params={stat.name: stat.params for stat in STATISTICS},
        ))

    with open(DATA_DIR / "features_torus.pkl", "wb") as f:
        pickle.dump({
            "features": features,
            "labels": bundle["labels"],
            "label_names": bundle["label_names"],
            "statistic_params": {s.name: s.params for s in STATISTICS},
        }, f)

    print(f"Computed features for {len(features)} clouds "
          f"({[s.name for s in STATISTICS]}) -> {DATA_DIR / 'features_torus.pkl'}")


if __name__ == "__main__":
    main()