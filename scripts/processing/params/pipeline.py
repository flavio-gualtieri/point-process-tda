# scripts/processing/params/pipeline.py

from __future__ import annotations

import argparse
import copy
import inspect
import itertools
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import yaml

# scripts/processing/params/<this file> -> parents[3] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cloudforger.core.base import PointProcess
from cloudforger.core.betti import BettiCurveFeature
from cloudforger.core.cloud import PointCloud
from cloudforger.core.diagram import PersistenceDiagram
from cloudforger.core.features import CorrelationFeatures
from cloudforger.core.region import Box, Region
from cloudforger.processes.matern import MaternHardCoreProcess
from cloudforger.processes.poisson import PoissonProcess
from cloudforger.processes.thomas import ThomasProcess
from cloudforger.stats.pair_dist import PairDistanceCDF
from cloudforger.tda.calibration import calibrate, calibrate_report
from cloudforger.tda.features.betti_curve import BettiCurve
from cloudforger.tda.filtration.rips import RipsFiltration
from cloudforger.tda.vectorizer.multi_channel import MultiChannelImager
from cloudforger.tda.vectorizer.persistence_image import PersistenceImager


# ---------------------------------------------------------------------------
# Execution mode
# ---------------------------------------------------------------------------
# Controls how the dimension list is dispatched:
#   "auto"  -> run as a SLURM array (one dimension per task) iff the env var
#              SLURM_ARRAY_TASK_ID is present; otherwise run the full list
#              serially. This is rsync-safe: the same file behaves correctly
#              on the cluster and locally with no edits.
#   "hpc"   -> require SLURM_ARRAY_TASK_ID and run exactly one dimension.
#   "local" -> always run the full dimension list serially, ignoring SLURM.
RUN_MODE = "local"


DEFAULT_STAGE_ORDER = (
    "generate_clouds",
    "compute_pairwise",
    "compute_diagrams",
    "vectorize_diagrams",
    "compute_betti",
)

DEFAULT_SPLITS: dict[str, dict[str, str]] = {
    "train_test": {
        "clouds": "clouds.pkl",
        "diagrams": "diagrams.pkl",
        "images": "images.pkl",
        "betti": "betti.pkl",
        "features": "features.pkl",
    },
    "adversarial": {
        "clouds": "adversarial_clouds.pkl",
        "diagrams": "adversarial_diagrams.pkl",
        "images": "adversarial_images.pkl",
        "betti": "adversarial_betti.pkl",
        "features": "adversarial_features.pkl",
    },
}

PIPELINE_CONFIG: dict[str, Any] = {
    "process": "",
    "dimensions": [2],
    "data_root": "data/params",
    "skip_missing": True,
    "overwrite": True,
    "stages": list(DEFAULT_STAGE_ORDER),
    "splits": DEFAULT_SPLITS,
    "formats": {
        "clouds": "dict",
        "diagrams": "dict",
        "betti": "dict",
        "features": "dict",
    },
    "cloud_generation": {
        "config_path": "configs/params/thomas_cloudgen.yaml",
    },
    "filtration": {
        "maxdim": 1,
        "thresh": None,
    },
    "vectorization": {
        "homology_dims": [0, 1],
        "resolution": 64,
        "sigma": 0.05,
        "calibration_split": "train_test",
    },
    "betti": {
        "homology_dims": (0, 1),
        "grid_size": 128,
        "grid_range": (0.0, 1.0),
        "drop_infinite": True,
        "normalize": False,
    },
    "pair_distance": {
        "n_samples": 5_000,
        "grid_size": 64,
    },
}

PROCESS_REGISTRY: dict[str, type[PointProcess]] = {
    "thomas": ThomasProcess,
    "matern": MaternHardCoreProcess,
    "poisson": PoissonProcess,
}


@dataclass(frozen=True)
class CloudDesignBundle:
    config: dict[str, Any]
    train_test_vectors: list[dict[str, float]]
    adversarial_vectors: list[dict[str, float]]
    adversarial_indices: list[int]
    train_test_design: list[dict[str, float]]
    adversarial_design: list[dict[str, float]]
    base_seed: int
    output_format: str
    n_hint: int
    adversarial_seed_offset: int
    dimension_seed_stride: int


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_int_list(value: int | Sequence[int], name: str) -> list[int]:
    if isinstance(value, int):
        return [value]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = list(value)
        if all(isinstance(v, int) for v in result):
            return result

    raise TypeError(f"CONFIG[{name!r}] must be an int or a list of ints.")


def require_format(value: str, name: str) -> str:
    if value not in {"dict", "object"}:
        raise ValueError(f"{name} must be 'dict' or 'object'.")
    return value


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_items(path: Path, key: str) -> list[Any]:
    data = load_pickle(path)
    if isinstance(data, dict) and key in data:
        return data[key]
    return data


def build_labels(
    items: list[Any],
    params_fn: Callable[[Any], dict[str, Any]],
    item_name: str,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    if not items:
        raise ValueError(f"Cannot build labels from an empty {item_name} list.")

    params = [params_fn(item) for item in items]
    label_names = list(params[0].keys())

    for p in params:
        if list(p.keys()) != label_names:
            raise ValueError(
                f"{item_name.capitalize()} have inconsistent parameter keys; "
                "cannot stack labels."
            )

    labels = np.array([[p[k] for k in label_names] for p in params], dtype=float)
    return labels, label_names, params


# ---------------------------------------------------------------------------
# Point-cloud generation
# ---------------------------------------------------------------------------


def load_yaml_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return config


def default_region(dimension: int) -> Region:
    return Box(low=np.zeros(dimension), high=np.ones(dimension))


def expand_grid_axis(spec: Any) -> list[float]:
    if isinstance(spec, list):
        return [float(v) for v in spec]

    if not isinstance(spec, dict):
        raise ValueError(f"Grid axis must be a list or mapping, got {spec!r}.")

    if "values" in spec:
        return [float(v) for v in spec["values"]]

    kind = spec.get("kind")
    low = float(spec["low"])
    high = float(spec["high"])
    num = int(spec["num"])
    if num < 1:
        raise ValueError(f"Grid axis num must be >= 1, got {num}.")

    if kind == "linspace":
        return [float(v) for v in np.linspace(low, high, num)]
    if kind == "logspace":
        if low <= 0 or high <= 0:
            raise ValueError("logspace grid bounds must be positive.")
        return [float(v) for v in np.exp(np.linspace(np.log(low), np.log(high), num))]

    raise ValueError(f"Unknown grid kind {kind!r}; use values, linspace, or logspace.")


def iter_param_grid(grid: dict[str, Any]) -> Iterator[dict[str, float]]:
    keys = list(grid)
    axes = [expand_grid_axis(grid[k]) for k in keys]
    for combo in itertools.product(*axes):
        yield dict(zip(keys, combo))


def sample_param_vector(
    ranges: dict[str, dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, float]:
    params: dict[str, float] = {}

    for key, spec in ranges.items():
        low = float(spec["low"])
        high = float(spec["high"])
        scale = spec.get("scale", "linear")

        if scale == "log":
            if low <= 0 or high <= 0:
                raise ValueError(f"Log-scaled parameter {key!r} must have positive bounds.")
            value = np.exp(rng.uniform(np.log(low), np.log(high)))
        elif scale == "linear":
            value = rng.uniform(low, high)
        else:
            raise ValueError(f"Unknown scale {scale!r} for parameter {key!r}.")

        params[key] = float(value)

    return params


def build_param_vectors(
    config: dict[str, Any],
    design_rng: np.random.Generator,
) -> list[dict[str, float]]:
    design = config["design"]
    mode = design.get("mode", "grid")

    if mode == "grid":
        return list(iter_param_grid(design["grid"]))

    if mode == "random":
        random_cfg = design["random"]
        n_param_vectors = int(random_cfg["n_param_vectors"])
        ranges = random_cfg["ranges"]
        return [sample_param_vector(ranges, design_rng) for _ in range(n_param_vectors)]

    raise ValueError(f"Unknown design.mode {mode!r}; use 'grid' or 'random'.")


def split_adversarial_vectors(
    param_vectors: list[dict[str, float]],
    config: dict[str, Any],
    base_seed: int,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[int]]:
    adv_cfg = config.get("adversarial", {})
    if not adv_cfg.get("enabled", False):
        return param_vectors, [], []

    n_total = len(param_vectors)
    exact_n = adv_cfg.get("n_param_vectors")
    if exact_n is None:
        fraction = float(adv_cfg.get("fraction", 0.0))
        n_adv = int(round(fraction * n_total))
    else:
        n_adv = int(exact_n)

    if n_adv < 0 or n_adv >= n_total:
        raise ValueError(
            f"adversarial holdout size must be in [0, {n_total - 1}], got {n_adv}."
        )

    seed_offset = int(adv_cfg.get("seed_offset", 100_000))
    holdout_rng = np.random.default_rng(base_seed + seed_offset)
    adversarial_idx = set(int(i) for i in holdout_rng.choice(n_total, size=n_adv, replace=False))

    train_test_vectors = [p for i, p in enumerate(param_vectors) if i not in adversarial_idx]
    adversarial_vectors = [p for i, p in enumerate(param_vectors) if i in adversarial_idx]
    return train_test_vectors, adversarial_vectors, sorted(adversarial_idx)


def repeat_vectors(param_vectors: Iterable[dict[str, float]], reps: int) -> list[dict[str, float]]:
    return [dict(params) for params in param_vectors for _ in range(reps)]


def cloud_to_record(cloud: PointCloud) -> dict[str, Any]:
    record: dict[str, Any] = {
        "points": cloud.points,
        "params": dict(cloud.generator_params),
        "process": cloud.generator_name,
        "seed": cloud.seed,
        "n_points": cloud.n_points,
        "dimension": cloud.dimension,
    }

    if isinstance(cloud.region, Box):
        record["region"] = {"low": cloud.region.low, "high": cloud.region.high}

    return record


def save_clouds(path: Path, clouds: list[PointCloud], output_format: str) -> None:
    payload = [cloud_to_record(c) for c in clouds] if output_format == "dict" else clouds
    dump_pickle(path, payload)


def cloud_stats(clouds: list[PointCloud]) -> dict[str, int]:
    if not clouds:
        return {"n_clouds": 0, "total_points": 0, "min_points": 0, "max_points": 0}
    return {
        "n_clouds": len(clouds),
        "total_points": int(sum(c.n_points for c in clouds)),
        "min_points": int(min(c.n_points for c in clouds)),
        "max_points": int(max(c.n_points for c in clouds)),
    }


def generate_clouds_for_design(
    process_name: str,
    region: Region,
    design: list[dict[str, float]],
    base_seed: int,
    n_hint: int = 0,
) -> list[PointCloud]:
    if process_name not in PROCESS_REGISTRY:
        raise ValueError(f"Unknown process {process_name!r}. Available: {sorted(PROCESS_REGISTRY)}")

    cls = PROCESS_REGISTRY[process_name]
    valid_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    clouds: list[PointCloud] = []

    for offset, params in enumerate(design):
        filtered = {k: v for k, v in params.items() if k in valid_params}
        clouds.append(cls(**filtered).sample(n=n_hint, region=region, seed=base_seed + offset))

    return clouds


def write_cloud_manifest(
    path: Path,
    config: dict[str, Any],
    train_test_vectors: list[dict[str, float]],
    adversarial_vectors: list[dict[str, float]],
    adversarial_indices: list[int],
    train_test_clouds: list[PointCloud],
    adversarial_clouds: list[PointCloud],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": config,
        "n_train_test_param_vectors": len(train_test_vectors),
        "n_adversarial_param_vectors": len(adversarial_vectors),
        "adversarial_param_indices": adversarial_indices,
        "adversarial_params": adversarial_vectors,
        "train_test_stats": cloud_stats(train_test_clouds),
        "adversarial_stats": cloud_stats(adversarial_clouds),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Shared cloud/diagram adapters
# ---------------------------------------------------------------------------


def to_pointcloud(cloud: Any) -> PointCloud:
    if not isinstance(cloud, dict):
        return cloud

    region = None
    reg = cloud.get("region")
    if reg is not None:
        try:
            region = Box(
                low=np.asarray(reg["low"], dtype=float),
                high=np.asarray(reg["high"], dtype=float),
            )
        except Exception:
            region = None

    return PointCloud(
        points=np.asarray(cloud["points"]),
        generator_name=cloud.get("process", ""),
        generator_params=dict(cloud.get("params", {})),
        seed=cloud.get("seed"),
        region=region,
    )


def cloud_params(cloud: Any) -> dict[str, Any]:
    return dict(cloud["params"]) if isinstance(cloud, dict) else dict(cloud.generator_params)


def cloud_seed(cloud: Any) -> Any:
    return cloud.get("seed") if isinstance(cloud, dict) else cloud.seed


def diagram_to_record(diagram: PersistenceDiagram) -> dict[str, Any]:
    return {
        "diagrams": {int(dim): np.asarray(pairs) for dim, pairs in diagram.diagrams.items()},
        "params": dict(diagram.generator_params),
        "seed": diagram.seed,
        "process": diagram.generator_name,
        "filtration": diagram.filtration_name,
        "filtration_params": dict(diagram.filtration_params),
    }


def record_to_diagram(record: dict[str, Any]) -> PersistenceDiagram:
    return PersistenceDiagram(
        diagrams={int(dim): np.asarray(pairs, dtype=float) for dim, pairs in record["diagrams"].items()},
        generator_name=record.get("process", ""),
        generator_params=dict(record.get("params", {})),
        seed=record.get("seed"),
        filtration_name=record.get("filtration", ""),
        filtration_params=dict(record.get("filtration_params", {})),
    )


def as_diagram(value: Any) -> PersistenceDiagram:
    if isinstance(value, PersistenceDiagram):
        return value
    if isinstance(value, dict):
        return record_to_diagram(value)
    raise TypeError(f"Expected PersistenceDiagram or dict, got {type(value)}.")


def diagram_params(diagram: Any) -> dict[str, Any]:
    return dict(diagram["params"]) if isinstance(diagram, dict) else dict(diagram.generator_params)


def diagram_seed(diagram: Any) -> Any:
    return diagram.get("seed") if isinstance(diagram, dict) else diagram.seed


def load_diagram_bundle(path: Path) -> dict[str, Any]:
    data = load_pickle(path)
    if isinstance(data, dict) and "diagrams" in data:
        return data
    return {"diagrams": data}


def load_diagrams(path: Path) -> tuple[list[PersistenceDiagram], dict[str, Any]]:
    bundle = load_diagram_bundle(path)
    return [as_diagram(d) for d in bundle["diagrams"]], bundle


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------


def build_filtration(config: dict[str, Any]) -> RipsFiltration:
    filtration_cfg = config["filtration"]
    return RipsFiltration(
        maxdim=int(filtration_cfg.get("maxdim", 1)),
        thresh=filtration_cfg.get("thresh", None),
    )


def compute_diagrams_from_clouds(
    clouds_path: Path,
    out_path: Path,
    filtration: RipsFiltration,
    process: str,
    diagram_format: str,
) -> None:
    clouds = load_items(clouds_path, "clouds")
    labels, label_names, params = build_labels(clouds, cloud_params, "clouds")
    seeds = [cloud_seed(c) for c in clouds]

    computed: list[PersistenceDiagram] = []
    print(f"  Computing {len(clouds)} diagrams from {clouds_path} ...")

    for i, cloud in enumerate(clouds):
        computed.append(filtration.compute(to_pointcloud(cloud)))
        print(f"\r    {i + 1}/{len(clouds)}", end="", flush=True)
    print()

    diagrams = [diagram_to_record(d) for d in computed] if diagram_format == "dict" else computed
    dump_pickle(
        out_path,
        {
            "diagrams": diagrams,
            "labels": labels,
            "label_names": label_names,
            "params": params,
            "seeds": seeds,
            "process": process,
            "filtration_params": filtration.params,
        },
    )
    print(f"  Saved {len(diagrams)} diagrams → {out_path}")


# ---------------------------------------------------------------------------
# Persistence images
# ---------------------------------------------------------------------------


def build_imagers(
    diagrams: list[PersistenceDiagram],
    homology_dims: list[int],
    resolution: int,
    sigma: float,
) -> MultiChannelImager:
    print(calibrate_report(diagrams))
    stats = calibrate(diagrams)
    imagers: dict[int, PersistenceImager] = {}

    for dim in homology_dims:
        axes = stats.get(dim)
        if axes is None:
            birth_hi = 1.0
            persistence_hi = 1.0
        else:
            birth_hi = axes["birth"].get(99.0, 1.0)
            persistence_hi = axes["persistence"].get(99.0, 1.0)
            birth_hi = 1.0 if birth_hi <= 0 else birth_hi
            persistence_hi = 1.0 if persistence_hi <= 0 else persistence_hi

        imagers[dim] = PersistenceImager(
            birth_range=(0.0, float(birth_hi)),
            pers_range=(0.0, float(persistence_hi)),
            resolution=int(resolution),
            sigma=float(sigma),
        )

    return MultiChannelImager(imagers)


def stack_images_by_dim(
    images: list[dict[int, np.ndarray]],
    homology_dims: list[int],
    resolution: int,
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for dim in homology_dims:
        if images:
            result[dim] = np.stack([img[dim] for img in images])
        else:
            result[dim] = np.empty((0, resolution, resolution), dtype=float)
    return result


def vectorize_diagram_split(
    diagrams_path: Path,
    out_path: Path,
    imager: MultiChannelImager,
    homology_dims: list[int],
    resolution: int,
    dim: int,
    split_name: str,
) -> None:
    diagrams, bundle = load_diagrams(diagrams_path)
    print(f"  Vectorizing {split_name} ({len(diagrams)} diagrams) from {diagrams_path} ...")

    images = [imager.transform(d) for d in diagrams]
    image_tensors = stack_images_by_dim(images, homology_dims, resolution)

    payload: dict[str, Any] = {
        "images": images,
        "image_tensors": image_tensors,
        "homology_dims": imager.dimensions,
        "imager_params": imager.params,
        "dimension": dim,
        "split": split_name,
        "source_diagrams": str(diagrams_path),
    }

    for key in ("labels", "label_names", "params", "seeds", "process", "filtration_params"):
        if key in bundle:
            payload[key] = bundle[key]

    dump_pickle(out_path, payload)
    shapes = {k: v.shape for k, v in image_tensors.items()}
    print(f"  Saved {len(images)} images → {out_path}")
    print(f"  Image tensors by dimension: {shapes}")


# ---------------------------------------------------------------------------
# Betti curves
# ---------------------------------------------------------------------------


def build_betti(config: dict[str, Any]) -> BettiCurve:
    betti_cfg = config["betti"]
    return BettiCurve(
        homology_dims=betti_cfg.get("homology_dims", (0, 1)),
        grid_size=betti_cfg.get("grid_size", 128),
        grid_range=betti_cfg.get("grid_range", (0.0, 1.0)),
        drop_infinite=betti_cfg.get("drop_infinite", True),
        normalize=betti_cfg.get("normalize", False),
    )


def betti_feature_to_record(feature: BettiCurveFeature) -> dict[str, Any]:
    return {
        "curves": {int(dim): np.asarray(curve) for dim, curve in feature.curves.items()},
        "vector": feature.vector(),
        "params": dict(feature.generator_params),
        "seed": feature.seed,
        "process": feature.generator_name,
        "filtration": feature.filtration_name,
        "filtration_params": dict(feature.filtration_params),
        "feature": feature.feature_name,
        "feature_params": dict(feature.feature_params),
    }


def compute_betti_from_diagrams(
    diagrams_path: Path,
    out_path: Path,
    betti: BettiCurve,
    process: str,
    curve_format: str,
) -> None:
    diagrams = load_items(diagrams_path, "diagrams")
    labels, label_names, params = build_labels(diagrams, diagram_params, "diagrams")
    seeds = [diagram_seed(d) for d in diagrams]

    computed: list[BettiCurveFeature] = []
    print(f"  Computing {len(diagrams)} Betti curves from {diagrams_path} ...")

    for i, d in enumerate(diagrams):
        computed.append(betti.compute(as_diagram(d)))
        print(f"\r    {i + 1}/{len(diagrams)}", end="", flush=True)
    print()

    curves = [betti_feature_to_record(c) for c in computed] if curve_format == "dict" else computed
    homology_dims = list(getattr(betti, "homology_dims", betti.params.get("homology_dims", (0, 1))))
    betti_matrices = {int(dim): np.stack([feature.curves[dim] for feature in computed]) for dim in homology_dims}

    payload: dict[str, Any] = {
        "curves": curves,
        "betti_matrices": betti_matrices,
        "labels": labels,
        "label_names": label_names,
        "params": params,
        "seeds": seeds,
        "process": process,
        "betti_params": betti.params,
    }

    # Keep the historical convenience keys for downstream code that expects them.
    if 0 in betti_matrices:
        payload["betti0_matrix"] = betti_matrices[0]
    if 1 in betti_matrices:
        payload["betti1_matrix"] = betti_matrices[1]

    dump_pickle(out_path, payload)
    print(f"  Saved {len(curves)} Betti curves → {out_path}")


# ---------------------------------------------------------------------------
# Pairwise/correlation features
# ---------------------------------------------------------------------------


def features_to_record(cf: CorrelationFeatures) -> dict[str, Any]:
    return {
        "features": {k: np.asarray(v) for k, v in cf.features.items()},
        "params": dict(cf.generator_params),
        "seed": cf.seed,
        "process": cf.generator_name,
        "statistic_params": dict(cf.statistic_params),
    }


def compute_pairwise_features(
    clouds_path: Path,
    out_path: Path,
    statistics: list[Any],
    process: str,
    feature_format: str,
) -> None:
    clouds = load_items(clouds_path, "clouds")
    labels, label_names, params = build_labels(clouds, cloud_params, "clouds")
    seeds = [cloud_seed(c) for c in clouds]
    stat_params = {s.name: s.params for s in statistics}
    order = sorted(s.name for s in statistics)

    computed: list[CorrelationFeatures] = []
    rows: list[np.ndarray] = []

    print(f"  Computing pairwise features for {len(clouds)} clouds {[s.name for s in statistics]} from {clouds_path} ...")
    for i, cloud in enumerate(clouds):
        pc = to_pointcloud(cloud)
        vectors = {s.name: np.asarray(s.compute(pc)) for s in statistics}
        rows.append(np.concatenate([vectors[name] for name in order]))
        computed.append(
            CorrelationFeatures(
                features=vectors,
                generator_name=pc.generator_name,
                generator_params=pc.generator_params,
                seed=pc.seed,
                statistic_params=stat_params,
            )
        )
        print(f"\r    {i + 1}/{len(clouds)}", end="", flush=True)
    print()

    feature_matrix = np.stack(rows)
    features = [features_to_record(cf) for cf in computed] if feature_format == "dict" else computed

    dump_pickle(
        out_path,
        {
            "features": features,
            "feature_matrix": feature_matrix,
            "feature_order": order,
            "labels": labels,
            "label_names": label_names,
            "params": params,
            "seeds": seeds,
            "process": process,
            "statistic_params": stat_params,
        },
    )
    print(f"  Saved pairwise features for {len(computed)} clouds, feature_matrix {feature_matrix.shape} → {out_path}")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


class ParamsPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config)
        self.process = str(self.config.get("process", ""))
        if not self.process:
            _cloud_cfg_path = resolve_path(
                self.config.get("cloud_generation", {}).get(
                    "config_path", "configs/params/thomas_cloudgen.yaml"
                )
            )
            if _cloud_cfg_path.exists():
                self.process = str(load_yaml_config(_cloud_cfg_path).get("process", ""))
                self.config["process"] = self.process
        self.dimensions = normalize_int_list(self.config.get("dimensions", [2]), "dimensions")
        # dimension_indices lets a single-dimension run (e.g. one SLURM array
        # task) keep its ORIGINAL position in the full list, so that the seed
        # `base_seed + dim_index * dimension_seed_stride` matches a serial run.
        dim_indices = self.config.get("dimension_indices")
        if dim_indices is None:
            self.dim_indices = list(range(len(self.dimensions)))
        else:
            self.dim_indices = normalize_int_list(dim_indices, "dimension_indices")
            if len(self.dim_indices) != len(self.dimensions):
                raise ValueError(
                    "CONFIG['dimension_indices'] must have the same length as "
                    "CONFIG['dimensions']."
                )
        self.data_root = resolve_path(self.config.get("data_root", "data/params"))
        self.splits = self.config.get("splits", DEFAULT_SPLITS)
        self.skip_missing = bool(self.config.get("skip_missing", True))
        self.overwrite = bool(self.config.get("overwrite", True))
        self.stages = tuple(self.config.get("stages", DEFAULT_STAGE_ORDER))
        self._validate()

    def _validate(self) -> None:
        unknown = set(self.stages) - set(DEFAULT_STAGE_ORDER)
        if unknown:
            raise ValueError(f"Unknown stage(s): {sorted(unknown)}")

        formats = self.config.get("formats", {})
        for key in ("clouds", "diagrams", "betti", "features"):
            require_format(formats.get(key, "dict"), f"formats.{key}")

        for split_name, split_cfg in self.splits.items():
            missing = {"clouds", "diagrams", "images", "betti", "features"} - set(split_cfg)
            if missing:
                raise ValueError(f"Split {split_name!r} is missing file role(s): {sorted(missing)}")

    def base_dir(self, dim: int) -> Path:
        return self.data_root / f"{dim}d" / self.process

    def split_path(self, dim: int, split_name: str, role: str) -> Path:
        return self.base_dir(dim) / self.splits[split_name][role]

    def should_skip_missing(self, path: Path, message: str) -> bool:
        if path.exists():
            return False
        if self.skip_missing:
            print(f"  Skipping: {message}: {path}")
            return True
        raise FileNotFoundError(f"{message}: {path}")

    def should_skip_existing(self, path: Path, message: str) -> bool:
        if self.overwrite or not path.exists():
            return False
        print(f"  Skipping: {message}; output exists: {path}")
        return True

    def prepare_cloud_design(self) -> CloudDesignBundle:
        cloud_cfg_path = resolve_path(self.config["cloud_generation"].get("config_path", "configs/params/thomas_cloudgen.yaml"))
        cloud_cfg = load_yaml_config(cloud_cfg_path)

        self.process = self.config["process"] = str(cloud_cfg.get("process", self.process))
        cloud_cfg["dimension"] = list(self.dimensions)

        output_format = require_format(
            self.config.get("formats", {}).get("clouds", cloud_cfg.get("format", "dict")),
            "formats.clouds",
        )
        cloud_cfg["format"] = output_format

        base_seed = int(cloud_cfg.get("seed", 0))
        n_hint = int(cloud_cfg.get("n_hint", 0))
        reps = int(cloud_cfg["design"].get("reps", 1))
        if reps < 1:
            raise ValueError("cloud_generation design.reps must be >= 1.")

        design_rng = np.random.default_rng(base_seed)
        param_vectors = build_param_vectors(cloud_cfg, design_rng)
        train_test_vectors, adversarial_vectors, adversarial_indices = split_adversarial_vectors(
            param_vectors, cloud_cfg, base_seed
        )

        train_test_design = repeat_vectors(train_test_vectors, reps)
        adversarial_design = repeat_vectors(adversarial_vectors, reps)
        adversarial_seed_offset = int(cloud_cfg.get("adversarial", {}).get("seed_offset", 100_000))
        dimension_seed_stride = int(cloud_cfg.get("dimension_seed_stride", 1_000_937_000))

        return CloudDesignBundle(
            config=cloud_cfg,
            train_test_vectors=train_test_vectors,
            adversarial_vectors=adversarial_vectors,
            adversarial_indices=adversarial_indices,
            train_test_design=train_test_design,
            adversarial_design=adversarial_design,
            base_seed=base_seed,
            output_format=output_format,
            n_hint=n_hint,
            adversarial_seed_offset=adversarial_seed_offset,
            dimension_seed_stride=dimension_seed_stride,
        )

    def generate_clouds_for_dim(self, dim_index: int, dim: int, design: CloudDesignBundle) -> None:
        dim_seed = design.base_seed + dim_index * design.dimension_seed_stride
        region = default_region(dim)

        train_test_output = self.split_path(dim, "train_test", "clouds")
        adversarial_output = self.split_path(dim, "adversarial", "clouds")
        manifest_output = self.base_dir(dim) / "cloud_generation_manifest.yaml"

        outputs = [train_test_output, adversarial_output, manifest_output]
        if all(path.exists() for path in outputs) and not self.overwrite:
            print(f"  Skipping cloud generation; outputs exist in {self.base_dir(dim)}")
            return

        train_test_clouds = generate_clouds_for_design(
            process_name=self.process,
            region=region,
            design=design.train_test_design,
            base_seed=dim_seed,
            n_hint=design.n_hint,
        )
        adversarial_clouds = generate_clouds_for_design(
            process_name=self.process,
            region=region,
            design=design.adversarial_design,
            base_seed=dim_seed + design.adversarial_seed_offset,
            n_hint=design.n_hint,
        )

        save_clouds(train_test_output, train_test_clouds, design.output_format)
        save_clouds(adversarial_output, adversarial_clouds, design.output_format)

        manifest_config = dict(design.config)
        manifest_config["dimension"] = dim
        manifest_config["dimension_seed"] = dim_seed
        write_cloud_manifest(
            manifest_output,
            manifest_config,
            design.train_test_vectors,
            design.adversarial_vectors,
            design.adversarial_indices,
            train_test_clouds,
            adversarial_clouds,
        )

        train_stats = cloud_stats(train_test_clouds)
        adv_stats = cloud_stats(adversarial_clouds)
        reps = int(design.config["design"].get("reps", 1))

        print(
            f"  Generated {train_stats['n_clouds']} train/test clouds from "
            f"{len(design.train_test_vectors)} parameter vectors x {reps} rep(s)."
        )
        print(
            f"  Generated {adv_stats['n_clouds']} adversarial clouds from "
            f"{len(design.adversarial_vectors)} held-out parameter vectors x {reps} rep(s)."
        )
        print(
            f"  Train/test points: total {train_stats['total_points']}, "
            f"min {train_stats['min_points']}, max {train_stats['max_points']}."
        )
        print(
            f"  Adversarial points: total {adv_stats['total_points']}, "
            f"min {adv_stats['min_points']}, max {adv_stats['max_points']}."
        )
        print(f"  Saved clouds and manifest in {self.base_dir(dim)}")

    def compute_diagrams_for_dim(self, dim: int) -> None:
        filtration = build_filtration(self.config)
        diagram_format = require_format(self.config.get("formats", {}).get("diagrams", "dict"), "formats.diagrams")

        for split_name in self.splits:
            clouds_path = self.split_path(dim, split_name, "clouds")
            out_path = self.split_path(dim, split_name, "diagrams")
            if self.should_skip_missing(clouds_path, f"[dim={dim}] Missing {split_name} clouds"):
                continue
            if self.should_skip_existing(out_path, f"[dim={dim}] {split_name} diagrams"):
                continue
            print(f"  Split: {split_name}")
            compute_diagrams_from_clouds(clouds_path, out_path, filtration, self.process, diagram_format)

    def vectorize_diagrams_for_dim(self, dim: int) -> None:
        vec_cfg = self.config["vectorization"]
        homology_dims = normalize_int_list(vec_cfg.get("homology_dims", [0, 1]), "vectorization.homology_dims")
        resolution = int(vec_cfg.get("resolution", 64))
        sigma = float(vec_cfg.get("sigma", 0.05))
        calibration_split = str(vec_cfg.get("calibration_split", "train_test"))

        calibration_path = self.split_path(dim, calibration_split, "diagrams")
        if self.should_skip_missing(calibration_path, f"[dim={dim}] Missing calibration diagrams"):
            return

        calibration_diagrams, _ = load_diagrams(calibration_path)
        imager = build_imagers(calibration_diagrams, homology_dims, resolution, sigma)

        for split_name in self.splits:
            diagrams_path = self.split_path(dim, split_name, "diagrams")
            out_path = self.split_path(dim, split_name, "images")
            if self.should_skip_missing(diagrams_path, f"[dim={dim}] Missing {split_name} diagrams"):
                continue
            if self.should_skip_existing(out_path, f"[dim={dim}] {split_name} images"):
                continue
            vectorize_diagram_split(diagrams_path, out_path, imager, homology_dims, resolution, dim, split_name)

    def compute_betti_for_dim(self, dim: int) -> None:
        betti = build_betti(self.config)
        curve_format = require_format(self.config.get("formats", {}).get("betti", "dict"), "formats.betti")

        for split_name in self.splits:
            diagrams_path = self.split_path(dim, split_name, "diagrams")
            out_path = self.split_path(dim, split_name, "betti")
            if self.should_skip_missing(diagrams_path, f"[dim={dim}] Missing {split_name} diagrams"):
                continue
            if self.should_skip_existing(out_path, f"[dim={dim}] {split_name} Betti curves"):
                continue
            compute_betti_from_diagrams(diagrams_path, out_path, betti, self.process, curve_format)

    def compute_pairwise_for_dim(self, dim: int) -> None:
        feature_format = require_format(self.config.get("formats", {}).get("features", "dict"), "formats.features")
        pair_cfg = self.config["pair_distance"]
        statistics = [
            PairDistanceCDF(
                n_samples=int(pair_cfg.get("n_samples", 5_000)),
                grid_size=int(pair_cfg.get("grid_size", 64)),
            )
        ]

        for split_name in self.splits:
            clouds_path = self.split_path(dim, split_name, "clouds")
            out_path = self.split_path(dim, split_name, "features")
            if self.should_skip_missing(clouds_path, f"[dim={dim}] Missing {split_name} clouds"):
                continue
            if self.should_skip_existing(out_path, f"[dim={dim}] {split_name} pairwise features"):
                continue
            compute_pairwise_features(clouds_path, out_path, statistics, self.process, feature_format)

    def run(self) -> None:
        cloud_design = self.prepare_cloud_design() if "generate_clouds" in self.stages else None

        for dim_index, dim in zip(self.dim_indices, self.dimensions):
            print(f"\n[dim={dim}] Base directory: {self.base_dir(dim)}")

            if "generate_clouds" in self.stages:
                assert cloud_design is not None
                print("\n  Stage: generate_clouds")
                self.generate_clouds_for_dim(dim_index, dim, cloud_design)

            if "compute_diagrams" in self.stages:
                print("\n  Stage: compute_diagrams")
                self.compute_diagrams_for_dim(dim)

            if "vectorize_diagrams" in self.stages:
                print("\n  Stage: vectorize_diagrams")
                self.vectorize_diagrams_for_dim(dim)

            if "compute_betti" in self.stages:
                print("\n  Stage: compute_betti")
                self.compute_betti_for_dim(dim)

            if "compute_pairwise" in self.stages:
                print("\n  Stage: compute_pairwise")
                self.compute_pairwise_for_dim(dim)

        print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the params processing pipeline.")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        help="Dimensions to run, e.g. --dimensions 2 3 5. Overrides PIPELINE_CONFIG.",
    )
    parser.add_argument("--process", help="Process name, e.g. thomas. Overrides PIPELINE_CONFIG.")
    parser.add_argument(
        "--cloud-config",
        help="Path to cloud_generation.yaml. Overrides PIPELINE_CONFIG['cloud_generation']['config_path'].",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=DEFAULT_STAGE_ORDER,
        help="Subset of stages to run in the order provided.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip stage outputs that already exist.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on missing split inputs instead of skipping them.",
    )
    return parser.parse_args()


def resolve_execution(config: dict[str, Any]) -> None:
    """Restrict this process to a single dimension when running as a SLURM
    array task, while preserving that dimension's original index so the
    per-dimension seed matches a local serial run.

    The mapping is positional: SLURM_ARRAY_TASK_ID == i selects
    config["dimensions"][i]. Keep --array=0-(N-1) in sync with the length of
    the dimension list, and do not reorder the list between runs.
    """
    if RUN_MODE == "local":
        return

    task_id_env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_env is None:
        if RUN_MODE == "hpc":
            raise RuntimeError(
                "RUN_MODE='hpc' but SLURM_ARRAY_TASK_ID is not set. Launch via "
                "`sbatch --array=...`, or use RUN_MODE='auto'/'local' to run locally."
            )
        return  # auto + not in an array -> full serial run

    task_id = int(task_id_env)
    full_dims = normalize_int_list(config.get("dimensions", [2]), "dimensions")
    if not 0 <= task_id < len(full_dims):
        raise IndexError(
            f"SLURM_ARRAY_TASK_ID={task_id} is out of range for "
            f"{len(full_dims)} dimension(s) {full_dims}. Use --array=0-{len(full_dims) - 1}."
        )

    config["dimensions"] = [full_dims[task_id]]
    config["dimension_indices"] = [task_id]
    print(
        f"[SLURM] array task {task_id} -> dimension {full_dims[task_id]} "
        f"(seed index {task_id} of {len(full_dims)})"
    )


def main() -> None:
    args = parse_args()
    config = copy.deepcopy(PIPELINE_CONFIG)

    if args.dimensions is not None:
        config["dimensions"] = args.dimensions
    if args.process is not None:
        config["process"] = args.process
    if args.cloud_config is not None:
        config.setdefault("cloud_generation", {})["config_path"] = args.cloud_config
    if args.stages is not None:
        config["stages"] = args.stages
    if args.no_overwrite:
        config["overwrite"] = False
    if args.strict:
        config["skip_missing"] = False

    resolve_execution(config)

    ParamsPipeline(config).run()


if __name__ == "__main__":
    main()