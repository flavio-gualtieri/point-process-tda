# dtm_experiment/validate_nested_thomas_data.py
"""
Sanity-checks the full nested_thomas pipeline output (all 3 stages) --
prints a structured report and flags anything that looks wrong, rather than
just trusting that "the jobs finished with exit code 0" means the data is
actually usable. Run AFTER the full pipeline (generate -> diagrams ->
compute_features) has completed; see run_validate_nested_thomas_data.sh.

Checks, per stage:
  1. clouds.pkl / adversarial_clouds.pkl -- counts, point-count distribution,
     realized parameter ranges vs. the manifest's declared ranges, points
     inside the unit box, seed uniqueness, no train_test/adversarial overlap.
  2. diagrams_dtm_k5.pkl / adversarial_* -- counts (and drop rate vs. stage
     1, from the ValueError-skip fix + any pre-existing empty-H0 clouds),
     filtration_params match what nested_thomas_diagrams.yaml asked for,
     seeds are a subset of stage 1's, no empty-H0 diagrams slipped through
     un-flagged.
  3. betti_dtm_k5.pkl / betti_weighted_dtm_k5.pkl / images_dtm_k5.pkl
     (+ adversarial_*) -- shapes, NaN/Inf, label consistency across the 3
     files, persistence entropy sanity, seeds a subset of stage 2's.

Exits non-zero if any check fails outright (missing file, shape mismatch,
NaN/Inf); prints "! " warnings (does not fail) for things worth a human
look (dropped-cloud rate, realized ranges narrower than declared, etc).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "processing" / "params"))

from pipeline_lib.io import load_pickle, load_yaml_config
from pipeline_lib.records import load_diagrams

from compute_features import _is_broken  # reuse the exact same empty-H0 check compute_features.py uses

DATA_DIR = ROOT / "data" / "params" / "2d" / "nested_thomas"
MANIFEST_PATH = ROOT / "configs" / "params" / "processing" / "nested_thomas_cloudgen.yaml"

FAILURES: list[str] = []
WARNINGS: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL: {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  ! {msg}")


def _require(path: Path) -> bool:
    if not path.exists():
        fail(f"missing file: {path}")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: clouds
# ══════════════════════════════════════════════════════════════════════════════

def check_clouds() -> dict[str, Any]:
    print(f"\n{'=' * 88}\n  Stage 1: clouds\n{'=' * 88}")

    manifest_cfg = load_yaml_config(MANIFEST_PATH)
    declared_ranges = manifest_cfg["design"]["random"]["ranges"]

    result: dict[str, Any] = {}
    for split, fname in [("train_test", "clouds.pkl"), ("adversarial", "adversarial_clouds.pkl")]:
        path = DATA_DIR / fname
        if not _require(path):
            result[split] = None
            continue

        clouds = load_pickle(path)
        n = len(clouds)
        points_counts = np.array([len(c["points"]) for c in clouds])
        seeds = np.array([c["seed"] for c in clouds])

        print(f"\n  [{split}] {n} clouds, from {path.name}")
        print(f"    n_points: min={points_counts.min()} median={int(np.median(points_counts))} "
              f"mean={points_counts.mean():.1f} max={points_counts.max()} "
              f"p95={np.percentile(points_counts, 95):.0f}")

        if points_counts.min() == 0:
            fail(f"[{split}] {int((points_counts == 0).sum())} clouds have 0 points")
        if (points_counts < 5).any():
            warn(f"[{split}] {int((points_counts < 5).sum())}/{n} clouds have <5 points "
                 "(below DTM k=5's minimum -- these should show up as skipped in stage 2's logs)")

        if len(np.unique(seeds)) != n:
            fail(f"[{split}] {n - len(np.unique(seeds))} duplicate seeds")

        # Manifest range-spec keys != cloud.params keys for one param:
        # NestedThomasProcess exposes meta_parent_intensity under the
        # "parent_intensity" key (the root Poisson layer's own name, see
        # its docstring) -- everything else matches 1:1.
        manifest_to_params_key = {"meta_parent_intensity": "parent_intensity"}
        params_by_name = {
            name: np.array([c["params"][manifest_to_params_key.get(name, name)] for c in clouds])
            for name in declared_ranges
        }
        for name, arr in params_by_name.items():
            lo_decl, hi_decl = declared_ranges[name]["low"], declared_ranges[name]["high"]
            lo_real, hi_real = arr.min(), arr.max()
            if lo_real < lo_decl * 0.999 or hi_real > hi_decl * 1.001:
                fail(f"[{split}] {name} realized range [{lo_real:.4g}, {hi_real:.4g}] "
                     f"escapes declared range [{lo_decl}, {hi_decl}]")

        all_points = np.concatenate([c["points"] for c in clouds if len(c["points"])], axis=0)
        if all_points.size and (not np.all((all_points >= 0) & (all_points <= 1))):
            fail(f"[{split}] some points fall outside the unit box [0,1]^2")

        result[split] = {"n": n, "seeds": set(seeds.tolist()), "point_counts": points_counts}

    if result.get("train_test") and result.get("adversarial"):
        overlap = result["train_test"]["seeds"] & result["adversarial"]["seeds"]
        if overlap:
            fail(f"{len(overlap)} seeds appear in BOTH train_test and adversarial clouds")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: diagrams
# ══════════════════════════════════════════════════════════════════════════════

def check_diagrams(cloud_info: dict[str, Any]) -> dict[str, Any]:
    print(f"\n{'=' * 88}\n  Stage 2: DTM diagrams\n{'=' * 88}")

    result: dict[str, Any] = {}
    for split, fname in [("train_test", "diagrams_dtm_k5.pkl"), ("adversarial", "adversarial_diagrams_dtm_k5.pkl")]:
        path = DATA_DIR / fname
        if not _require(path):
            result[split] = None
            continue

        diagrams, bundle = load_diagrams(path)
        n = len(diagrams)
        seeds = set(int(s) for s in bundle["seeds"])

        print(f"\n  [{split}] {n} diagrams, from {path.name}")
        fp = bundle.get("filtration_params", {})
        print(f"    filtration_params: {fp}")
        if fp.get("k") != 5:
            warn(f"[{split}] filtration k={fp.get('k')}, expected 5")

        for key, expected_len in [("labels", n), ("seeds", n), ("params", n)]:
            actual_len = len(bundle[key])
            if actual_len != expected_len:
                fail(f"[{split}] {key} length {actual_len} != n_diagrams {n} (index misalignment)")

        n_broken = sum(1 for d in diagrams if _is_broken(d))
        if n_broken:
            warn(f"[{split}] {n_broken}/{n} diagrams have empty H0 -- these will be dropped by "
                 "compute_features.py's own filter, not a stage-2 bug, just confirming the count.")

        if cloud_info.get(split):
            n_clouds = cloud_info[split]["n"]
            dropped = n_clouds - n
            pct = 100 * dropped / n_clouds if n_clouds else 0
            print(f"    {dropped}/{n_clouds} clouds from stage 1 did not make it into diagrams "
                  f"({pct:.1f}%, expected from the k=5-minimum-points skip fix)")
            if pct > 5:
                warn(f"[{split}] {pct:.1f}% of clouds were dropped between stage 1 and stage 2 -- "
                     "higher than expected, worth checking the compute logs for skip messages.")
            if not seeds <= cloud_info[split]["seeds"]:
                fail(f"[{split}] diagram seeds are not a subset of stage-1 cloud seeds")

        result[split] = {"n": n, "seeds": seeds}

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: betti / images
# ══════════════════════════════════════════════════════════════════════════════

def _check_array(name: str, arr: np.ndarray, expected_shape_tail: tuple[int, ...]) -> None:
    if arr.shape[1:] != expected_shape_tail:
        fail(f"{name} shape {arr.shape} -- expected (n, {', '.join(map(str, expected_shape_tail))})")
    if not np.all(np.isfinite(arr)):
        fail(f"{name} contains NaN/Inf")


def check_features(diagram_info: dict[str, Any]) -> None:
    print(f"\n{'=' * 88}\n  Stage 3: betti / persistence images / entropy\n{'=' * 88}")

    for split, suffix in [("train_test", ""), ("adversarial", "adversarial_")]:
        betti_path = DATA_DIR / f"{suffix}betti_dtm_k5.pkl"
        betti_w_path = DATA_DIR / f"{suffix}betti_weighted_dtm_k5.pkl"
        images_path = DATA_DIR / f"{suffix}images_dtm_k5.pkl"

        if not all(_require(p) for p in (betti_path, betti_w_path, images_path)):
            continue

        betti = load_pickle(betti_path)
        betti_w = load_pickle(betti_w_path)
        images = load_pickle(images_path)

        n = len(betti["seeds"])
        print(f"\n  [{split}] {n} feature rows, from {betti_path.name}/{images_path.name}")

        for label, payload in [("betti", betti), ("betti_weighted", betti_w), ("images", images)]:
            if len(payload["seeds"]) != n:
                fail(f"[{split}] {label} has {len(payload['seeds'])} seeds, expected {n} (mismatch vs. betti file)")
            if list(payload["seeds"]) != list(betti["seeds"]):
                fail(f"[{split}] {label}'s seed order doesn't match betti's -- these files must be row-aligned")

        _check_array(f"[{split}] betti0_matrix", np.asarray(betti["betti0_matrix"]), (512,))
        _check_array(f"[{split}] betti1_matrix", np.asarray(betti["betti1_matrix"]), (512,))
        _check_array(f"[{split}] images pi_0", np.asarray(images["image_tensors"][0]), (128, 128))
        _check_array(f"[{split}] images pi_1", np.asarray(images["image_tensors"][1]), (128, 128))

        for dim in (0, 1):
            entropy = np.asarray(betti["persistence_entropy"][dim])
            if entropy.shape != (n,):
                fail(f"[{split}] persistence_entropy[{dim}] shape {entropy.shape} != ({n},)")
            elif (entropy < 0).any() or not np.all(np.isfinite(entropy)):
                fail(f"[{split}] persistence_entropy[{dim}] has negative/non-finite values")

        label_names = betti["label_names"]
        if images["label_names"] != label_names:
            fail(f"[{split}] betti/images label_names disagree: {label_names} vs {images['label_names']}")
        expected_labels = {"parent_intensity", "meta_offspring", "meta_cluster_scale", "mean_offspring", "cluster_scale"}
        if not expected_labels <= set(label_names):
            fail(f"[{split}] label_names {label_names} missing expected nested_thomas params {expected_labels}")

        seeds = set(int(s) for s in betti["seeds"])
        if diagram_info.get(split) and not seeds <= diagram_info[split]["seeds"]:
            fail(f"[{split}] betti/images seeds are not a subset of stage-2 diagram seeds")


# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"Validating nested_thomas data under {DATA_DIR}")

    cloud_info = check_clouds()
    diagram_info = check_diagrams(cloud_info)
    check_features(diagram_info)

    print(f"\n{'=' * 88}")
    if FAILURES:
        print(f"  RESULT: {len(FAILURES)} FAILURE(S), {len(WARNINGS)} warning(s) -- see above")
        print(f"{'=' * 88}")
        sys.exit(1)
    else:
        print(f"  RESULT: all checks passed ({len(WARNINGS)} warning(s) worth a look)")
        print(f"{'=' * 88}")


if __name__ == "__main__":
    main()
