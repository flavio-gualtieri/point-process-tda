#!/usr/bin/env python
"""Simulations backing docs/theory/notes.tex. Submit with theory_sims.sh (SLURM);
do not run on the login node.

Everything uses the repo's own code, unmodified: the point-process classes in
cloudforger.data_generation.point_processes, the isotropic L(r)-r estimator in
cloudforger.baselines.vihrs, the border-corrected F/G in
cloudforger.baselines.summstats, and the Rips / DTM filtration classes.
Raw arrays go to docs/theory/scripts/out/<part>.npz; make_figures.py reads them
(plotting only).

Parts (run in this order; `data` needs `null`):
  null      binomial (CSR | n) null of L-r, F, G on the repo grids
  data      departure statistics S_L, S_G, S_F for the current datasets (from the
            existing L and F/G caches; nothing is recomputed from clouds)
  kfun      closed-form K vs simulators: Thomas x3, Matern cluster, nested x2,
            LGCP, Matern II (thinning run on the dilated window)
  strauss   default MH budget vs 10x, and the Poisson-saddlepoint intensity
  matern2   repo Matern II simulator as-is: edge-band intensity excess
  examples  one realization per process at E[N] = 400
  ph        Rips / DTM diagrams of one Thomas and one CSR cloud + sanity checks
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from cloudforger.core.region import Box  # noqa: E402
from cloudforger.data_generation.point_processes.poisson import PoissonProcess  # noqa: E402
from cloudforger.data_generation.point_processes.thomas import ThomasProcess  # noqa: E402
from cloudforger.data_generation.point_processes.matern_cluster import MaternClusterProcess  # noqa: E402
from cloudforger.data_generation.point_processes.nested_thomas import NestedThomasProcess  # noqa: E402
from cloudforger.data_generation.point_processes.aniso_thomas import AnisotropicThomasProcess  # noqa: E402
from cloudforger.data_generation.point_processes.matern import MaternHardCoreProcess  # noqa: E402
from cloudforger.data_generation.point_processes.gibbs import StraussProcess, _adaptive_steps  # noqa: E402
from cloudforger.data_generation.point_processes.cox import LGCPProcess  # noqa: E402
from cloudforger.baselines.vihrs import _isotropic_l_minus_r, default_r_grid  # noqa: E402
from cloudforger.baselines import summstats  # noqa: E402

import departure as dep  # noqa: E402

OUT = HERE / "out"
UNIT = Box(np.zeros(2), np.ones(2))
LOW, HIGH = [0.0, 0.0], [1.0, 1.0]
R_GRID = default_r_grid()                 # linspace(0, 0.25, 513), the vihrs L grid
FG_GRID = np.linspace(0.0, 0.25, 513)     # the summ_fg0250 F/G grid (asserted in `data`)
LAMBDA = 400.0                            # every kfun setting has intensity 400 on [0,1]^2


# ------------------------------------------------------------------ parametrizations
# The repo's design axes (configs/runs/*/): K = parent intensity, EN = expected
# count, c = 2 sigma sqrt(K) (Thomas) or R sqrt(K) (Matern cluster).

def thomas_kw(K, EN, c):
    return dict(parent_intensity=K, mean_offspring=EN / K, cluster_scale=c / (2 * math.sqrt(K)))


def matern_cluster_kw(K, EN, c):
    return dict(parent_intensity=K, mean_offspring=EN / K, cluster_radius=c / math.sqrt(K))


def nested_kw(K, mu1, EN, c1, c2):
    return dict(meta_parent_intensity=K, meta_offspring=mu1, mean_offspring=EN / (K * mu1),
                meta_cluster_scale=c1 / (2 * math.sqrt(K)),
                cluster_scale=c2 / (2 * math.sqrt(K * mu1)))


def simulate(kind: str, kw: dict, seed: int) -> np.ndarray:
    if kind == "binomial":
        return PoissonProcess().sample(n=int(kw["n"]), region=UNIT, seed=seed).points
    if kind == "matern_ii_dilated":
        # The repo's own Type II thinning, run on W (+) R and cropped to W, so every
        # point in W sees all its potential thinning neighbours (the as-is simulator
        # thins on W only; see the `matern2` part).
        proc = MaternHardCoreProcess(**kw)
        pts = proc._sample_points(None, UNIT.expanded(kw["hardcore_radius"]),
                                  np.random.default_rng(seed))
        return pts[UNIT.contains(pts)]
    cls = {
        "poisson": PoissonProcess, "thomas": ThomasProcess,
        "matern_cluster": MaternClusterProcess, "nested_thomas": NestedThomasProcess,
        "aniso_thomas": AnisotropicThomasProcess, "matern_ii": MaternHardCoreProcess,
        "strauss": StraussProcess, "lgcp": LGCPProcess,
    }[kind]
    return cls(**kw).sample(region=UNIT, seed=seed).points


def curves(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lmr = _isotropic_l_minus_r(pts, LOW, HIGH, R_GRID)
    f, g, _ = summstats.compute_fgj(pts, LOW, HIGH, FG_GRID)
    return lmr.astype(np.float32), f.astype(np.float32), g.astype(np.float32)


# ------------------------------------------------------------------ null
N_GRID = np.array([30, 50, 75, 100, 150, 200, 300, 400, 500, 650, 800, 1000, 1300, 1600])
N_NULL = 400


def _null_task(args):
    n, seed = args
    return curves(simulate("binomial", {"n": n}, seed))


def run_null(pool):
    tasks = [(int(n), 2_000_000 + 10_000 * i + m) for i, n in enumerate(N_GRID) for m in range(N_NULL)]
    res = pool.map(_null_task, tasks, chunksize=4)
    shape = (len(N_GRID), N_NULL, len(R_GRID))
    arr = {k: np.empty(shape, np.float32) for k in ("lmr", "f", "g")}
    for k, (l, f, g) in enumerate(res):
        i, m = divmod(k, N_NULL)
        arr["lmr"][i, m], arr["f"][i, m], arr["g"][i, m] = l, f, g
    np.savez_compressed(OUT / "null.npz", n_grid=N_GRID, r_grid=R_GRID, fg_grid=FG_GRID, **arr)
    tabs = dep.null_tables(OUT / "null.npz")
    for fn in dep.FUNCS:
        print(f"  null {fn}: c95_max by n = " + ", ".join(
            f"{n}:{c:.2f}" for n, c in zip(N_GRID, tabs[fn]["c95_max"])))


# ------------------------------------------------------------------ data
PROCS = ("thomas", "matern_cluster", "nested_thomas", "aniso_thomas", "strauss")
# scripts/build_classification_data.py: sources are the sorted data dirs, and a
# bundled cloud's seed is src_idx * 1_000_000 + its per-process seed.
SOURCES = ("aniso_thomas", "matern_cluster", "nested_thomas", "strauss", "thomas")
SEED_OFFSET = 1_000_000
PARAM_KEYS = {
    "thomas": ("parent_intensity", "mean_offspring", "cluster_scale"),
    "matern_cluster": ("parent_intensity", "mean_offspring", "cluster_radius"),
    "nested_thomas": ("parent_intensity", "meta_offspring", "meta_cluster_scale",
                      "mean_offspring", "cluster_scale"),
    "aniso_thomas": ("parent_intensity", "mean_offspring", "cluster_scale",
                     "cluster_aspect", "cluster_theta"),
    "strauss": ("beta", "gamma", "radius"),
}


def population_lmr(proc: str, P: np.ndarray) -> np.ndarray | None:
    r = R_GRID[None, :]
    if proc == "thomas":
        K = dep.K_thomas(r, P[:, [0]], P[:, [2]])
    elif proc == "matern_cluster":
        K = dep.K_matern_cluster(r, P[:, [0]], P[:, [2]])
    elif proc == "nested_thomas":
        K = dep.K_nested(r, P[:, [0]], P[:, [1]], P[:, [2]], P[:, [4]])
    else:
        return None
    return dep.l_minus_r(K, r)


def run_data(pool=None):
    tabs = dep.null_tables(OUT / "null.npz")
    files = {"test": ("clouds.lfunc_cache.npz", "clouds.summ_fg0250_classify_cache.npz", "clouds.pkl"),
             "adv": ("adversarial_clouds.lfunc_cache.npz",
                     "adversarial_clouds.summ_fg0250_classify_cache.npz", "adversarial_clouds.pkl")}
    out = {}
    for split, (lf, cf, pf) in files.items():
        cls = np.load(REPO / "data" / "classification" / cf)
        assert np.allclose(cls["fg_grid"], FG_GRID), "classification F/G grid is not linspace(0,.25,513)"
        c_seeds = cls["cloud_seeds"]
        c_f, c_g, c_l = cls["f_func"], cls["g_func"], cls["l_minus_r"]
        src, loc = c_seeds // SEED_OFFSET, c_seeds % SEED_OFFSET
        for proc in PROCS:
            z = np.load(REPO / "data" / proc / lf)
            assert np.allclose(z["r_grid"], R_GRID)
            seeds, n, lmr = z["cloud_seeds"], z["n_points"].astype(float), z["l_minus_r"]
            sel = np.flatnonzero(src == SOURCES.index(proc))
            pos = {int(s): k for k, s in zip(sel, loc[sel])}
            rows = np.array([pos[int(s)] for s in seeds])
            join_err = float(np.abs(c_l[rows] - lmr).max())
            f, g = c_f[rows], c_g[rows]
            with open(REPO / "data" / proc / pf, "rb") as fh:
                recs = pickle.load(fh)
            by_seed = {int(rec["seed"]): rec["params"] for rec in recs}
            P = np.array([[by_seed[int(s)][k] for k in PARAM_KEYS[proc]] for s in seeds], float)
            key = f"{proc}__{split}"
            out[f"{key}__n"] = n
            out[f"{key}__params"] = P
            for fn, cur in (("L", lmr), ("G", g), ("F", f)):
                d = dep.departure(cur, n, fn, tabs)
                for stat, v in d.items():
                    out[f"{key}__{fn}_{stat}"] = v
            pop = population_lmr(proc, P)
            if pop is not None:
                d = dep.departure(pop, n, "L", tabs, center=False)
                out[f"{key}__Lpop_S_max"] = d["S_max"]
            frac = {fn: float(np.mean(out[f"{key}__{fn}_S_max"] <= 1.0)) for fn in dep.FUNCS}
            print(f"  data {key:28s} N={len(n):5d} join_err={join_err:.2e} "
                  f"frac(S<=1) L={frac['L']:.3f} G={frac['G']:.3f} F={frac['F']:.3f}")
    np.savez_compressed(OUT / "data_departure.npz",
                        param_keys=np.array([f"{p}:{','.join(k)}" for p, k in PARAM_KEYS.items()]),
                        **out)


# ------------------------------------------------------------------ kfun
KFUN = {  # name: (simulator kind, kwargs)
    "thomas_A": ("thomas", thomas_kw(15, 400, 0.30)),           # few tight clusters
    "thomas_B": ("thomas", thomas_kw(60, 400, 0.30)),           # many small clusters
    "thomas_C": ("thomas", thomas_kw(60, 400, 0.90)),           # overlapping, near CSR
    "matern_cluster": ("matern_cluster", matern_cluster_kw(30, 400, 0.40)),
    "nested_A": ("nested_thomas", nested_kw(20, 4.0, 400, 0.50, 0.10)),
    "nested_B": ("nested_thomas", nested_kw(40, 3.0, 400, 0.60, 0.20)),
    "lgcp": ("lgcp", dict(mu=math.log(LAMBDA) - 0.5, sigma2=1.0, s=0.05)),
    "matern_ii": ("matern_ii_dilated", dict(parent_intensity=784.0, hardcore_radius=0.025)),
}
N_KFUN = 400


def _kfun_task(args):
    name, seed = args
    kind, kw = KFUN[name]
    pts = simulate(kind, kw, seed)
    return (len(pts),) + curves(pts)


def run_kfun(pool):
    names = list(KFUN)
    tasks = [(nm, 1_000_000 + 10_000 * j + m) for j, nm in enumerate(names) for m in range(N_KFUN)]
    res = pool.map(_kfun_task, tasks, chunksize=4)
    out = {}
    for j, nm in enumerate(names):
        block = res[j * N_KFUN:(j + 1) * N_KFUN]
        out[f"{nm}__n"] = np.array([b[0] for b in block])
        for k, key in enumerate(("lmr", "f", "g"), start=1):
            out[f"{nm}__{key}"] = np.stack([b[k] for b in block])
        out[f"{nm}__kw"] = np.array(repr(KFUN[nm]))
        n = out[f"{nm}__n"]
        print(f"  kfun {nm:15s} mean n={n.mean():7.1f} sd={n.std(ddof=1):6.1f} (target {LAMBDA:.0f})")
    np.savez_compressed(OUT / "kfun.npz", r_grid=R_GRID, fg_grid=FG_GRID, **out)


# ------------------------------------------------------------------ strauss
STRAUSS = {
    "weak": dict(beta=400.0, gamma=0.8, radius=0.02),
    "moderate": dict(beta=600.0, gamma=0.3, radius=0.03),
    "strong": dict(beta=900.0, gamma=0.05, radius=0.05),
    "small_R": dict(beta=400.0, gamma=0.05, radius=0.008),
}
STRAUSS_MULTS = (1, 10)
N_STRAUSS = 40
MARGIN = 2.0  # StraussProcess default margin_factor


def _strauss_task(args):
    name, mult, seed = args
    kw = STRAUSS[name]
    area = (1.0 + 2.0 * MARGIN * kw["radius"]) ** 2
    n_steps = mult * _adaptive_steps(None, kw["beta"] * area)   # mult=1 == the repo default
    t0 = time.perf_counter()
    pts = StraussProcess(**kw, n_steps=n_steps).sample(region=UNIT, seed=seed).points
    dt = time.perf_counter() - t0
    return (len(pts), dt, n_steps) + curves(pts)


def run_strauss(pool):
    from scipy.special import lambertw
    tasks = [(nm, mult, 3_000_000 + 10_000 * j + 1000 * mult + m)
             for j, nm in enumerate(STRAUSS) for mult in STRAUSS_MULTS for m in range(N_STRAUSS)]
    res = pool.map(_strauss_task, tasks, chunksize=1)
    out = {}
    k = 0
    for nm, kw in STRAUSS.items():
        G = (1.0 - kw["gamma"]) * math.pi * kw["radius"] ** 2
        lam_ps = float(np.real(lambertw(kw["beta"] * G)) / G)
        out[f"{nm}__lam_ps"] = lam_ps
        for mult in STRAUSS_MULTS:
            block = res[k:k + N_STRAUSS]
            k += N_STRAUSS
            key = f"{nm}__x{mult}"
            out[f"{key}__n"] = np.array([b[0] for b in block])
            out[f"{key}__secs"] = np.array([b[1] for b in block])
            out[f"{key}__steps"] = block[0][2]
            for i, name in enumerate(("lmr", "f", "g"), start=3):
                out[f"{key}__{name}"] = np.stack([b[i] for b in block])
            n = out[f"{key}__n"]
            print(f"  strauss {nm:9s} x{mult:<3d} steps={block[0][2]:7d} mean n={n.mean():7.1f} "
                  f"(se {n.std(ddof=1) / math.sqrt(len(n)):4.1f})  PS approx={lam_ps:7.1f}  "
                  f"secs/chain={out[f'{key}__secs'].mean():6.2f}")
    np.savez_compressed(OUT / "strauss.npz", r_grid=R_GRID, fg_grid=FG_GRID,
                        settings=np.array(repr(STRAUSS)), **out)


# ------------------------------------------------------------------ matern2 (as-is)
N_M2 = 400


def _matern2_task(seed):
    kw = KFUN["matern_ii"][1]
    R = kw["hardcore_radius"]
    res = []
    for kind in ("matern_ii", "matern_ii_dilated"):
        pts = simulate(kind, kw, seed)
        b = np.minimum(pts, 1.0 - pts).min(axis=1)
        res += [len(pts), int((b < R).sum()), int((b >= 2 * R).sum())]
    return res


def run_matern2(pool):
    kw = KFUN["matern_ii"][1]
    R = kw["hardcore_radius"]
    arr = np.array(pool.map(_matern2_task, [4_000_000 + m for m in range(N_M2)]), float)
    band, inner = 1.0 - (1.0 - 2 * R) ** 2, (1.0 - 4 * R) ** 2
    lam = dep.matern2_intensity(kw["parent_intensity"], R)
    out = {"raw": arr, "lambda_theory": lam, "band_area": band, "inner_area": inner}
    for j, kind in enumerate(("asis", "dilated")):
        n, nb, ni = arr[:, 3 * j], arr[:, 3 * j + 1], arr[:, 3 * j + 2]
        out[f"{kind}__mean_n"] = n.mean()
        out[f"{kind}__edge_ratio"] = (nb.sum() / band) / (ni.sum() / inner)
        print(f"  matern2 {kind:8s} mean n={n.mean():7.1f} (theory {lam:.1f})  "
              f"lambda(edge band)/lambda(interior)={out[f'{kind}__edge_ratio']:.3f}")
    np.savez_compressed(OUT / "matern2.npz", **out)


# ------------------------------------------------------------------ examples
EXAMPLES = {
    "Poisson": ("poisson", dict(intensity=LAMBDA)),
    "Thomas": ("thomas", thomas_kw(25, 400, 0.40)),
    "Matern cluster": ("matern_cluster", matern_cluster_kw(25, 400, 0.40)),
    "Anisotropic Thomas": ("aniso_thomas", dict(**thomas_kw(25, 400, 0.40),
                                                cluster_aspect=4.0, cluster_theta=math.pi / 4)),
    "Nested Thomas": ("nested_thomas", nested_kw(10, 5.0, 400, 0.60, 0.15)),
    "LGCP": ("lgcp", dict(mu=math.log(LAMBDA) - 1.0, sigma2=2.0, s=0.05)),
    "Matern II": ("matern_ii_dilated", dict(parent_intensity=784.0, hardcore_radius=0.025)),
    "Strauss": ("strauss", dict(beta=750.0, gamma=0.2, radius=0.025)),
}


def run_examples(pool=None):
    out = {}
    for j, (name, (kind, kw)) in enumerate(EXAMPLES.items()):
        pts = simulate(kind, kw, 5_000_000 + j)
        out[f"{name}__pts"] = pts
        out[f"{name}__kw"] = np.array(repr(kw))
        print(f"  example {name:20s} n={len(pts)}")
    np.savez_compressed(OUT / "examples.npz", **out)


# ------------------------------------------------------------------ ph
def run_ph(pool=None):
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    from gudhi.point_cloud.dtm import DistanceToMeasure
    from cloudforger.data_generation.filtration.dtm import DTMFiltration
    from cloudforger.data_generation.filtration.rips import RipsFiltration

    out = {}
    thomas = ThomasProcess(**thomas_kw(30, 400, 0.35)).sample(region=UNIT, seed=6_000_001)
    csr = PoissonProcess().sample(n=thomas.n_points, region=UNIT, seed=6_000_002)
    for tag, cloud in (("thomas", thomas), ("csr", csr)):
        pts = cloud.points
        rips = RipsFiltration(maxdim=1).compute(cloud)
        dtm = DTMFiltration(maxdim=1, k=5, q=2.0).compute(cloud)
        D = cdist(pts, pts)
        dtm_vals = DistanceToMeasure(5, q=2, metric="precomputed").fit_transform(D)
        mst = np.sort(minimum_spanning_tree(D).data)
        out[f"{tag}__pts"] = pts
        out[f"{tag}__dtm_vals"] = dtm_vals
        for d in (0, 1):
            out[f"{tag}__rips_h{d}"] = rips.diagrams[d]
            out[f"{tag}__dtm_h{d}"] = dtm.diagrams[d]
        r0 = rips.diagrams[0]
        fin = np.sort(r0[np.isfinite(r0[:, 1]), 1])
        mst_err = float(np.abs(fin - mst).max())
        births = np.sort(dtm.diagrams[0][:, 0])
        dtm_err = float(np.abs(births - np.sort(2.0 * dtm_vals)).max())
        print(f"  ph {tag}: n={len(pts)} |rips H0 deaths - MST edges|max={mst_err:.2e} "
              f"|dtm H0 births - 2*DTM_5|max={dtm_err:.2e} "
              f"#H1 rips={len(rips.diagrams[1])} dtm={len(dtm.diagrams[1])}")
        out[f"{tag}__mst_err"], out[f"{tag}__dtm_birth_err"] = mst_err, dtm_err

    # CSR checks over many clouds: share of MST edges that are nearest-neighbour
    # edges, and E[DTM_k^2] = (k-1)/(2 pi lambda) (gudhi counts the point itself).
    frac, dtm2 = [], []
    for m in range(100):
        pts = PoissonProcess().sample(n=400, region=UNIT, seed=6_100_000 + m).points
        D = cdist(pts, pts)
        T = minimum_spanning_tree(D).tocoo()
        mst_edges = {tuple(sorted(e)) for e in zip(T.row.tolist(), T.col.tolist())}
        nn = cKDTree(pts).query(pts, k=2)[1][:, 1]
        nn_edges = {tuple(sorted((i, int(j)))) for i, j in enumerate(nn)}
        frac.append(len(mst_edges & nn_edges) / len(mst_edges))
        v = DistanceToMeasure(5, q=2, metric="precomputed").fit_transform(D)
        interior = np.minimum(pts, 1.0 - pts).min(axis=1) > 0.1
        dtm2.append(float(np.mean(v[interior] ** 2)))
    out["csr_mst_nn_frac"] = np.array(frac)
    out["csr_dtm2"] = np.array(dtm2)
    print(f"  ph CSR n=400: MST edges that are NN edges {np.mean(frac):.3f} +- {np.std(frac):.3f}; "
          f"interior E[DTM_5^2]={np.mean(dtm2):.3e} vs (k-1)/(2 pi lambda)={4 / (2 * math.pi * 400):.3e}")
    np.savez_compressed(OUT / "ph.npz", **out)


# ------------------------------------------------------------------ main
PARTS = {"null": run_null, "data": run_data, "kfun": run_kfun, "strauss": run_strauss,
         "matern2": run_matern2, "examples": run_examples, "ph": run_ph}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--parts", nargs="*", default=list(PARTS))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    failed = []
    with Pool(args.workers) as pool:
        for part in args.parts:
            t0 = time.perf_counter()
            print(f"[{part}] start", flush=True)
            try:
                PARTS[part](pool)
            except Exception:
                traceback.print_exc()
                failed.append(part)
            print(f"[{part}] done in {time.perf_counter() - t0:.1f}s", flush=True)
    if failed:
        print(f"FAILED parts: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
