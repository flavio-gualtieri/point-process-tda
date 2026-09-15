#!/usr/bin/env python3
# scripts/processing/dv3_diagrams.py
"""Persistence diagrams for the DV3 clouds (data/dv3/<set>/<family>/clouds.pkl),
spread over a few wide SLURM jobs (slurm/dv3_diagrams_{compute,merge}.sh).

Same filtrations, same per-cloud computation and same output bundle as
scripts/processing/diagrams_shard.py (imported from there), laid out the way
DataPaths(family, root=data/dv3/<set>) expects:

    data/dv3/<set>/<family>/<tag>/diagrams.pkl        tag in rips, dtm_k5, dtm_k10, dtm_k15

How the work is split
---------------------
1. plan    Every (set, family, filtration) cloud list is cut into CONTIGUOUS
           chunks with a bounded estimated cost (DTM ~5 CPU-min or 40 clouds,
           whichever comes first; one huge cloud = its own chunk). Chunk
           boundaries depend only on the cloud sizes (points.npz offsets), so
           they are identical in every array task and on every re-run.
           Chunks are dealt to the --n-jobs array tasks by LPT (largest
           estimated cost first, to the least-loaded task).
2. run     One array task = one node allocation (e.g. 96 CPUs). A driver
           process stages its chunks' cloud records to node-local disk, then
           launches ONE FRESH `python ... worker` PROCESS PER CHUNK (fork+exec,
           no worker pool), up to one per CPU, admitting a chunk only if its
           estimated peak memory fits the remaining budget.
3. merge   Stitches a (set, family, filtration)'s chunks back in cloud order
           and writes the bundle; refuses on gaps, overlaps or seed mismatches.

Why this shape (memory)
-----------------------
* GUDHI's DTM / weighted-Rips backend leaks ~80 MB per diagram (see
  diagrams_shard.py; a forked in-process pool deadlocked, job 25038922).
  Each worker process handles <= 40 diagrams and exits, so the leak resets.
* With thresh=None the DTM complex is the full 2-skeleton: C(n,3) triangles,
  ~n^3 memory. n=1320 (the largest DV3 A cloud) is ~3.8e8 simplices -- tens
  of GB for ONE diagram, while a typical n~330 cloud needs well under 1 GB.
  So the driver estimates each chunk's peak from its largest cloud
  (DTM_BYTES_PER_SIMPLEX, deliberately pessimistic) and only launches what
  fits in ~88% of the job's --mem. A blocked big chunk reserves its memory
  so small chunks cannot starve it.
* Self-correcting: every worker reports its real peak RSS; if the model
  under-predicted, the driver raises its scale factor for all later
  launches. A worker killed by the OOM killer is re-queued with twice the
  estimate (up to MAX_ATTEMPTS), so one bad estimate never loses the job.

Usage
-----
    # dry run: chunk counts, per-task balance, heaviest chunks (reads only sizes)
    python scripts/processing/dv3_diagrams.py plan --n-jobs 20

    # one array task (inside an allocation; CPUs / memory from SLURM env):
    python scripts/processing/dv3_diagrams.py run --job-index 3 --n-jobs 20

    # after all run tasks finished; group index = one (set, family, filtration):
    python scripts/processing/dv3_diagrams.py merge --group-index 17
    python scripts/processing/dv3_diagrams.py groups        # list the group indices

`run` is resumable: chunks whose output exists (or whose group is already
merged) are skipped. Re-submit the same array (same --n-jobs) to finish.
"""

from __future__ import annotations

import os

# Single-threaded numeric backends, set before numpy is imported. Each worker
# process is one CPU; GUDHI's DTM backend is multi-threaded otherwise.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import heapq
import json
import pickle
import resource
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from cloudforger.core.io import dump_pickle, load_pickle
from cloudforger.core.records import diagram_to_record, to_pointcloud
from diagrams_shard import FILTRATIONS, _final_bundle, build_filtration

DEFAULT_DV3_ROOT = ROOT / "data" / "dv3"
DEFAULT_SETS = ("A", "B", "C")
FILTRATION_ORDER = ("rips", "dtm_k5", "dtm_k10", "dtm_k15")
assert set(FILTRATION_ORDER) == set(FILTRATIONS)

CHUNKS_SUBDIR = "_chunks"

# --- cost model (CPU seconds; only used for chunking + load balancing) -------
# DTM: ~10 s per diagram at n=400, ~n^3.1 (docs/theory generation budget).
DTM_S_AT_400, DTM_EXP = 10.0, 3.1
RIPS_S_AT_400, RIPS_EXP = 0.5, 2.5
CHUNK_TARGET_S = 300.0
CHUNK_MAX_CLOUDS = {"rips": 400, "dtm": 40}

# --- memory model (MB; gates launches) ----------------------------------------
BASE_MB = 600.0                  # interpreter + numpy + gudhi + cloudforger (~140 MB measured)
# Measured: max observed peak / model = 0.50-0.55 at 100 B/simplex over ~4,000
# chunks incl. the n=1320 cloud (job 26796165 tasks 0-3,18) -> ~55 B/simplex real.
# 70 keeps a ~1.27x margin; the adaptive scale and OOM retry cover the rest.
DTM_BYTES_PER_SIMPLEX = 70.0
LEAK_MB_PER_DIAGRAM = 100.0      # GUDHI DTM leak (~80 MB measured), accumulates per process
RIPS_MB_PER_N2 = 4.0e-4          # ripser, dense distances + H1 reduction; generous
BUDGET_FRACTION = 0.88           # of the job's --mem; rest = driver, page cache, slack
MAX_ATTEMPTS = 3

# --- transient OS resource exhaustion (fork()/pthread_create() EAGAIN) --------
# The `compute` partition shares physical nodes across jobs (CR_CORE_MEMORY
# select type, not exclusive), so a burst of ~96 near-simultaneous forks can
# hit "Resource temporarily unavailable" from either our own Popen() or from
# GUDHI's compute_persistence() spawning a thread -- observed 2026-09-14/15,
# job 26796165: 14/20 tasks died in under a minute, all at this error, none
# from OOM. Distinct from a real chunk bug: retried with backoff, capped
# separately, and never counted as a memory (OOM) problem.
TRANSIENT_MARKERS = ("Resource temporarily unavailable", "pthread_create has failed")
TRANSIENT_MAX_ATTEMPTS = 12
TRANSIENT_BACKOFF_BASE_S = 3.0
TRANSIENT_BACKOFF_CAP_S = 60.0
LAUNCH_STAGGER_S = 0.03           # between successive Popen calls in one burst
LAUNCH_RETRY_BACKOFF_S = 2.0      # after Popen itself fails to fork


def _kind(filt: str) -> str:
    return "rips" if filt == "rips" else "dtm"


def est_seconds(filt: str, n: int) -> float:
    if _kind(filt) == "rips":
        return 0.05 + RIPS_S_AT_400 * (n / 400.0) ** RIPS_EXP
    return 0.05 + DTM_S_AT_400 * (n / 400.0) ** DTM_EXP


def n_simplices_2skeleton(n: int) -> int:
    return n * (n - 1) * (n - 2) // 6 + n * (n - 1) // 2 + n


def transient_mb(filt: str, n: int) -> float:
    """Model of one diagram's peak working set above BASE_MB."""
    if _kind(filt) == "rips":
        return 256.0 + RIPS_MB_PER_N2 * n * n
    return DTM_BYTES_PER_SIMPLEX * n_simplices_2skeleton(n) / 2**20


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    set: str
    family: str
    filt: str
    start: int
    end: int
    n_total: int
    max_n: int
    cost_s: float

    @property
    def key(self) -> str:
        return f"{self.set}/{self.family}/{self.filt}/{self.start:06d}-{self.end:06d}"

    @property
    def stage_name(self) -> str:
        # DTM k5/k10/k15 share boundaries, so they share one staged input.
        return f"{self.set}_{self.family}_{_kind(self.filt)}_{self.start:06d}_{self.end:06d}.pkl"

    @property
    def count(self) -> int:
        return self.end - self.start

    def mem_mb(self, scale: float) -> float:
        leak = LEAK_MB_PER_DIAGRAM * self.count if _kind(self.filt) == "dtm" else 0.0
        return BASE_MB + scale * transient_mb(self.filt, self.max_n) + leak


def group_dir(root: Path, set_: str, family: str, filt: str) -> Path:
    # == DataPaths(family, root=root/set_).filtration_dir([filt]) -- asserted in build_filtration.
    return root / set_ / family / filt


def chunk_out_path(root: Path, c: Chunk) -> Path:
    return group_dir(root, c.set, c.family, c.filt) / CHUNKS_SUBDIR / f"chunk_{c.start:06d}_{c.end:06d}.pkl"


def final_path(root: Path, set_: str, family: str, filt: str) -> Path:
    return group_dir(root, set_, family, filt) / "diagrams.pkl"


def families(root: Path, set_: str) -> list[str]:
    d = root / set_
    return sorted(p.name for p in d.iterdir() if (p / "clouds.pkl").exists() and (p / "points.npz").exists())


def cloud_sizes(root: Path, set_: str, family: str) -> np.ndarray:
    with np.load(root / set_ / family / "points.npz") as z:  # reads only the offsets member
        return np.diff(z["offsets"]).astype(np.int64)


def chunk_ranges(sizes: np.ndarray, filt: str) -> list[tuple[int, int]]:
    cap = CHUNK_MAX_CLOUDS[_kind(filt)]
    ranges: list[tuple[int, int]] = []
    start, acc = 0, 0.0
    for i, n in enumerate(sizes):
        c = est_seconds(filt, int(n))
        if i > start and (acc + c > CHUNK_TARGET_S or i - start >= cap):
            ranges.append((start, i))
            start, acc = i, 0.0
        acc += c
    if start < len(sizes):
        ranges.append((start, len(sizes)))
    return ranges


def groups(root: Path, sets: list[str], filts: list[str]) -> list[tuple[str, str, str]]:
    return [(s, fam, f) for s in sets for fam in families(root, s) for f in filts]


def build_plan(root: Path, sets: list[str], filts: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    sizes_cache: dict[tuple[str, str], np.ndarray] = {}
    for s, fam, f in groups(root, sets, filts):
        sizes = sizes_cache.setdefault((s, fam), cloud_sizes(root, s, fam))
        for a, b in chunk_ranges(sizes, f):
            seg = sizes[a:b]
            chunks.append(Chunk(s, fam, f, a, b, len(sizes), int(seg.max()),
                                float(sum(est_seconds(f, int(n)) for n in seg))))
    return chunks


def assign(chunks: list[Chunk], n_jobs: int) -> dict[str, int]:
    """LPT: heaviest first, each to the currently least-loaded job (ties -> lowest index)."""
    heap = [(0.0, j) for j in range(n_jobs)]
    owner: dict[str, int] = {}
    for c in sorted(chunks, key=lambda c: (-c.cost_s, c.key)):
        load, j = heapq.heappop(heap)
        owner[c.key] = j
        heapq.heappush(heap, (load + c.cost_s, j))
    return owner


def run_plan_report(root: Path, sets: list[str], filts: list[str], n_jobs: int, cpus: int, mem_gb: float) -> None:
    chunks = build_plan(root, sets, filts)
    owner = assign(chunks, n_jobs)
    budget = BUDGET_FRACTION * mem_gb * 1024
    per_job = np.zeros(n_jobs)
    for c in chunks:
        per_job[owner[c.key]] += c.cost_s
    by_kind = {k: [c for c in chunks if _kind(c.filt) == k] for k in ("rips", "dtm")}
    n_groups = len(groups(root, sets, filts))
    print(f"sets={sets} filtrations={filts}  groups={n_groups}  chunks={len(chunks)} "
          f"(rips {len(by_kind['rips'])}, dtm {len(by_kind['dtm'])})")
    print(f"estimated total {per_job.sum() / 3600:.0f} CPU-h; per job min/mean/max "
          f"{per_job.min() / 3600:.1f}/{per_job.mean() / 3600:.1f}/{per_job.max() / 3600:.1f} CPU-h "
          f"-> ~{per_job.max() / cpus / 60:.0f} min wall at {cpus} CPUs (model, before memory throttling)")
    heavy = sorted(chunks, key=lambda c: -c.mem_mb(1.0))[:5]
    print(f"memory budget per job {budget / 1024:.0f} GB; largest chunk estimates:")
    for c in heavy:
        print(f"  {c.key:40s} n_max={c.max_n:5d} clouds={c.count:3d} est {c.mem_mb(1.0) / 1024:5.1f} GB "
              f"{c.cost_s / 60:5.1f} CPU-min")
    too_big = [c for c in chunks if c.mem_mb(1.0) > budget]
    if too_big:
        print(f"WARNING: {len(too_big)} chunk(s) exceed the whole budget; they will run alone, capped.")
    print(f"group indices for merge: 0..{n_groups - 1}")


# ---------------------------------------------------------------------------
# run (driver)
# ---------------------------------------------------------------------------


def _atomic_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    dump_pickle(tmp, payload)
    os.replace(tmp, path)


def _slurm_cpus() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE") or os.cpu_count() or 1)


def _slurm_mem_mb() -> float:
    if os.environ.get("SLURM_MEM_PER_NODE"):
        return float(os.environ["SLURM_MEM_PER_NODE"])
    if os.environ.get("SLURM_MEM_PER_CPU"):
        return float(os.environ["SLURM_MEM_PER_CPU"]) * _slurm_cpus()
    raise SystemExit("no SLURM_MEM_PER_NODE / SLURM_MEM_PER_CPU in the environment; pass --mem-gb")


def _stage(root: Path, stage_dir: Path, todo: list[Chunk]) -> None:
    by_family: dict[tuple[str, str], list[Chunk]] = {}
    for c in todo:
        by_family.setdefault((c.set, c.family), []).append(c)
    for (s, fam), cs in sorted(by_family.items()):
        clouds = load_pickle(root / s / fam / "clouds.pkl")
        sizes = cloud_sizes(root, s, fam)
        if len(clouds) != len(sizes) or any(int(c["n_points"]) != int(n) for c, n in zip(clouds, sizes)):
            raise SystemExit(f"{s}/{fam}: clouds.pkl and points.npz disagree on cloud count/sizes")
        for c in cs:
            p = stage_dir / c.stage_name
            if not p.exists():
                _atomic_pickle(p, {"records": clouds[c.start:c.end], "start": c.start, "end": c.end,
                                   "n_total": len(clouds)})
        print(f"  staged {s}/{fam}: {len({c.stage_name for c in cs})} input(s) for {len(cs)} chunk(s)", flush=True)
        del clouds


def run_driver(root: Path, sets: list[str], filts: list[str], job_index: int, n_jobs: int,
               cpus: int, mem_mb: float, mem_scale: float, stage_root: Path, force: bool) -> None:
    chunks = build_plan(root, sets, filts)
    owner = assign(chunks, n_jobs)
    mine = [c for c in chunks if owner[c.key] == job_index]
    merged = {g for g in groups(root, sets, filts) if final_path(root, *g).exists()}
    todo = [c for c in mine
            if force or ((c.set, c.family, c.filt) not in merged and not chunk_out_path(root, c).exists())]
    todo.sort(key=lambda c: (-c.cost_s, c.key))

    budget = BUDGET_FRACTION * mem_mb
    print(f"job {job_index}/{n_jobs}: {len(mine)} chunks assigned, {len(todo)} to do "
          f"({sum(c.cost_s for c in todo) / 3600:.1f} est CPU-h); {cpus} CPUs, "
          f"memory budget {budget / 1024:.0f} of {mem_mb / 1024:.0f} GB", flush=True)
    if not todo:
        print("nothing to do.")
        return

    stage_dir = stage_root / f"dv3diag_{os.environ.get('SLURM_JOB_ID', 'local')}_{job_index}"
    log_dir = stage_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[subprocess.Popen, tuple[Chunk, float, float, Path]] = {}

    def _cleanup(*_):
        for p in running:
            p.kill()
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise SystemExit("terminated (SIGTERM / time limit) -- re-submit to resume; finished chunks are kept.")

    signal.signal(signal.SIGTERM, _cleanup)

    try:
        _stage(root, stage_dir, todo)
        _schedule(root, todo, stage_dir, log_dir, running, cpus, budget, mem_scale)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def _schedule(root: Path, todo: list[Chunk], stage_dir: Path, log_dir: Path,
              running: dict, cpus: int, budget: float, scale: float) -> None:
    pending = list(todo)
    boost: dict[str, float] = {}      # per-chunk multiplier after an OOM kill
    attempts: dict[str, int] = {}
    transient_attempts: dict[str, int] = {}   # separate, more generous cap -- not the chunk's fault
    retry_after: dict[str, float] = {}        # backoff for transient failures
    failed: list[str] = []
    done = 0
    peak_ratio = 0.0                  # max observed (peak - base - leak) / modelled transient
    t_start = last_report = time.time()
    last_launch_fail_log = 0.0
    launch_paused_until = 0.0
    worker_cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "worker"]

    def need(c: Chunk) -> float:
        return min(c.mem_mb(scale) * boost.get(c.key, 1.0), budget)

    while pending or running:
        # ---- reap -----------------------------------------------------------
        for p in [p for p in running if p.poll() is not None]:
            c, est, t0, log = running.pop(p)
            stats_path = log.with_suffix(".json")
            if p.returncode == 0 and chunk_out_path(root, c).exists():
                done += 1
                if stats_path.exists():
                    st = json.loads(stats_path.read_text())
                    leak = LEAK_MB_PER_DIAGRAM * c.count if _kind(c.filt) == "dtm" else 0.0
                    model = transient_mb(c.filt, c.max_n)
                    if _kind(c.filt) == "dtm" and model >= 2048:
                        ratio = max(st["peak_rss_mb"] - BASE_MB - leak, 0.0) / model
                        peak_ratio = max(peak_ratio, ratio)
                        if 1.2 * ratio > scale:
                            scale = 1.2 * ratio
                            print(f"  [mem] {c.key} n_max={c.max_n} peaked {st['peak_rss_mb'] / 1024:.1f} GB "
                                  f"(est {est / 1024:.1f}) -> memory scale raised to {scale:.2f}", flush=True)
                    if st["peak_rss_mb"] > est:
                        print(f"  [mem] WARNING {c.key} peaked {st['peak_rss_mb'] / 1024:.1f} GB "
                              f"above its {est / 1024:.1f} GB estimate", flush=True)
                continue
            oom = p.returncode in (-signal.SIGKILL, 137)
            tail = log.read_text()[-2000:] if log.exists() else "(no log)"
            transient = (not oom) and any(m in tail for m in TRANSIENT_MARKERS)
            if transient:
                transient_attempts[c.key] = transient_attempts.get(c.key, 0) + 1
                n = transient_attempts[c.key]
                backoff = min(TRANSIENT_BACKOFF_BASE_S * (2 ** min(n - 1, 4)), TRANSIENT_BACKOFF_CAP_S)
                print(f"  [fail] {c.key} rc={p.returncode} (transient: node thread/process exhaustion, "
                      f"not this chunk's fault) attempt {n}/{TRANSIENT_MAX_ATTEMPTS}; "
                      f"retrying in {backoff:.0f}s", flush=True)
                if n < TRANSIENT_MAX_ATTEMPTS:
                    retry_after[c.key] = time.time() + backoff
                    pending.append(c)   # back of the queue -- let others make progress meanwhile
                else:
                    failed.append(c.key)
                continue
            attempts[c.key] = attempts.get(c.key, 0) + 1
            print(f"  [fail] {c.key} rc={p.returncode}{' (killed: likely OOM)' if oom else ''} "
                  f"attempt {attempts[c.key]}/{MAX_ATTEMPTS}; log tail:\n{tail}", flush=True)
            if attempts[c.key] < MAX_ATTEMPTS:
                if oom:
                    boost[c.key] = 2.0 * boost.get(c.key, 1.0)
                pending.insert(0, c)
            else:
                failed.append(c.key)

        # ---- launch (first fit, one reservation for the first blocked chunk) ---
        used = sum(v[1] for v in running.values())
        reserve = 0.0
        i = 0
        now_launch = time.time()
        while len(running) < cpus and i < len(pending) and now_launch >= launch_paused_until:
            c = pending[i]
            if retry_after.get(c.key, 0.0) > now_launch:
                i += 1
                continue
            m = need(c)
            if used + m <= budget - reserve:
                log = log_dir / (c.key.replace("/", "__") + ".log")
                try:
                    with open(log, "w") as fh:
                        p = subprocess.Popen(
                            worker_cmd + ["--stage", str(stage_dir / c.stage_name), "--filtration", c.filt,
                                          "--out", str(chunk_out_path(root, c)), "--stats", str(log.with_suffix(".json"))],
                            stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT),
                        )
                except OSError as exc:
                    # fork()/pthread_create() itself refused (EAGAIN under node-wide thread/PID
                    # pressure) -- transient and not this chunk's fault. Don't burn an attempt,
                    # don't advance i (retry the same chunk once things ease), just stop trying
                    # to launch anything for a bit so we're not hammering an already-strained node.
                    if time.time() - last_launch_fail_log > 5.0:
                        print(f"  [launch] Popen failed ({exc}) -- node thread/process pressure; "
                              f"pausing launches {LAUNCH_RETRY_BACKOFF_S:.0f}s", flush=True)
                        last_launch_fail_log = time.time()
                    launch_paused_until = time.time() + LAUNCH_RETRY_BACKOFF_S
                    break
                running[p] = (c, m, time.time(), log)
                used += m
                pending.pop(i)
                time.sleep(LAUNCH_STAGGER_S)  # avoid forking the whole burst in one instant
                now_launch = time.time()
            else:
                if reserve == 0.0:
                    reserve = m   # memory freed later goes to this chunk before anything smaller
                i += 1

        now = time.time()
        if now - last_report >= 60 or (not pending and not running):
            print(f"  [{(now - t_start) / 60:6.1f} min] done {done}  running {len(running)}  "
                  f"pending {len(pending)}  failed {len(failed)}  est mem in use {used / 1024:.0f} GB  "
                  f"scale {scale:.2f}", flush=True)
            last_report = now
        if pending or running:
            time.sleep(0.5)

    print(f"max observed DTM peak / model = {peak_ratio:.2f} (model = {DTM_BYTES_PER_SIMPLEX:.0f} B/simplex)")
    if failed:
        raise SystemExit(f"{len(failed)} chunk(s) failed after {MAX_ATTEMPTS} attempts: {failed}")
    print(f"all {done} chunk(s) finished in {(time.time() - t_start) / 60:.1f} min.")


# ---------------------------------------------------------------------------
# worker (one fresh process per chunk)
# ---------------------------------------------------------------------------


def run_worker(stage: Path, filt_name: str, out: Path, stats: Path) -> None:
    t0 = time.time()
    payload = load_pickle(stage)
    filt = build_filtration(filt_name)
    records = payload["records"]
    diagrams = []
    for i, rec in enumerate(records):
        diagrams.append(diagram_to_record(filt.compute(to_pointcloud(rec))))
        if (i + 1) % 10 == 0 or i + 1 == len(records):
            print(f"{filt_name} [{payload['start']}:{payload['end']}) {i + 1}/{len(records)} "
                  f"{time.time() - t0:.0f}s", flush=True)
    _atomic_pickle(out, {
        "diagrams": diagrams,
        "start": payload["start"],
        "end": payload["end"],
        "n_total": payload["n_total"],
        "filtration": filt_name,
        "seeds": [r.get("seed") for r in records],
    })
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB on Linux
    stats.write_text(json.dumps({"peak_rss_mb": peak_mb, "wall_s": time.time() - t0, "count": len(records),
                                 "max_n": max(len(r["points"]) for r in records)}))


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def run_merge(root: Path, set_: str, family: str, filt: str, force: bool, keep_chunks: bool) -> None:
    tag = f"{set_}/{family}/{filt}"
    out = final_path(root, set_, family, filt)
    if out.exists() and not force:
        print(f"[{tag}] {out} exists -- skipping (pass --force).")
        return
    clouds = load_pickle(root / set_ / family / "clouds.pkl")
    n = len(clouds)
    chunk_dir = group_dir(root, set_, family, filt) / CHUNKS_SUBDIR
    files = sorted(chunk_dir.glob("chunk_*.pkl")) if chunk_dir.exists() else []

    records: list[dict] = []
    pos = 0
    for f in files:
        part = load_pickle(f)
        if part["n_total"] != n or part["filtration"] != filt:
            raise SystemExit(f"[{tag}] {f.name}: n_total={part['n_total']} filtration={part['filtration']!r}, "
                             f"expected {n} / {filt!r} -- stale chunk, delete it and recompute.")
        if part["start"] != pos:
            raise SystemExit(f"[{tag}] gap/overlap: expected a chunk starting at {pos}, got {f.name}. "
                             f"Re-run the compute array (it is resumable), then merge again.")
        if part["seeds"] != [c["seed"] for c in clouds[part["start"]:part["end"]]]:
            raise SystemExit(f"[{tag}] {f.name}: seeds do not match clouds.pkl -- stale chunk.")
        records.extend(part["diagrams"])
        pos = part["end"]
    if pos != n:
        raise SystemExit(f"[{tag}] chunks cover [0, {pos}) of {n} clouds -- missing chunks from {pos}. "
                         f"Re-run the compute array (it is resumable), then merge again.")

    bundle = _final_bundle(records, clouds)
    bundle["case_ids"] = [c.get("case_id") for c in clouds]
    bundle["splits"] = [c.get("split") for c in clouds]
    _atomic_pickle(out, bundle)
    print(f"[{tag}] {len(records)} diagrams from {len(files)} chunk(s) -> {out}")
    if not keep_chunks:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        print(f"[{tag}] removed {chunk_dir}")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--dv3-root", type=Path, default=DEFAULT_DV3_ROOT)
        p.add_argument("--sets", nargs="+", default=list(DEFAULT_SETS))
        p.add_argument("--filtrations", nargs="+", default=list(FILTRATION_ORDER), choices=FILTRATION_ORDER)

    p = sub.add_parser("plan", help="dry run: chunking and per-job balance")
    common(p)
    p.add_argument("--n-jobs", type=int, required=True)
    p.add_argument("--cpus", type=int, default=48)
    p.add_argument("--mem-gb", type=float, default=180)  # matches slurm/dv3_diagrams_compute.sh

    p = sub.add_parser("groups", help="list merge group indices")
    common(p)

    p = sub.add_parser("run", help="driver for one array task")
    common(p)
    p.add_argument("--job-index", type=int, required=True)
    p.add_argument("--n-jobs", type=int, required=True)
    p.add_argument("--cpus", type=int, default=None, help="default: SLURM_CPUS_PER_TASK")
    p.add_argument("--mem-gb", type=float, default=None, help="default: SLURM_MEM_PER_NODE")
    p.add_argument("--mem-scale", type=float, default=1.0, help="multiply the memory model (>1 = more cautious)")
    p.add_argument("--stage-root", type=Path, default=Path(os.environ.get("TMPDIR") or "/tmp"))
    p.add_argument("--force", action="store_true", help="recompute chunks that already exist")

    p = sub.add_parser("worker", help="internal: compute one staged chunk")
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--filtration", required=True, choices=FILTRATION_ORDER)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stats", type=Path, required=True)

    p = sub.add_parser("merge", help="stitch one (set, family, filtration) group")
    common(p)
    p.add_argument("--group-index", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-chunks", action="store_true")

    a = ap.parse_args(argv)

    if a.mode == "worker":
        run_worker(a.stage, a.filtration, a.out, a.stats)
        return
    root = a.dv3_root.resolve()
    if a.mode == "plan":
        run_plan_report(root, a.sets, a.filtrations, a.n_jobs, a.cpus, a.mem_gb)
    elif a.mode == "groups":
        for i, g in enumerate(groups(root, a.sets, a.filtrations)):
            print(i, "/".join(g), "(merged)" if final_path(root, *g).exists() else "")
    elif a.mode == "run":
        if not (0 <= a.job_index < a.n_jobs):
            raise SystemExit(f"--job-index {a.job_index} out of range for --n-jobs {a.n_jobs}")
        cpus = a.cpus or _slurm_cpus()
        mem_mb = a.mem_gb * 1024 if a.mem_gb else _slurm_mem_mb()
        run_driver(root, a.sets, a.filtrations, a.job_index, a.n_jobs, cpus, mem_mb,
                   a.mem_scale, a.stage_root, a.force)
    elif a.mode == "merge":
        gs = groups(root, a.sets, a.filtrations)
        if not (0 <= a.group_index < len(gs)):
            print(f"group index {a.group_index} >= {len(gs)} groups -- nothing to do.")
            return
        run_merge(root, *gs[a.group_index], a.force, a.keep_chunks)


if __name__ == "__main__":
    main()
