# Frozen legacy state — `legacy-pre-kernel`

Snapshot of the project as it stood **before** the `Kernel` abstraction replaced the
`displacement_sampler` + hand-computed `edge_buffer` model. Captured so the exact
code, configs, environment, and results behind `writeup/short_report.pdf` can be
recovered after the refactor and the full data regeneration that follows it.

Created 2026-09-02.

---

## Pointers

| | |
|---|---|
| **Git tag** | `legacy-pre-kernel` (annotated, pushed to `origin`) |
| **Legacy code commit** | `270eb88c77664e60a5815e8350ab5c6396624579` — "Update README to remove obsolete archive details", 2026-08-28 19:59 +0100 |
| **Tagged commit** | the freeze commit that adds this file, `configs/frozen/`, and `requirements.lock`; its parent is `270eb88` |
| **Numbers of record** | `writeup/short_report.pdf` (sha256 `d30faf5c…836e`), source `writeup/short_report.tex` (sha256 `2ec9f78a…f437`) |
| **Results ledger** | `results/experiments.jsonl` — 765 rows, sha256 `dbdb3852…ac28`; every row already carries the producing commit hash via `provenance.py` |
| **Offline history copy** | `legacy-pre-kernel.bundle` (`git bundle`, stored off-repo) |
| **Exact byte archive of `data/` + `results/`** | `legacy-pre-kernel-data.tar.zst`, stored off-repo — **required** (see reproducibility verdict below); sha256 in `legacy-pre-kernel-data.sha256` |

---

## What is frozen

- **Source** — the whole tree at `270eb88`. The legacy generator is
  `src/cloudforger/data_generation/point_processes/{neyman_scott,thomas,nested_thomas,inhom_thomas}.py`
  with `NeymanScottProcess(displacement_sampler=…, edge_buffer=…)`, the
  `gaussian_displacements` / `uniform_ball_displacements` factories, and the
  `edge_buffer = 4.0 * cluster_scale` convention in each subclass.
- **Configs** — `configs/frozen/report-2026-09/` is a verbatim copy of `configs/runs/`
  as of this commit (77 YAML files). These stay runnable and are what reproduce the
  report's experiment set.
- **Environment** — `requirements.lock` (`pip freeze`, 102 entries). Note: it was
  produced in a conda env, so some lines are `@ file:///…` local paths and are **not**
  portable for a clean reinstall — treat the file as a record. The versions that
  actually matter:

  | package | version |
  |---|---|
  | Python | 3.14.4 |
  | numpy | 2.4.5 |
  | scipy | 1.17.1 |
  | torch | 2.10.0 |
  | gudhi | 3.12.0 |
  | ripser | 0.6.14 |

  Platform at freeze time: macOS 26.6.2 (build 25G83), Darwin 25.6.0, arm64.
  **The committed `data/` predates this environment** (`data/nested_thomas/clouds.pkl`
  dated 2026-08-08, `data/thomas/clouds.pkl` 2026-08-10) and was produced with an
  older numpy — see the verdict below.

---

## Reproduction recipe

```bash
git checkout legacy-pre-kernel
python -m venv .venv-legacy && source .venv-legacy/bin/activate
pip install -e .            # or reconstruct from requirements.lock where portable

# 1. clouds (train/test + adversarial, one call each)
python scripts/generate.py  configs/frozen/report-2026-09/thomas/thomas_pi_multik_k5k10k15.yaml
python scripts/generate.py  configs/frozen/report-2026-09/nested_thomas/pi_multik.yaml

# 2. diagrams + persistence images
python scripts/featurize.py configs/frozen/report-2026-09/thomas/thomas_pi_multik_k5k10k15.yaml
python scripts/featurize.py configs/frozen/report-2026-09/nested_thomas/pi_multik.yaml

# 3. train / fit each arm in the report (pi_multik, vihrs, mincontrast, mincontrast_g,
#    the vec_multik / betti_multik vectorization ablations) then evaluate — see README.
```

Baselines and ablation arms use the other configs in the same `configs/frozen/report-2026-09/`
subfolders (`*_vihrs.yaml`, `mincontrast*.yaml`, `vec_multik_*.yaml`, `betti_*.yaml`).

---

## Reproducibility verdict (checked 2026-09-02, Python 3.14.4 / numpy 2.4.5 / macOS 26.6 arm64)

Regenerated both processes from `configs/frozen/report-2026-09/` into a scratch dir and
compared against the committed `data/`.

**Thomas — bit-identical.**

| file | committed sha256 | regenerated sha256 | |
|---|---|---|---|
| `data/thomas/clouds.pkl` | `249b05d7…9996` | `249b05d7…9996` | match |
| `data/thomas/adversarial_clouds.pkl` | `dbea77eb…f7de` | `dbea77eb…f7de` | match |
| `cloud_generation_manifest.yaml` | — | — | identical |

**Nested Thomas — statistically identical, NOT bit-identical.**

| file | committed sha256 | regenerated sha256 | |
|---|---|---|---|
| `data/nested_thomas/clouds.pkl` | `7d11354b…c5d0` | `33977b8b…f150` | **differ** |
| `data/nested_thomas/adversarial_clouds.pkl` | `a5a4f22d…7df3` | `717a88e0…92ac` | **differ** |
| manifest `train_test_stats` / `adversarial_stats` | — | — | identical (7000 + 1000 clouds; total/min/max point counts match exactly) |
| manifest `adversarial_params` float values | — | — | differ at the ~1 ULP level |

The sampled parameter vectors differ by ~1e-16 (floating-point rounding in the longer
`derived:` chain — nested `sqrt`/division — under a newer numpy than produced the
2026-08-08 committed data). Cloud coordinates then drift by the same order and the pickle
hashes diverge. **No reported statistic is affected** (same seeds, same design grid,
identical point-count stats), but the exact bytes of `data/nested_thomas/*.pkl` used for
the report exist only in the external archive.

**Consequence:** Thomas is fully reproducible from code + config + seed. Nested Thomas is
reproducible only up to floating-point noise → the external `data/` + `results/` archive
(`legacy-pre-kernel-data.tar.zst`) is the authoritative copy and **must** be retained.

---

## How to recover

```bash
# code + configs + this file + lockfile
git checkout legacy-pre-kernel
#   or, if the repo/remote is gone:
git clone legacy-pre-kernel.bundle point-process-tda && cd point-process-tda && git checkout legacy-pre-kernel

# environment
python -m venv .venv-legacy && source .venv-legacy/bin/activate && pip install -e .

# exact data + results the report used
#   download legacy-pre-kernel-data.tar.zst, verify against legacy-pre-kernel-data.sha256, then:
tar -I zstd -xf legacy-pre-kernel-data.tar.zst      # restores data/ and results/

# Thomas can instead be regenerated bit-for-bit with the recipe above.
```

Every existing `results/**/results.json` and every `results/experiments.jsonl` row already
records the commit that produced it; this tag just gives that commit a name.
