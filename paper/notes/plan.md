# Plan: freeze, audit, protagonist analysis (tasks 1–3)

2026-09-30 · commit `b5d617bb` (dirty) · run `oneshot/results/default` · task 4 (paper plan) not started.

Where each number comes from:
- **[S]** `results/summary.csv` (built by `paper/scripts/summary.py` from the run's stored reports).
- **[X]** recomputed today from the run's `predictions.npz` / `clouds.csv`, read-only, outside the repo.
  Not yet a citable result: section 3b, E1–E2 turn these into stored numbers.
- Intervals are 95% bootstrap over test θ (models) or over clouds (end to end). One training seed everywhere.

---

## 1. Data audit

### 1.1 What is current

| what | where | state |
|---|---|---|
| bank, 8 families × 50 000 θ × 2 | `data/bank/<family>/` | complete, 100 000 rows each |
| 13 classifiers, 12 × 7 estimators | `oneshot/results/default/{classify,estimate}` | all have `report.json`; all 8 families, `n_train` 592 000 |
| comparison | `…/compare/` (27 Sep) | covers 11 classifiers, 10 estimators |
| min. contrast | `…/mincontrast/` (27 Sep) | complete |
| end to end | `…/evaluation/heldout{,_mincontrast,_ph_ablation,_ph_cell}` | complete, 0 failed fits |
| power check | `cascade/scoring/results/power/` (24 Sep) | 5 of 8 families |

### 1.2 Stale

- `…/compare/` predates `fusion_curves_dtm10` and `fusion_curves_alpha_dtm10` (trained 29 Sep). So do
  `paper/tables/t3,t4,t9` and F5. The two models have no intervals and are not candidates for `best`.
- `…/evaluation/heldout` variant `best` was scored with the 25 Sep assignment (thomas, nested, matern1 →
  `nn_curves`; the rest → `hgb_classical_ph`). Compare's `best` is now different, and changes again with
  13 models [X]: thomas, matern1 → `fusion_curves_alpha_dtm10`; matern2, ring → `fusion_curves_alpha`;
  nested → `nn_curves`; lgcp → `hgb_classical_ph`; cell → `hgb_classical_ph6`.
  `paper/tables/t6_poisson_gap.tex` is built from this stale variant (`paper.yaml` `headline_variant: best`).
- T8 power (`paper/tables/t8_power.tex`): run before ring, Matérn I and cell existed; binned by δ̃, while
  the paper stratifies by regime; produced by pre-camera-ready code in `cascade/`.
- `paper/main.tex` says "11 models" / "all 11 combined 0.537" (lines 352, 326, 694; T3 hand rows). Config has 13.
- Not the paper run, 8-family mismatch or older design. Do not cite:
  `results/{classify,params}` (15 G, 5 families), `results_pre_split20k/` (17 G, old split),
  `cascade/results/`, `pilot/`, `oneshot/results/{smoke,smoke_nn}`, `extensions/results/learned/smoke`,
  `data/{simulation,cascade,pilot}`, `sanity_check_scoring/`.
- Unreferenced leftovers inside the run: `evaluation/{same,heldout_limit16}`, `train_filter/`.

### 1.3 Duplicated or inconsistent

- `classical` is scored in both `heldout` and `heldout_mincontrast`: same 480 clouds, same fits, but
  only 38% of per-cloud fit scores are identical (fits were re-simulated) [X]. Overall skill agrees
  (0.906 vs 0.907); per family it moves by up to 0.06 (cell 0.69 vs 0.63, matern2 0.77 vs 0.73).
  `extensions/results/softroute` rescored `ph` on the same 480 clouds: 0.946 vs 0.923.
  → Per-family skill at n = 60 is not a finding. Overall skill carries about ±0.02 of re-simulation noise.
- **"Wrong family, right fit: 0.91 misclassified vs 0.95 identified"** (`main.tex:427`, `outline.tex:191`).
  This is from `heldout_ph_ablation` (strong-signal strata only; 358 vs 722 clouds) [S]. On the headline
  480 clouds it is **0.73 [0.41, 0.92] vs 0.95 [0.92, 0.98]** [S, X]. See 3d, claim 6, for the defensible version.
- **"Min. contrast Thomas error 2.3× higher overall and 3.3× higher outside the regime"** (`main.tex:430`).
  2.3× = 1.619 / 0.706 vs the pipeline. 3.3× = min. contrast outside (2.116) over min. contrast inside
  τ = 0.9 (0.642). Against the pipeline outside the regime it is 2.5× (2.116 / 0.839). Reword.
- **"PH never hurts"** holds for estimator error over all test clouds (7 of 7 differences ≤ 0 or n.s.) [S].
  Exposed spots: T10's skill column prints LGCP 0.93 → 0.86, Thomas 0.95 → 0.93, ring 0.97 → 0.96.
  None is significant (LGCP Δ −0.07 [−0.29, +0.02], from 14 clouds whose call changed; with the classifier
  held fixed LGCP is 0.96) [X]. Outside the regime PH is slightly worse for Thomas (+0.002) and ring (+0.004) [X].
- Headline mismatch: text and abstract quote `ph` (0.92); `paper.yaml` `headline_variant` is `best`;
  `evaluation.main` has no `ph` variant, so T6 cannot show the headline pipeline without a `paper.yaml` change.
- Title says "Topological Deep Learning"; section 3 names trees (`hgb_classical_ph`) as the headline model.
- Repo README describes `<results>/pipeline/`; the run lives in `oneshot/results/default`
  (`results_paper/pipeline` is a symlink to it; package loaders need `CLOUDFORGER_RESULTS=results_paper`).

### 1.4 Missing

- **Seeds.** Every unit has one seed. The only seed evidence: `experimental/posterior/results/default/twin`
  retrained `nn_curves` estimators (3 seeds for 5 families; jobs cancelled 29 Sep, Matérn I and cell have 1).
  Seed s.d. of RMSE/s.d.: thomas 0.0006, lgcp 0.0006, ring 0.0017, nested 0.0019, matern2 0.0044.
  Several network-vs-network differences are this size (3a).
- **Ablation** (`configs/ablation.yaml`, 40 inputs × 3 seeds): not run; `paper.yaml` `ablation: null`;
  `main.tex:330` promises it.
- **No end-to-end score for any network that uses PH.** Pipelines scored: trees, trees + `nn_curves` (`best`, stale).
- Provenance: the 81 units from before 29 Sep have no `provenance` block (retired `oneshot/` trainer, tag
  `pre-camera-ready`). The 16 new fusion units were trained by the package at dirty `b5d617bb`.
  Cross-check: the twin `nn_curves` (package trainer) matches the paper-run units to ≤ 0.003.
- Matérn I skill is unresolved on the 480-cloud sets (every stratum); resolved on the 1 080-cloud set.
- Baselines named in an earlier outline and absent: `spatstat` `kppm`, a real-data example, a VIHRS-style
  network on the current bank (`curves_L_fixed` in the ablation config; the only runs are in the stale
  `results_pre_split20k/`).
- Not logged: inference time, PH featurization time, parameter counts, GPU model.

### 1.5 Freeze

- `results/manifest.yaml`: commit, diff hash, sha256 of every config and bank manifest, output dates, known gaps.
- The tree is dirty: `src/cloudforger/pipeline/nn.py`, `tests/test_training.py`, `configs/pipeline.yaml`
  (the two-filtration fusion) are uncommitted, so the commit hash alone does not pin the code that trained
  the newest units. **Commit before anything else is trained.** Not done here (no commit was asked for).

---

## 2. Consolidation

- `results/summary.csv`: 2 665 rows; `method, family, metric, mean, std, n_seeds` + `lo, hi, se, n, subset, source`.
  `std` is empty and `n_seeds` is 1 throughout, because there is one seed. Bootstrap intervals are in `lo, hi`,
  never in `std`.
- `results/README.md`: what was run, headline numbers, column meanings, caveats.
- `paper/scripts/summary.py`: builds the CSV, then two tables from it:
  - `paper/tables/s1_pipelines.tex`: skill and regret of the 5 pipelines, overall and per family. A start for T7.
  - `paper/tables/s2_models.tex`: all 13 models + min. contrast, accuracy and per-family error.
- Not wired into `paper/scripts/make.py` or `paper/README.md` (existing files, left untouched). One line each if wanted.
- `results/` is gitignored and also holds the stale 5-family trees; the three new files need `git add -f`.

---

## 3. Protagonist analysis

Models: **DL** = `nn_curves` (no PH), `fusion_curves_{alpha,dtm10,alpha_dtm10}` (curves + PH images),
`pi_*`, `perslay_*` (PH only). **TDA** = any PH input, in trees (`hgb_ph`, `hgb_classical_ph`, `hgb_classical_ph6`) or networks.
**DL+TDA** = the three fusion networks.

### 3a. Where they already win or tie

**DL**

| regime | result | tag |
|---|---|---|
| estimation, overall | a network is best on val in 5 of 7 families (thomas, nested, matern2, ring, matern1); trees keep lgcp, cell | [X] |
| estimation, vs best trees | `fusion_curves_alpha_dtm10` − `hgb_classical_ph`: thomas −0.006, nested −0.009, ring −0.003, matern1 −0.003 (all sig.); matern2 tie; lgcp +0.002, cell +0.003 (sig. worse) | [X] |
| classification, overall | tie with the best model: `fusion_curves_alpha_dtm10` 0.5335 vs `hgb_classical_ph` 0.5339, Δ −0.0004 [−0.0018, +0.0009] | [X] |
| classification, in regime τ = 0.9 | best model: `fusion_curves_dtm10` 0.654, `fusion_curves_alpha_dtm10` 0.653, vs trees 0.646; Δ +0.007 [+0.005, +0.009] | [X]; `fusion_curves_alpha` 0.652 in [S] |
| calibration | ECE 0.003–0.006 for curve networks vs 0.010–0.016 for trees. NLL is a tie (1.084 vs 1.083; `hgb_classical_ph6` 1.074 is best) | [X] |
| CSR filter | networks call Poisson correctly more often (recall 0.79–0.81 vs 0.73), so fewer false structures | [S] |
| vs min. contrast | curve and fusion networks beat it in every family (e.g. thomas 0.70 vs 1.62). PH-only networks lose to it on both Matérn families (PersLay 0.18 vs 0.15, 0.23 vs 0.19) | [S] |

Where DL does not win. Do not claim:
- Outside the regime networks are worse classifiers (−0.013 to −0.020) [X]. Consistent with "sent to Poisson is free", not a win.
- PH-only networks (`pi_*`, `perslay_*`) are below classical trees everywhere except cell.
- Runtime: training is 6–43× the trees' (0.12 h for `hgb_classical_ph` vs 0.8–5.2 GPU-h) [S]. Inference not logged.
- Low-count patterns (lowest n̄ tercile): no DL advantage [X]. Sample efficiency and robustness: not run.
- Variance: unknown for fusion (one seed). Differences under ~0.005 between two networks are within seed noise.

**TDA**

| regime | result | tag |
|---|---|---|
| cell (K is exactly Poisson's) | error −15% in trees (0.142 → 0.120, Δ −0.021 [−0.023, −0.020]); −20% in networks (0.154 → 0.123, Δ −0.030 [−0.032, −0.029]) | [S], [X] |
| cell, end to end | skill 0.92 → 1.00 on 1 000 clouds (k ≥ 5), Δ +0.08 [+0.03, +0.13]; same with the classifier held fixed | [S], [X] |
| cell, classification at ambiguous k (3–4) | recall 0.52 → 0.59 with PH in trees | [X] |
| which filtration | DTM inputs carry cell (`fusion_curves_dtm10` 0.124, `perslay_dtm10` 0.129); alpha inputs do not (`fusion_curves_alpha` 0.138, `perslay_alpha` 0.159, `pi_alpha` 0.161); classical alone 0.142 | [S] |
| trees, other families | PH added to classical: sig. better in nested −0.004, matern2 −0.001, ring −0.007, matern1 −0.001; n.s. thomas, lgcp; never worse | [S] |
| trees, ring in regime τ = 0.9 | −0.020 (0.518 → 0.498, −3.8%) | [S] |
| networks, other families | PH images added to curves: better in thomas −0.004, matern2 −0.004, ring −0.006; worse in nested +0.004; tie lgcp, matern1 | [X] |
| networks, in regime τ = 0.9 | thomas −0.032 (0.465 → 0.433, −7%), ring −0.017 | [X] |
| classification | +0.005 [+0.003, +0.006] in trees; +0.007 in regime for networks | [S], [X] |
| where min. contrast is blind | it never calls cell (recall 0.00) and scores −0.30 there | [S] |

Where TDA does not win. Do not claim:
- PH alone is worse than classical in 6 of 7 families (`hgb_ph` +0.006 to +0.023). PH is a complement, not a replacement.
- End to end outside cell, PH changes nothing resolvable: `ph` − `classical` Δskill +0.016 [−0.016, +0.058] (480 clouds),
  −0.005 [−0.03, +0.02] (1 080 clouds) [X].
- Effects outside cell are under 1% of the error except ring (1–4%) and Thomas in regime (networks).
  Matérn II and nested effects in networks are at seed-noise level.
- Six filtrations are no better than two (`hgb_classical_ph6` vs `hgb_classical_ph`), except cell (0.117 vs 0.120).

### 3b. Experiments, ranked by payoff × cheapness

Costs from `train_seconds` [S] and the dated outputs. All need your go-ahead (they write results or edit configs).

| # | experiment | cost | payoff | risk |
|---|---|---|---|---|
| E1 | Rerun `scripts/compare.py` with 13 models, rebuild T3/T4/T9/F5 | ~2 min CPU. Overwrites `compare/` and changes `best` | intervals for DL+TDA; `best` becomes DL+TDA in 4 of 7 families | none; `evaluation/heldout` `best` is already stale |
| E2 | End-to-end set with network pipelines: `fusion` (classifier + estimators `fusion_curves_alpha_dtm10`), `curves` (`nn_curves` both), `ph`, new `best`; on the 480 headline clouds | new `evaluation.sets` entry; ~1 h, 32 CPUs | **the only way the abstract's "neural networks … combine" gets a headline number** | score resolves Δskill ≳ 0.04; expect a tie with `ph` at 0.92–0.93, which is enough |
| E3 | Same variants on the cell k ≥ 5 clouds (1 000), classifier fixed | ~1.5 h, 32 CPUs | "topology carries signal where K is blind, in trees and in networks": likely 0.92 → ~1.00 again | low; estimator gap is large (−20%) |
| E4 | 3 seeds of `nn_curves`, `fusion_curves_alpha`, `fusion_curves_alpha_dtm10` | ~26 GPU-h as an array; 2–5 h wall | turns PH-in-DL effects into seed-backed claims; fills `std`, `n_seeds`; answers the checklist's error-bar item | matern2 and nested effects may vanish; thomas, ring, cell should hold |
| E5 | Power check on all 8 families, by regime stratum (`scripts/power.py`) | ~30 min, 16 CPUs | not a protagonist result, but T8 is a hole a reviewer will find (nested lowest bin: d′ −0.55) | may show weak power for Matérn I |
| E6 | Which PH features carry cell: permutation importance on `hgb_classical_ph` (stored model + tables) | minutes, CPU; needs a small paper script | one interpretability figure; backs "DTM sees local density" | low |
| E7 | Family-vs-family confusion against the limit parameters (ring σ/ρ, nested σ₁√κ, Matérn R) from stored predictions | minutes, CPU; no training | backs abstract claim 5 quantitatively | low |
| E8 | Low-data curve: train at 1/4 and 1/16 of the train θ, 4 models | ~3 GPU-h + CPU; needs fixed-test handling (`thin` also thins test) | sample-efficiency story | DL likely loses to trees at low data; PH may help. Coin flip |
| E9 | Full single-model ablation (`configs/ablation.yaml`) | ~960 units, order 300+ GPU-h | appendix completeness, VIHRS-style baseline | too slow for today |
| E10 | Noisy / thinned test clouds | re-featurize perturbed clouds (PH + curves), hours | robustness story | speculative |
| E11 | Leave-one-family-out generalisation | new design, retrain | speculative | high |
| — | PH summaries fed to the network, or stacking | — | classification is at a ceiling: 13-model mean posterior 0.539 [X] | not worth it |

Suggested order today: E1 → E2 + E3 + E5 in parallel → E4 in the background → E6, E7 while writing.

### 3c. Framing that holds on current numbers if experiments fail

1. **Topology where second-order statistics are blind.** Cell is invisible to K by construction; min. contrast
   never detects it. PH cuts its error 15–20% in both learners and lifts end-to-end skill 0.92 → 1.00.
   Add the filtration finding (DTM yes, alpha no). Strongest, fully supported.
2. **Topology-aware learning is learner-agnostic and free.** Adding PH never significantly worsens an estimator
   over the test set (trees, 7 of 7), helps in 5 of 7, and the same holds in networks for 4 of 7 (one family slightly worse:
   say so). Frame as "a new lens that complements second-order summaries", not as a replacement.
3. **DL as a calibrated CSR filter that matches trees.** Networks tie the best classifier overall, lead where
   the family is detectable (τ = 0.9), are the best-calibrated models, and give the best estimator in 5 of 7 families.
   "Matches or exceeds" is supported; "outperforms" overall is not.

"First to…" claims (first unified 8-family pipeline with PH, first proper-score evaluation on replicates) cannot
be checked from the repo. They need a literature pass before they go in (`main.tex:79` is still TBD).

### 3d. Claims in the submitted abstract (`paper/main.tex:33–48`)

| # | claim | status | evidence / what is needed |
|---|---|---|---|
| 1 | "a unified pipeline for these tasks across multiple SPP families" | **supported** | 8 families, one classifier + per-family estimators |
| 2 | "neural networks trained on simulated patterns combine classical spatial summary statistics with topological features" | **needs experiment** (E1, E2) | such networks exist and are scored as components; the headline pipeline and every end-to-end number are trees. Without E2, the body must say "learned models (trees and networks)" |
| 3 | "these features improve estimation performance" | **needs framing** | trees: sig. in 5 of 7, large only for cell (−15%); networks: 4 of 7, one worse, one seed (E4). State per family; never "across the board" |
| 4 | "state-of-the-art results on classical benchmarks" | **needs framing**, ideally a baseline | one comparator, our own min. contrast (tuned on val): selection 0.53 vs 0.33; error 1.2–4.6× lower in all 7 families; skill 0.92 vs 0.84, Δ +0.08 [+0.03, +0.15]. No external benchmark, no `kppm`, no composite likelihood, no VIHRS network. Write "outperforms minimum contrast on eight classical families" |
| 5 | "characterizing regimes in which different point process families are indistinguishable" | **needs framing** (E7 helps) | characterized: indistinguishability from Poisson (one coordinate, AUC 0.87–0.96 vs 0.88–0.96 from all parameters; T5, F4). Family-vs-family confusion is shown (F3) but not characterized |
| 6 | "even in these ambiguous regimes, our pipeline can generate new realizations that preserve the statistical properties" | **needs framing** | below the boundary the true model beats a Poisson fit in 0 of 14 strata (max z = 1.4, n = 20 each); structured clouds sent to Poisson: gain +5e−5 ± 5e−5 (n = 102). Misclassified clouds score 0.73 vs 0.95 identified, but routing them to the true family gives only 0.84 and the difference is n.s. (z = 1.4). Say "the true family would not do significantly better", not "as good as identified". "Statistical properties" = local configurations within 1.5 mean spacings. Power caveat stands until E5 |
| 7 | "a practical framework" | **needs framing** | no real-data example, no runtime numbers. Keep it as a closing phrase, claim nothing specific |

---

## 4. Paper plan

Not started (stop after task 3). Inputs already settled here: displays that are stale (1.2), the headline
decision (`ph` vs `best` vs a network pipeline, pending E2), and the claim wording in 3d.

---

## 5. Controls on the kernel score (2026-09-30)

Kernel score only, seed 0 reproduces the stored oracle/CSR/fit scores exactly (max diff 0.0). Code and outputs
are in `/gpfs/scratch/qp252676/globus/scorecheck/` (outside the repo, not committed; `analysis.out` has every table).

**1. Wrong-model ladder** (2 100 clouds, 100 per regime stratum; regret ×1e-4; failed fits excluded, ≤ 13% at 2 sd).
Perturbation = independent Gaussian noise on log θ, in units of the family's test s.d. (the RMSE/s.d. unit).

| model | strong strata (gain 56.0, z 13) | middle (gain 1.14, z 4) | weak (gain 0.13, z 0.9) |
|---|---|---|---|
| noise 0.1 sd | 0.4 (z 0.8) | 0.3 (1.7) | 0.1 (0.6) |
| noise 0.25 sd | 7.7 (5.3) | 0.5 (2.7) | 0.1 (0.6) |
| noise 0.5 sd | 28.6 (8.9) | 3.6 (5.8) | 0.4 (1.7) |
| noise 1 sd | 94.3 (8.9) | 38.9 (5.6) | 3.7 (4.4) |
| constant guess (train median) | 40.4 (11.1) | 5.2 (9.0) | 6.5 (11.1) |
| random θ, same family | 60.8 (10.2) | 62.0 (7.5) | 64.8 (7.9) |
| random other family | 93.7 (13.4) | 57.1 (8.0) | 70.7 (7.0) |

- Regret rises monotonically with error. The score notices a 0.25 sd error in strong strata (z 5), not 0.1 sd.
- It also separates wrong models from the truth in weak strata (the constant guess: z 11), so it is not noise there.
- Family-dependent: Matérn I is nearly blind to parameter error (0.5 sd: z 0.2; gain only 3.5e-4); cell and Matérn II need ≥ 0.5–1 sd.
- Skill calibration in strong strata: 1 − regret/gain = 0.99 (0.1 sd), 0.86 (0.25), 0.49 (0.5), −0.7 (1 sd), 0.28 (constant guess).
  Pipelines score 0.92–0.96 yet have RMSE/s.d. 0.4–0.7: their errors sit in directions the score does not punish. So 0.92 must not be read as "parameters nearly right".

**2. Minimum detectable gain** (oracle − CSR; 100 clouds per stratum; MDG = 2·sd/√n).
- Below 0.5 (6 families): all |z| ≤ 0.8. Gains above about 0.7–1.0e-4 excluded at n = 100 (0.7–6% of the family's strong-stratum gain; Matérn I 27%).
- **0.5–0.9 band: LGCP (z 2.75, gain 2.2e-4) and ring (z 2.72, 3.6e-4) are positive.** About 2–3% of the strong-stratum gain. This replaces "0 of 14 strata" (n = 20) in claim 6 and in `main.tex`.
  With 14 strata about 0.7 z > 2 are expected by chance; both hits sit in the transition band, where a small gain is expected.
- Cell: k 2 z 1.8 (0.6e-4), k 3–4 z −0.1. Thomas 0.5–0.9 z 1.8 (1.3e-4). Nested, Matérn I/II: nothing.
- At n = 20 (the stored sets) MDG is 3–6e-4 in the middle band, so the earlier null was underpowered there.

**3. Repeatability** (same 480 clouds, same fits, 5 simulation seeds; seed 0 = stored).

| pipeline | seed 0 | mean of 5 | s.d. over seeds |
|---|---|---|---|
| oracle_ph | 0.941 | 0.957 | 0.017 |
| best | 0.931 | 0.943 | 0.014 |
| fusion | 0.957 | 0.941 | 0.016 |
| ph | 0.925 | 0.940 | 0.016 |
| classical | 0.913 | 0.918 | 0.016 |
| curves | 0.923 | 0.915 | 0.009 |
| mincontrast | 0.846 | 0.853 | 0.010 |

- Overall skill moves ±0.015 from re-simulation alone. Per-family skill: Matérn I 0.4–0.7, Matérn II 0.15–0.23, cell 0.15–0.27, LGCP 0.05, Thomas 0.02–0.06, nested 0.01, ring 0.02–0.03. **Do not report per-family skill for Matérn I, Matérn II, cell.**
- Paired differences over the 5 seeds (mean, range): ph − mincontrast +0.087 (+0.076..+0.096); classical − mincontrast +0.065 (+0.046..+0.078);
  ph − classical +0.023 (+0.012..+0.031); fusion − curves +0.025 (+0.007..+0.036); oracle_ph − ph +0.016 (+0.009..+0.024);
  **fusion − ph +0.000 (−0.013..+0.032): the 0.96 vs 0.92 seen in the `networks` set was seed luck.**
- These cover simulation noise only; the bootstrap over clouds (earlier interval for ph − classical [−0.016, +0.058]) is separate and still applies.
- Report skill as the mean over seeds. Within-cloud regret s.d. over seeds is 2.7e-4 against 8.1e-4 across clouds.
