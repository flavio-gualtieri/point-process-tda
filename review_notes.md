# Investigation notes for `review_main_project.tex`

## 1. Headline claims in `short_report.tex` — verified against the repo

Spot-checked against `results/experiments.jsonl` (per-seed `test_loss` in the
same log-normalized units used throughout) and individual `results.json`
files. Everything below **matches** what's in the current (uncommitted)
`writeup/short_report.tex`:

- **Thomas, `mincontrast` (K):** raw seeds `[0.365, 0.379, 0.379, 0.381,
  0.396, 0.399, 0.903, 1.369, 1.925, 2.056]` → mean `0.855±0.682`. The
  "6/10 seeds land at 0.365–0.399, the other 4 at 0.903–2.056" claim in
  §3.4 (ssec:results) is exact, not approximate.
- **Thomas, `mincontrast_g`:** seeds in `[5.76, 6.66]`, mean `6.20±0.34` — matches Table 1.
- **Thomas `pi_multik`/ParamNet, `vihrs`, `vec_multik`/`betti_multik`
  (vectorization ablations), `logn_only`:** aggregate values in
  `results/experiments.jsonl` are consistent with Tables 1–3.
- `writeup/short_report_pct_errors.tex` (untracked, clearly generated very
  recently from the same numbers to add a %-error column) reproduces
  Table 1/2's $\mathcal{L}$ values exactly — independent confirmation
  they're current, not stale copy-paste.
- `sigma_pixels=0.5`: I initially flagged a possible conflict with memory
  saying "fixed sigma_pixels=2.0" was the resolution of the adaptive-σ
  investigation (2026-07-23). Resolved: `2.0` is the *class default* in
  `persistence_image.py`; the actual training configs
  (`configs/runs/thomas/thomas_pi_multik.yaml`,
  `configs/runs/nested_thomas/pi_multik.yaml`) explicitly override it to
  `sigma_pixels: 0.5`, chosen by the later calibration sweep the report
  describes. No discrepancy — the report is right.

## 2. Figures and tables on disk (concrete paths)

- `figs/calibration_comparison.pdf` — naive vs. coverage-quantile
  calibration, used as Fig. 1 in `short_report.tex`. Exists, current.
- `figs/vectorization_schematic.pdf` — one cloud → PD → the five
  vectorizations compared in Table 3 (tight-cluster vs. near-Poisson rows).
  Exists, current.
- No figure on disk reproduces Table 1/2 as a plot with the *current*
  n=10-seed numbers. `docs/figures/overall_comparison.pdf`,
  `thomas_per_target.pdf`, `nested_thomas_per_target.pdf` (dated Aug 10)
  belong to a different, 3-seed comparison (see §4 below) and would
  introduce arms (`mph_pi`, `mph_fusion`) `short_report.tex` never
  mentions — I have **not** used these.

## 3. Error floor / label-indexing bug — still unresolved

- `scripts/estimate_error_floor.py`, `configs/runs/{thomas,nested_thomas}/error_floor.yaml`,
  and matching `slurm/error_floor_*.sh` exist (added in commit `6e6d0d8`,
  "slurm scripts").
- **No output exists anywhere** — no `data/error_floor/`, no result JSON,
  nothing under `results/`. Per `writeup_new.tex` (lines ~1210–1233, not
  carried into `short_report.tex`), a prior run on a 50θ×200-realization
  Thomas dataset completed all 10 SLURM shards without crashing but every
  per-cloud loss reduced to `NaN`; the suspected cause is that the shard
  JSONs' `label_names` included `edge_buffer`/`c1` — fields that don't
  belong to Thomas's `(κ,μ,σ)` — i.e. the per-target loss array is indexed
  against the wrong label list. This has **not been fixed or re-run**
  since. There is no floor number to report, planned only.

## 4. Newer-than / inconsistent-with `short_report.tex` — please read this section

**(a) Nested-Thomas minimum contrast was actually run, and the report says it wasn't.**
Table 1's footnote $^d$ reads *"Not run: no closed-form summary statistic
for nested Thomas has yet been derived, so its absence is an outstanding
experiment rather than an impossibility."* That's now stale:

- Commit `447969c` ("rips results", same day as the rest of the current
  numbers) added `configs/runs/nested_thomas/{mincontrast,mincontrast_g}.yaml`,
  `src/cloudforger/baselines/mincontrast_{nested,g_nested}.py`, and SLURM
  scripts, implementing the closed-form $K$/$g$ derived in
  `writeup_new.tex` (`eq:g-nested`).
- `results/experiments.jsonl` shows both were run, 5 seeds each
  (9371–9375), on 2026-08-21:
  - `mincontrast_nested` (K): losses `[394.5, 74.9, 65.6, 79.6, 280.9]`,
    mean **179.1 ± 134.5** — two orders of magnitude worse than chance
    (chance ≈ 1.0).
  - `mincontrast_g_nested` (g): losses tightly clustered at
    `[14.5, 14.8, 14.9, 15.1, 14.7]`, mean **14.81 ± 0.19** — an order of
    magnitude worse than chance, but *stable* across seeds (unlike K).
  - The per-seed `results.json` files themselves live only on the HPC
    scratch path recorded in the log (`/gpfs/scratch/.../results/...`);
    only the summary JSONL rows synced back to this repo, so I can't
    pull per-parameter breakdowns for these two arms.
- **Important caveat, from the module itself**
  (`src/cloudforger/baselines/mincontrast_nested.py` docstring): *"NOT yet
  validated end-to-end against ground truth... the derivation needs
  independent checking before either claim is relied on."* You flagged
  this yourself in the code before I found it. So the numbers are real
  runs, but their author-attached status is "unvalidated, possibly a bug
  in the closed form or the fit, not a trustworthy finding yet."
- **My recommendation:** don't report the raw 179× / 15× numbers as a
  finding ("minimum contrast collapses on nested Thomas") — the code
  itself says not to trust them yet. But the footnote's *"not run"* is no
  longer accurate either. I'd suggest wording it as *"attempted but not
  yet validated"* rather than either extreme. **Your call — flagging
  rather than deciding.**

**(b) A separate multiparameter-persistent-homology (MPH) side-experiment exists and isn't in `short_report.tex` at all.**
`docs/mph_comparison_report*.tex` (commits `860a747`, `12f4c26`, Aug 10)
describe a 3-seed comparison adding two more arms — `mph_pi` (a genuine
joint density/radius bifiltration image) and `mph_fusion` (late-fusion of
`vihrs`'s $L(r)-r$ branch with `mph_pi`) — against `vihrs` and `pi_multik`.
Notably `mph_fusion` *beats* `vihrs` on Thomas (0.113 vs. 0.115) in that
3-seed run, which `short_report.tex`'s framing ("vihrs wins on every Thomas
parameter") doesn't reflect. This is real work, but it's a distinct
architectural thread `short_report.tex` never adopted into its own
narrative (different vectorization idea, different report, only 3 seeds).
**I have left this out of the draft** — it would require re-litigating the
"vihrs wins on Thomas" claim and doesn't fit the compressed, single-method
pitch the progression report wants. Flagging in case you want a one-line
mention of it under future directions instead.

**(c) "Matérn cluster process" (mentioned in `short_report.tex`'s simulator
section as a hypothetical third kernel) and "Matérn" as actually
implemented in the repo are two different processes.**
`short_report.tex` §3.1 lists the Matérn *cluster* process (offspring
uniform on a disk of radius R around each parent — a genuine Neyman-Scott
family member) as an example of the simulator's generality. What's
actually implemented under `src/cloudforger/data_generation/point_processes/matern.py`
is `MaternHardCoreProcess` — a Matérn Type II **hard-core/repulsion**
process (dependent thinning), which is not a Neyman-Scott process at all.
Only a data-generation report exists for it
(`docs/matern_data_report.md`); there's no training or evaluation run,
under either interpretation of "Matérn." `short_report.tex` never actually
claims Matérn results (only mentions it as a kernel example), so there's
no live inconsistency to fix — just flagging so I don't accidentally
imply Matérn coverage in the compressed version.

## 5. Process coverage, current state

| Process | Simulated | ParamNet trained/eval | vihrs trained/eval | Classical baseline |
|---|---|---|---|---|
| Thomas | yes | yes (10 seeds) | yes (10 seeds) | mincontrast K + g, both run, matches report |
| Nested Thomas | yes | yes (main arm n=3 at current calibration; earlier calibration n=10 for ablations) | yes (10 seeds) | run but unvalidated (§4a) |
| Matérn (hard-core, not cluster) | data only | no | no | no |

## Proposed outline (for your go-ahead)

1. **Motivation and problem** (~1 p): scientific meaning of Neyman-Scott
   parameters (galaxy clustering / halo occupancy, BCI tree dispersal
   example), single-realization setting, the three classical-estimator
   limitations compressed to one paragraph (no closed form / regularity;
   minimum-contrast hyperparameter sensitivity, Waagepetersen & Guan
   $r_{max}$/$q$; near-Poisson instability, Thomas $K$-degeneracy
   equation).
2. **Approach** (~1–1.5 p): persistence images as a process-agnostic
   descriptor (minimal PH intuition, no definition environments except
   maybe the persistence-image sentence); $(\lambda,\kappa,\nu)$
   reparameterization and deliberate near-Poisson retention (2–3 sentences
   on the simulator); ParamNet in brief (pipeline eq., DTM filtration
   rationale, CoordConv rationale, one sentence on calibration).
3. **Results** (~1 p): what's compared, headline numbers, honest
   loss breakdown (PI signal vs. label-shuffle ceiling and vs. $\log N$;
   vihrs beats us on Thomas and on both nested-Thomas scale params; our
   edge is confined to nested-Thomas density/count targets; $\kappa$
   hardest/scale easiest as a problem property; minimum-contrast margin
   decomposes into K-seed-instability + g-exponent bias). Nested-Thomas
   classical baseline mentioned per your call on §4a above.
4. **Limitations and future directions** (~0.5 p): bullets as specified —
   universal encoder; inhomogeneous/non-stationary (two-step comparator
   needs covariates+correct intensity, ours needs neither); real data
   (galaxy catalogues, BCI).

**Figure/table budget (≤2 figs, ≤1 table, both must exist on disk):**
- Fig. 1: `figs/vectorization_schematic.pdf` — doing double duty as PH
  intuition-builder (near-Poisson vs. tight-cluster clouds → diagrams) and
  as the results-section anchor for the vectorization ablation. Placed in
  Approach or Results, whichever reads better once drafted.
- Table 1: a condensed merge of `short_report.tex`'s Tables 1+2 — aggregate
  $\mathcal{L}$ per method per process, plus the per-parameter breakdown
  collapsed into prose (e.g. "vihrs beats ParamNet on all three Thomas
  parameters") rather than a second table, to stay inside the page budget.
- I'd drop `figs/calibration_comparison.pdf` from this version — it's a
  real contribution but requires diagram literacy the "no PH background"
  instruction argues against spending a figure on; happy to add it back as
  Fig. 2 if you want the calibration point kept visual rather than prose.

## Open items needing your call before I draft

1. **§4a wording** — "attempted but not yet validated" vs. keep "not run"
   vs. something else, for the nested-Thomas classical-baseline gap.
2. **§4b** — leave the MPH side-experiment out entirely (my default), or
   want one sentence on it in future directions?
3. **REF placeholders** (`short_report.tex` lines 81, 89 — "SPPs REF" for
   galaxy clustering, "REF" for halo occupancy models constraining
   $(\kappa,\mu,\sigma)$): no version of the writeup ever filled these in,
   and I don't have a clear source to attribute. I'll mark both with a
   visible `\todo{}` rather than guess a citation — flagging in case you
   already know what belongs there.
4. **Figures** — confirm the vectorization-schematic-only default above, or
   ask for calibration_comparison back in.
