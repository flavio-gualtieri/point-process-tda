# Experiments still to run

Working note, not part of the paper. Every item here is a run that has not happened on the **v2** bank
(`data_v2/`, `results_v2/pipeline/`, `configs/v2/`); the paper's prose was written so that none of them is load
bearing for a claim it makes. Where a v1 number exists it is named, because that is what the text's qualitative
statement rests on and what the rerun is expected to reproduce.

Status keys: **blocking** = a claim in `main.tex` depends on it; **supporting** = strengthens a claim that already
stands on other evidence; **optional** = would be a new result.

---

## 1. Single-model ablation — submitted

`configs/v2/ablation.yaml` over `configs/ablation.yaml`. 40 input representations × 3 seeds × (1 classifier +
7 estimators) = **960 GPU units**, plus 40 input-cache prebuilds.

Axes: 6 filtrations (rips, alpha, DTM$_{k}$ for $k\in\{5,10,15,20\}$) × {H0, H1, H0+H1} × {persistence image,
PersLay} = 36 PH inputs, and 4 summary-curve inputs (`curves_L_fixed` is the VIHRS-style baseline).

- **Status:** supporting. `main.tex` promises it in the appendix; `paper_v2.yaml` sets `ablation: null`, so every
  display that needs it is skipped and no current claim rests on it.
- **Gives:** which filtration and which homology dimension carry each family, on a grid far wider than the
  three-point contrast in Section 6 (Table `tab:filtrations`); whether PersLay or persistence images win; a
  seed s.d. for every cell, which is what sets the "differences under 0.005 are seed noise" caveat.
- **After it lands:** set `ablation:` in `paper_v2.yaml`, run `scripts/compare.py --config ablation.yaml`, then
  `paper/scripts/summary.py` and `make.py`.

## 2. Out-of-distribution end-to-end skill

Score Matérn I fits, not just the family assignment. `scripts/endtoend.py` needs to score a family the run never
trained on; Matérn I is simulable, so its oracle and CSR references exist.

- **Status:** optional. Section 6's "Out of distribution" paragraph now reports the assignment and says why, so
  this is a question about library coverage rather than a hole.
- **Gives:** how much fidelity a nearest-mechanism substitution actually costs.

## 3. Wrong-model ladder (dose) on v2

`scripts/mincontrast.py`-adjacent score check; `paper_v2.yaml` has `dose: null`. The v1 result is
`results/frozen/scorecheck/dose_seed0.csv` and `paper/tables/sA_score_ladder.tex`.

- **Status:** supporting. Appendix C.3's statement is written from the v1 ladder and attributed to it: regret rises
  monotonically with $d$ in all three strata, every rung from $d=0.25$ up resolves in the strong stratum
  ($z\ge5.3$), only $d\ge1$ resolves in the weak one.
- **Gives:** the same table on the v2 bank, with Strauss in and Matérn I out.
- **Note:** do not reuse the v1 clause "the wrong-family rung is the worst" — it is false even on v1, where
  `noise_2.0` regret (184.2) exceeds `wrong_family` (93.7) in the strong stratum.

## 4. Five-seed rescoring — submitted

`paper_v2.yaml` `repeat: null`. v1 had 5 simulation seeds on 480 clouds; v2 is scored at one seed.

- **Status:** **blocking**, upgraded from supporting. `tab:families-skill` is a MAIN BODY table reported at one
  simulation seed with intervals over clouds only. On v1 per-family skill moved 0.15-0.7 across seeds at 60 clouds
  per family, and the main text's one marginally-resolved per-family cell is Thomas, +0.10 [+0.01, +0.26] --
  smaller than that movement. `summary.py` hard-codes a refusal to print per-family skill for this reason while
  `main.tex` prints it, so the internal doc is stricter than the paper.
- **Gives:** re-simulation noise on overall skill (v1: about ±0.02), and the `±` seed column in `s5_fidelity` via
  `summary.py`'s `seeded()` path.
- **Submitted** as `slurm/run_rescore_v2.sh`: sets `fidelity_seed1..4` in `configs/v2/pipeline.yaml`, 12 shards
  plus a merge each, CPU only. `fidelity` itself is seed 0, so the four complete the five.
- **Plumbing added:** `evaluation.seed` was global, read at `scripts/endtoend.py` from the top-level block rather
  than the set. A set can now override it. The two seeds are different things and must not be confused:
  `clouds.seed` picks WHICH clouds and is held at 0 so the five sets stay paired; `seed` reseeds the simulations.
  Verified on a 6-cloud run: identical `scored_case_id` set across seeds, oracle scores differing, mean |delta| of
  the per-cloud regret 7.2e-4 -- larger than the headline mean regret (~3e-4), which is the concern in one number.
- **After it lands:** the five `clouds.csv` need names matching `summary.py`'s `repeat` glob, which parses the seed
  from the filename (`p.rsplit("seed", 1)[1]`). Copy or link them to e.g. `results/v2/scorecheck/repeat_seed<i>.csv`
  and set `repeat:` in `paper_v2.yaml`, then rerun `summary.py` and `make.py`.

## 4b. Training-seed noise on the fusion - curves contrast — submitted

`configs/seedcheck.yaml` (+ `configs/v2/seedcheck.yaml`), both networks over seeds {1,2,3}: 2 models x 3 seeds x
8 tasks = **48 GPU units**. Writes `results_v2/seedcheck/`.

- **Status:** **blocking**. The introduction's TDA contribution claims persistent homology "never makes estimates
  significantly worse". At one training seed that is false on v2 -- Matern II is resolved worse in all three
  subsets (+0.0031 [+0.0023, +0.0038] on all patterns, +0.0135 at tau 0.5, +0.0060 at 0.9) and Strauss pooled over
  all patterns (+0.0076 [+0.0068, +0.0084]). Those intervals cover test-theta sampling only.
- **Gives:** the seed s.d. that decides whether the claim stands with an interval or has to be reworded. The
  paper's own caveat puts seed noise near 0.005, which straddles the smallest of those gaps but not the largest,
  so a reword is the likely outcome for the in-regime figures.
- **Note:** the single-model ablation does NOT cover this. It trains the 40 single-input models, never the fusion
  network. Its `curves_LFGJ_fixed` is spec-identical to `nn_curves`, so it does give the no-PH arm at 3 seeds, but
  in a different run directory; this config puts both arms in one run so `compare` pairs them seed by seed.
- **Submitted** as `slurm/run_seedcheck_v2.sh`.

## 5. Score power check on v2

`paper_v2.yaml` `power: null`; `t8_power` is skipped by `make.py`. v1 covered 5 of 8 families.

- **Status:** supporting. Appendix C's propriety argument is analytic; this is the empirical $d'$ companion.
- **Gives:** T8, the only display in the README's table with no v2 counterpart.

## 6. Cell evaluation set

`evaluation.networks_cell` is absent from `paper_v2.yaml`, so `s6_topology`'s "skill, cell $k\ge5$" row is `---`
and `story.cell_pipelines` is `{}`.

- **Status:** supporting. The cell claim in Section 6 rests on estimation error (0.143 against 0.173), which is
  current; this would add the end-to-end half.

## 7. PH ablation evaluation sets

`evaluation/ph_ablation` and `ph_cell` exist in `run_all.sh` but were never run on v2.

- **Status:** optional, and largely superseded. The `fidelity_curves` and `fidelity_filtrations` sets now cover
  the same ground with proper pairing on the same 1 440 clouds.

---

## Not experiments — writing and plumbing

These are the remaining `\NI`/`\review` markers in `main.tex`. None needs a run.

`main.tex` now has **no `\TBD` and no `\NI` markers**. What is left is not marked in the source:

| Where | What |
|---|---|
| F5 | `f5_main.pdf` is generated every `make.py` run but has no `\includegraphics` or `\label` in `main.tex` |
| T4, T5, T6, T10 | generated into `paper/tables/v2/` and current, but not `\input` anywhere |
| `tab:families` | hand-built; the README's note to re-check its $K$ cells against the code is still open |
| `compare/regime.md` | stale (Oct 1 11:39), computed on a superseded cutoffs axis; `regime_size0.05.md` is the live one that `tab:regime` is built from. Re-run or delete so the two cannot be confused |
| Bibliography | the draft has no `\bibliography`, and one plain-text attribution each to Adams et al. (2017), Gneiting & Raftery (2007) and Gretton et al. (2012) is written out in prose rather than cited. Related Work and Limitations are not written |
| Page count | `RESTRUCTURE_NOTES.md` measured the draft at about 0.8 page over 8, before the Intro becomes prose and Related Work/Limitations are added. The analysis section is now 309 words lighter, which pays for part of it |

Closed since that note was written: the Figure 2 "PHNet-Fusion" label (renamed in `scripts/f2_pipeline.py` and
regenerated), the F2 include path (now `figs/v2/`), the §5 training sentences, and all of Appendix B.
