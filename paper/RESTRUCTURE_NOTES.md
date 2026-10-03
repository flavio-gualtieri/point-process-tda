# Restructure notes: `main.tex` → Sikorski-style layout

Branch `restructure-sikorski`. The pre-restructure source is saved at
`paper/backup_pre_restructure/main.tex`; diff against it to see every edit.

**Conventions used**

- **Run-in headers.** Every main-text division below a section is now a bold run-in `\paragraph{…}` ending in a period. There is exactly one `\subsection` in the main text, 5.1 Goodness of Fit. It is there because the score is a contribution and the title of Section 5 does not name it.
- **Note marker.** The source already defined `\NI{}` ("[NEEDS INPUT: …]", red) and never used it. Per the brief's rule 2 (use an existing convention), every note I inserted is an `\NI`, so `grep -n '\\NI{' main.tex` finds all of them. Your own `\WIP`/`\TBD` markers are untouched and stay distinct.
- **Sentence openers.** One-sentence openers or transitions were added where a section lacked one. Each is listed in the move map as *new sentence*. No new passage is longer than one sentence, except the closing summary in Section 6, which the brief asked for.

---

## 1. Move map

Old numbering is the pre-restructure draft. "Verbatim" means the sentences are unchanged apart from `\ref` targets.

| Old location | New location | Action |
|---|---|---|
| Abstract | Abstract | untouched |
| 1 Introduction (bullets) | 1 Introduction | untouched; its `\ref`s now resolve to 5.1, 6, 4 and 6, 6 |
| 2 heading + 2.1 Spatial Point Processes | 2 Background (opening) | moved verbatim; *new sentence* opener ("We fix notation, …") |
| 2.1, paragraph on K/L/F/G/J | 2, **Summary statistics.** | moved verbatim; run-in header added |
| 2.2 Families (intro paragraph) | 2, **Families.** | moved verbatim; subsection → run-in (label `sec:families` kept) |
| 2.2 Poisson … Cell, Limits between families | 2, same run-ins | moved verbatim |
| Table 1 | Table 1 (Section 2) | placeholder caption replaced (see §3 for wording rationale) |
| Figure 1 | Figure 1 (Section 2) | placeholder caption replaced |
| 2.3 Data Generation (opening, Priors, Size and splits) | 3 Data Generation | moved verbatim; "(Sections 4 and 5)" → `(Section~\ref{sec:wall})`, because both labels now resolve to Section 6; *new sentence* pointer to Appendix A at the end of Size and splits |
| 2.4 Goodness of Fit (opening) | 5.1 Goodness of Fit (opening) | moved verbatim; *new sentence* lead-in ("Accuracy and estimation error do not say …"); *new sentence* pointer to Appendix C |
| 2.4 Local configurations / Kernel score / Skill | 5.1, same run-ins | moved verbatim, equations unchanged; `\NI` on "gain over CSR" added after Skill |
| 3 Model, paragraph 1 (two-step procedure, Poisson MLE) | 4 ParamNet (opener) | factorization $p(f,\theta\mid x)=p(f\mid x)p(\theta\mid x,f)$ **copied from the Intro** (it was not at the top of old Section 3, contrary to the brief); colon after $\hat{\bar n}=n$ → comma (grammar) |
| 3 Model, paragraph 2 (ParamNet overview) | 4 (opener) | verbatim, minus two self-references "(Section 3.1)", "(Section 3.2)" |
| Figure 2 (was after old 3.3 Evaluation) | Figure 2, right after the Section 4 opener | float moved to sit near its first reference; caption replaced |
| 3.1 Features: Classical features | 4, **Classical features.** | verbatim, minus the last sentence ("The tensor is fed to a 1-D CNN encoder …"), which duplicated Classical encoders |
| 3.1 Features: Topological features | 4, **Topological features.** | verbatim, minus the last sentence ("Each image is fed to its own 2-D (resp. 1-D) CNN encoder …"), which duplicated Topological encoders; `\NI` on alpha/DTM definitions added |
| 3.2 Architecture: Classical encoders | 4, **Classical encoders.** | verbatim (label `sec:architecture` sits here) |
| 3.2 Architecture: Topological encoders | 4, **Topological encoders.** | trimmed: the parenthetical list of the four images and their sizes (stated one paragraph earlier) and one self-reference removed |
| 3.2 Architecture: Head | 4, **Head.** | verbatim, minus the last sentence ("This encoder-and-head architecture … is ParamNet"), which duplicated the opener |
| (nothing in source) | 5 Training Procedure, *new sentence* opener + **Training.** | `\NI` only; no training settings exist anywhere in `main.tex` |
| 3.3 Model Comparison, first paragraph | 5, **Conventional learners.** | verbatim; header added (label `sec:learners` kept) |
| 3.3 Classical baseline | 5, **Classical baseline.** | verbatim; `\NI` on the two meanings of "oracle" |
| 3.3 Evaluation | 5, **Evaluation.** | verbatim |
| 5 Results, first paragraph (1 440 test clouds) | 5.1, **Test clouds.** | verbatim plus forward ref "; Section~\ref{sec:wall}" for *wall* and τ, which are defined in Section 6 |
| 5 Results, source comment (`% Draft. Numbers: …`) | top of Section 6 | moved with the results it describes |
| 4 The CSR Wall, opening paragraph | 6 (opener) | trimmed: "This mirrors practice: …" removed (it repeats Section 3 principle 3 and Intro bullet 3); "(Section~\ref{sec:data})" added; *new sentence* that frames Section 6 |
| 4, Detectability (4 sentences) | 6, **Detectability.** (2 sentences) | sentences 1–2 merged with a semicolon, sentence 3 kept; sentence 4 ("The calibration matters …") → Appendix D |
| 4, coordinate list ("For each family … η: the cluster overlap …") | 6, **One coordinate per family.** (claim + pointer to Table 1) and Appendix D, **Regime coordinates.** (list) | split: the claim stays, the per-family list moves |
| 4, "Cell has no monotone coordinate …" | 6, One coordinate per family | verbatim |
| 4, "We construct u = …, with a fitted by logistic regression …, and define the wall …" | 6 (formula + wall definition) and Appendix D, **The wall.** (fit of a) | split. **Deviation from the brief:** the formula $u=\log\eta+a\log\bar n$ stays in the main text because *u* is used there ("within 0.25 in u", Figure 3) |
| 4, AUC sentence; exponent range; Matérn II fixed $R\bar n$ | Appendix D, **The wall.** | moved verbatim |
| Figure 3 | Figure 3 (Section 6) | caption rewritten |
| Table 2 (CSR wall) | **Table 5, Appendix D** | moved; caption gains takeaway |
| 4, The wall belongs to the data | 6, same | verbatim; refs → Appendix D / Table 5; one self-ref "(Section 5)" removed; the 85% at k = 2 claim now points to Table 5, where that number lives |
| 4, In-regime results | 6, same | verbatim; evidence pointers added: "(Table~\ref{tab:classification})" and "(Table~\ref{tab:wall})" |
| 4, Before the wall | 6, same | verbatim |
| 5, ParamNet reproduces observed patterns | 6, same | verbatim |
| 5, Wrong family, right fit | 6, same | verbatim except "aren't" → "are not" |
| 5, Sending near-CSR patterns to Poisson costs nothing | 6, same | grammar: "For these patterns for which CSR might …" → "For these patterns, CSR might …"; `\WIP` kept |
| 5, Per family | 6, same | verbatim |
| 5, The network against conventional learners | 6, same | verbatim minus self-ref "(Section 4)"; `\WIP` kept |
| 5, Out of distribution | 6, same | verbatim; `\WIP` kept |
| (new) | end of Section 6 | one summary sentence, (i)–(iii), built only from existing claims (Intro bullet 2, The wall belongs to the data, Wrong family, Sending to Poisson) |
| Tables 3, 4, 5, Figure 4 | Tables 3, 4, **2**, Figure 4 (Section 6) | kept in the main text; floats ordered by first reference; captions gain definitions and takeaways |
| App. A Simulation Design | App. A | verbatim; hard-coded "(Sections 4–5)" → `(Section~\ref{sec:wall})` |
| App. B Models and Training Cost (empty) | App. B | three `\NI` bullets |
| App. C Validating the Score | App. C | verbatim; C.3 "(Section~\ref{sec:fidelity})" → `(Section~\ref{sec:score})`, where the evaluation set is now described |
| (new) | **App. D The CSR Wall: Details** | *new sentence* opener, three run-ins of moved text, `\NI`, Table 5 |

**Renumbering.**

- Sections: old 2.1–2.2 → 2; 2.3 → 3; 2.4 → 5.1; 3 (minus 3.3) → 4; 3.3 → 5; 4 and 5 → 6.
- Tables: old 1 → 1, old 5 → **2**, old 3 → 3, old 4 → 4, old 2 → **5 (Appendix D)**.
- Figures: unchanged, 1–4.
- Appendices: A, B, C keep their letters, and D is new. That is also the order of first reference: A in Section 3, B in Section 4, C in 5.1, D in Section 6.

All 30 original labels are kept; two were added (`sec:training`, `app:wall`). Both builds report no undefined references and no "??".

---

## 2. Inserted notes (`\NI{…}`), in order

| # | Location (line in `main.tex`) | Note |
|---|---|---|
| 1 | Figure 2 caption (l. 273) | The figure labels the network **PHNet-Fusion**; rename to ParamNet. Strings are in `scripts/f2_pipeline.py` l. 118 and l. 171. |
| 2 | §4 Topological features (l. 282) | missing: define or cite the **alpha and DTM filtrations**; state DTM's *k*. See §6 below: Appendix A says *k* = 20, configs say `dtm_k10`. |
| 3 | §5 Training (l. 302) | missing: **loss** (classifier, estimators), **optimizer, learning rate, batch size, epochs and stopping rule, hardware, training time**. Candidates, unverified for v2: `configs/pipeline.yaml` `nn:` gives lr 1e-3, batch 128, ≤200 epochs, patience 30, seed 1. `paper/notes/04_models.tex` adds AdamW, weight decay 3e-4, cross-entropy / MSE on standardized log targets, best-val weights restored. That note describes the older 5-family, 7 000/1 000/2 000 design, so confirm before use. |
| 4 | §5 Classical baseline (l. 308) | "oracle" means **true-family routing** here but the **true model $P^\star$** in 5.1, the Intro, and Appendix C. Rename one (e.g. "true-family routing" vs "true model"). |
| 5 | §5.1 Skill (l. 348) | define a fit's **gain over CSR**, $S(P^0_i,y_i)-S(P,y_i)$. It is used in Section 6 ("Before the wall", "Sending … to Poisson") and the Table 3 caption, but never defined. |
| 6 | Figure 3 caption (l. 385) | name the **reference classifier**; the legend says "boundary", the text says "wall". |
| 7 | App. B (l. 570) | missing: persistence-image **bandwidth, weighting, birth/persistence box** (resolution 64×64 is in §4). |
| 8 | App. B (l. 572) | missing: training settings and seeds (the full version of note 3). |
| 9 | App. B (l. 574) | missing: **GPU/CPU model, training time and inputs per model**. `tables/v2/t3_cost.tex` (seconds per model) and `tables/v2/sA_models.tex` are generated and look like the intended contents; the README notes that parameter counts and the GPU model are not logged. |
| 10 | App. D, The wall (l. 677) | define the reference classifier on whose predictions the wall is fitted. `configs/pipeline.yaml` has `regime.classifier: hgb_classical`, i.e. trees on classical features. |

---

## 3. Figure and table suggestions (not implemented)

- **Figure 1: show the fit.** Yes. A third row with one simulation from the pipeline's fit $\hat P$ for each "structured" example would put the paper's actual target (does the fit reproduce the pattern?) on page 2.
  - Fix independently: the **cell column's bottom panel is k ≥ 20** (`f1_examples.py` l. 40). Per Figure 3 that pattern is detected almost always, so labelling it "near CSR" is wrong. The near-CSR cell patterns are k = 3–4. The new caption states k = 2 / k ≥ 20 honestly, but the panel choice should change.
  - The Poisson column shows two Poisson draws. One panel would do, and the freed slot could show Matérn I.
- **Figure for "wrong family, right fit".** Recommended. For 2–3 confused pairs (Thomas→nested, Thomas→ring, nested→Thomas), show three panels per row: the observed pattern, a simulation from the predicted-family fit, and a simulation from the true-family fit, each annotated with its skill.
  - This carries the paper's most counter-intuitive claim better than prose.
  - At column width it could replace Figure 4, whose key numbers are already in the text ("Matérn II 0.99, Strauss 0.92, …").
- **Table 2 (old; now Table 5) in the appendix.** Done. Figure 3 and the two-sentence definition carry the claim in the main text. Two problems remain with the table as evidence:
  - The main text says the **L test, a PH-only classifier and ParamNet** agree. The table's columns are "Fitted" (reference classifier, which the config says is `hgb_classical`, not PH-only), the L test, and ParamNet. Add a PH-only column or reword the claim.
  - The table never shows the "30% in the physical coordinate" figure.
- **Merge or slim Tables 3/4/2.**
  - Table 4 (per family) could become a second block of the full-width Table 3, with families as columns and one row each for ParamNet skill, Δ predicted, Δ true. That removes a column float.
  - Table 3's bottom block ("by outcome") is fully restated in the "Wrong family" paragraph and could go.
  - Table 2 (classification) could drop the "Trees, PH" and "Logistic" rows to the appendix (`sA_models.tex` already has them) if the PH comparison stays in the `\WIP`.
- **Figure 3 in print.**
  - The 0 end of the colour scale is near-white, so "never called Poisson" reads as "no data".
  - Over the dark-blue regions of LGCP, Matérn II and Strauss the black dashed τ = 0.9 line has little contrast. Use a white halo under the lines, or a sequential map that stays mid-tone.
  - The legend says "boundary τ = …"; rename to "wall" to match the text.
  - The cell panel is a line plot among hexbins; label its y-axis with the same quantity name as the colour bar.
- **PHNet-Fusion / ParamNet in Figure 2.** Not fixed; the image is unchanged. Regenerate after renaming in `scripts/f2_pipeline.py`.

---

## 4. Missing sections: skeletons

These are drawn from what the draft already argues. Citations still need to be found; the draft has **no bibliography at all**.

**Related Work**

- *Classical SPP inference:* minimum contrast on K, penalized-contrast selection, and pseudo-likelihood for Gibbs models. Their limit, as the draft shows, is families without closed-form K (Strauss, cell).
- *Simulation-based / amortized neural estimation:* networks trained on simulated (pattern, f, θ) pairs. The draft's distinctive points are the classifier-then-estimator factorization and the score that judges simulations, not parameters.
- *TDA for point patterns:* persistence diagrams of alpha and DTM filtrations, and persistence images (Adams et al. 2017, already cited in `notes/03_featurize.tex`). Prior uses of PH for SPP testing and classification.
- *Scoring rules and kernel discrepancies:* strictly proper kernel scores (Gneiting & Raftery 2007) and MMD (Gretton et al. 2012), both already named in Appendix C's `\TBD`.
- *CSR testing:* L-function tests and envelope tests as the canonical detector the draft's wall is compared against.

**Limitations**

- Stationary families on a fixed unit window with $\bar n\in[50,950]$; no inhomogeneous or marked patterns.
- Closed library of eight families. Out-of-library patterns get the nearest mechanism, as Matérn I → Matérn II shows; the end-to-end skill of those fits is still `\WIP`.
- The score is proper only for the law of rescaled radius-1.5 local configurations (Appendix C.1). It is blind to intensity and to structure beyond that radius (e.g. the nested outer scale).
- Below the wall nothing beats CSR, which is a property of the data, so wide priors cap overall accuracy (0.45). Uncalibrated classifiers call up to half of CSR patterns structured.
- The contribution of topology is not yet isolated for the network on v2 (`\WIP` in Section 6).

**Conclusion**

- One pattern in, one simulable model out: a classifier-then-estimator pipeline whose networks combine summary curves with persistence images.
- Accuracy is the wrong target near CSR. Detectability is set by one coordinate per family, and every detector hits the same wall.
- End to end, fits reach skill 0.90 against 0.74 for minimum contrast. Wrong-family fits and Poisson routing cost little.
- Topology matters where second-order statistics are blind (cell).
- Outlook: real data, more families, non-stationary processes.

---

## 5. Page count, warnings, cuts

**Build.** No TeX installation exists on this node, so I built with **tectonic 0.15** (XeTeX engine, standard TeX Live packages) from a scratch directory. Nothing was added to the repo. Page breaks under pdfLaTeX should match to within a line or two; confirm with your usual build.

**Page limit.** Nothing in the repo states the page limit; `aistats2027.sty` has none. I used the AISTATS norm of 8 pages for the main text, excluding references and appendix. Please confirm against the 2027 call for papers.

| | Total pages | Main text ends | Page 9 fill (left / right column) |
|---|---|---|---|
| Before | 12 | page 9 (floats only: Figure 4, Tables 4, 5) | 0.95 / 0.63 |
| After | 13 | page 9 (text, Table 4) | 1.00 / 0.54 |

Both versions are **about 0.8 page over**. Moving the wall table out and trimming duplicates roughly paid for the new captions, openers, the summary sentence and the `\NI` notes (≈0.1 page; they disappear as you fill them, but the content replacing note 3 adds a little).

The real gap is larger:

- The Intro is still bullet notes; as prose it will likely grow by 0.3–0.5 page.
- Related Work, Limitations and Conclusion will need about 0.75–1 page.

Plan for **roughly 2 pages of cuts**.

**Warnings after.**

- Undefined references: none.
- Overfull hbox: one, 5.1 pt, in the abstract. It is pre-existing and comes from the style's abstract box.
- Underfull hbox: pre-existing ones in Classical features, Conventional learners, and The network against conventional learners. One new one in Appendix B, from the long `\NI` lines, which goes away when filled.
- Two underfull vbox warnings during output (float pages); cosmetic.

**Cuts to consider** (largest first; none made):

1. Move the encoder recipes (Classical encoders / Topological encoders: kernel sizes, widths, pooling, dropout) to Appendix B, keeping one sentence each. ≈0.25 page.
2. Merge Table 4 into Table 3 (see §3), or move Table 4 to the appendix and keep the "Per family" paragraph. ≈0.25 page.
3. Move Figure 4 to the appendix if the "wrong family, right fit" figure replaces it; the key numbers are already in the text. ≈0.3 column.
4. Compress Background: the Poisson paragraph restates $K=\pi r^2$, $L=r$, $J\equiv1$ from Summary statistics, and Mat\'ern I's held-out status is stated four times (Families, Mat\'ern paragraph, Table 1 footnote, Size and splits). ≈0.1 page.
5. Drop Table 3's "by outcome" block (restated in the text) and the Trees-PH / Logistic rows of Table 2. ≈0.1 page.
6. Shorten the F/G/J definitions to one sentence each plus $J$'s formula. ≈0.1 page.

---

## 6. Reviewer-pass findings not fixed (content decisions for you)

**Numbers in the text do not match Table 3.**

- "ParamNet … is ahead of it by +0.11 [+0.04, +0.22]": Table 3's Δ row has +0.11 [+0.04, **+0.21**] under *True family*, which compares ParamNet-true to MC-true.
- "On the five families with a K, ParamNet matches minimum contrast given the true family (+0.05 [−0.03, +0.15])": the table shows +0.06 [−0.00, +0.15] (predicted) and +0.04 [−0.03, +0.14] (true).

The text seems to compare ParamNet-*predicted* with MC-*true*, a pairing the table does not show. Either add that column or quote the table's numbers.

**Claims ahead of their evidence.**

- Abstract: "state-of-the-art results on classical benchmarks" (no benchmark appears) and "these features improve estimation performance" (the TDA evidence is still `\WIP`).
- Intro contribution 4 (TDA) rests on the same `\WIP`.
- "Out of distribution" (54–71%, 87%, 5%) and "minimum contrast identifies 33% of LGCP …" have no display. `tables/v2/sB_ood_matern1.tex` exists but is not included.

**Two senses of "detection".**

- §5 Evaluation: *detection* = not called Poisson (argmax).
- §6: a calibrated 5% test.

The sentence that separates them ("The calibration matters …") now sits in Appendix D. Consider renaming the §5 metric, or restoring that sentence to the main text if space allows.

**Undefined terms.**

- "s.d." in RMSE(log θ)/s.d.: the s.d. of what? Appendix C says the family's test s.d. This metric is defined in §5 but never reported in the main text.
- "Coarse" and "mechanism" are defined only in a caption.

**Notation collisions.**

| Symbol | Meanings in the draft |
|---|---|
| $K$ | Ripley's K; number of simulations, $K=16$ |
| $k$ | offspring kernel; cell parameter; score kernel $k(\phi,\psi)$; simulation index $z_k$; DTM's $k$ |
| $R$ | hard-core / Strauss radius; local-configuration radius $R=1.5$ |
| $u$ | location in $F$, $\Lambda(u)$, $\mu_\phi(u)$; regime coordinate |
| $\mu$ | offspring mean; $\mu_{\log}$; occupancy measure $\mu_\phi$ |
| $\alpha(r)$ | generic summary function, next to the "alpha" filtration |
| $N$ | count $N(W)$; number of test clouds; tensor batch dimension |

The most visible collisions are $u$, $K$ and $R$.

**DTM's k.** Appendix A says "the DTM filtration with *k* = 20", but `configs/pipeline.yaml` and `tables/v2/t3_cost.tex` name `dtm_k10`.

**"Boundary" vs "wall".** The Intro ("Before this boundary"), Appendix A ("detection boundary") and the Figure 3 legend say *boundary*; Section 6 says *wall*.

**Section 5's title.** "Training Procedure" now holds the score (5.1) and the evaluation design. "Training and Evaluation" would describe it better. Kept as you named it.
