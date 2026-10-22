# DONE: Writing Agent A5

## Revision round 2 (2026-10-22; internal review round 2, consistency report round 2, DD43)

**State now.**
- `main.pdf` has 21 pages and `supplementary.pdf` has 20 pages (fallback article class).
- All four builds (normal and anonymous) have 0 errors, 0 undefined references, 0 overfull boxes and 0 BibTeX warnings. The anonymous PDFs carry no CreationDate.
- The synthetic-output test build (`analysis_repo/synthetic_test`) is clean.

**Counts.**
- Sections 1-8: **7,057 running words** (8,058 including tables, captions and equations). Per section: Intro 723, Related 1,007, Problem 876, Method 1,848, Setup 1,263, Results 431, Discussion 658, Conclusion 243.
- Abstract: 330 words.
- **Main floats: 7** (Figures 1-5, Tables 1-2). Algorithm 1 moved to the supplement (Algorithm S1).

**Main changes.**
- Statistics text equals STAT_METHODS.md, with no adaptive seed rule.
- E4 is used throughout (T3, T3b, Fig. 4, tests_T3).
- Binding rates are reported.
- Cost of the guarantee follows DD43(7).
- The shared admitted requirement applies to all methods and passes, and the conformal variants get the gap.
- Fairness disclosures: EV-GNN protocol, MLP only at N ≤ 20, attn_proj_nolb with β = 0.01.
- Citation fixes: ACN, Henderson, EV-GNN row label, preprint flags, abstract-only sentence including eland2026safe.
- AI statement rewritten (M11).
- §4 simplified:
  - inline conditions table, with A5 defined as a feasible step;
  - three-part Theorem 1;
  - fallback rule in prose;
  - Lemma 1, FB codes, deficit equation, A6 detail and actuation detail moved to s1/s2.
- Bib post-processing: acronyms braced, `&amp;` fixed, month macros.

**Lint.** Both mirrors have no sentence, paragraph, opener or phrase flags. Only acronym false positives remain.

**Open items.**
- Fig. 1 artwork labels ("closed-form", regulariser formula) need the Orchestrator.
- MLP n_max: 200 in base.yaml vs "N_max = 20" in DD43/STAT_METHODS.
- IEEEtran.bst and saucis.cls are unavailable.
- Author TODOs as listed in `response_to_internal_review_round2.md` §C.

---

## Revision round 1 (2026-10-22; internal review round 1, consistency report round 1, DD30-DD42)

**State now.** `manuscript/main.pdf` has 23 pages and `supplementary/supplementary.pdf` has 19 pages (fallback article class). Both normal builds and both anonymous builds have 0 errors, 0 undefined references or citations and 0 overfull boxes. A test build against `analysis_repo/synthetic_test` (as `\analysisdir`) loads every generated table (T1-TS3) and figure without errors. `refs.bib` is re-extracted from the 114-key merged bib and has 79 cited keys.

**Counts.** Sections 1-8 have 8,535 words of running text (9,533 including tables, captions and display equations). Per section (all words): Introduction 757, Related Work 1,399, Problem 1,260, Method 2,930, Setup 1,345, Results 760, Discussion 820, Conclusion 254. The abstract has 284 words. The main text has 7 floats (Figs. 1-5, Tables 1-2) plus Algorithm 1. The supplementary mirror has 9,161 words.

**Main changes.**
- DD30 two-layer Theorem 1 (A6 added) and the working title with a \TODO.
- DD31 shared request pipeline.
- DD32 J^pen gap to the soft optimum (Prop. S3), with the hard optimum and taper rows (Lemma S1, Prop. S2).
- DD33-DD36 baselines (Lagrangian, no departure bound, Deep Sets, deploy-only, no admission, sLLF, price rule, eMPC-HiGHS 10-step and 24-h, Gurobi eMPC on the literature configuration, EV-GNN in-distribution and zero-shot, trainN50 reference). GNN-DT status corrected per DD42-6.
- DD37 tuning, DD38 statistics, DD39 sensitivities, DD40 Jacobian moved to the supplement, DD41 AI statement.
- Citation fixes M10-M12 and the TD3 sentence.
- Table 1 has no "unknown" cells, and there is a single abstract-only statement.
- The supplementary is restructured as s0 nomenclature, s1 proofs, s2 EV2Gym/A1/A3/A6/scenarios/features, s3 pipeline and optima, s4 methods/training/statistics/compute, s5 results.
- New macros `\restable` and `\resultfig` with `\analysisdir`, plus local fallbacks, and adjustbox.
- Item-by-item record: `response_to_internal_review_round1.md`.

**Lint.** `lint_latest_main.md` and `lint_latest_supplementary.md` show no sentence, paragraph, opener or phrase flags. The only flags are acronym false positives (assumption labels, S/T item numbers, FB codes, SA, SAUCIS, DT4EVs, V2G in the title).

**Deviations and open items.**
1. Example day is Fig. S7 and pooled constraints is Fig. S6. The analysis README still plans fig5_example_day as main Fig. 5.
2. IEEEtran.bst is unavailable (CTAN blocked, 403), so ieeetr is still used.
3. saucis.cls is unavailable.
4. No analysis output exists for the V2G profit share or an information/policy gap decomposition.
5. Author tasks: R1-R3 full texts, final title, rewriting the AI-drafted sections, result interpretation, hardware, repository link, re-measuring the scenario statistics.

---

## Revision 1 (2026-10-22, orchestrator request; DD25-DD29)

**State now.** Main text `manuscript/main.pdf`: 23 pages (fallback article class, 10 pt, single-spaced, with references and Article Information Form). Supplementary `supplementary/supplementary.pdf`: 12 pages. Both compile with 0 errors, 0 undefined references/citations, 0 overfull boxes; anonymous builds exist. `refs.bib`: 76 cited keys (both documents).

**Main-text words (linter count, prose only):** Introduction 813, Related Work 1,115, Problem Formulation 1,233, Method 3,037, Setup 980, Results 720, Discussion 613, Conclusion 270, so **8,781 for Sections 1-8** (9,120 with the 339-word abstract). This is about 3% above the 8,500 target. Supplementary: 4,481 words (S1 1,990; S2 1,313; S3 344; S4 677).

**Floats in the main text: 7** = Figs. 1-5 + Tables 1-2 (plus Algorithm 1):
Fig. 1 architecture; Fig. 2 two-panel concepts (`fig2_concepts`, captions written from the drawn panels); Fig. 3 zero-shot (`fig3_zero_shot_scaling.pdf`); Fig. 4 constraints (`fig4_constraint_satisfaction.pdf`); Fig. 5 cost of the guarantee (`fig6_cost_of_guarantee.pdf`); Table 1 positioning; Table 2 main comparison (`analysis/tables/T1.tex`, placeholder fallback).
**Deviation from the figure plan:** to stay at 7 floats with the positioning table and the main-results table, the example-day figure (`fig5_example_day.pdf`) moved to the supplement as Fig. S6. Moving it back is one figure block (then 8 floats).

**Moved to the supplement:** all proofs (main keeps the statements of Lemma 1 = one-step invariance, Proposition 1, Theorem 1 with a proof sketch, Corollary 1, Proposition 2), Lemma S1 equivariance, S2 reachable energy, S3 taper/A1, S4 A3 in EV2Gym, S5 fallback, S6 deficit bound, Proposition S1 LP bound incl. the J_LP,ceil variant, EV2Gym notes, scenario table (S1), admission LP, baseline implementation notes, PPO algorithm (Alg. S1), hyperparameter/tuning table (S2), compute budget, generated tables T2-T7 and TS1 (Tables S3-S9, placeholders until generated), sensitivity table (S10), Figs. S1-S6 (S5 pipeline exists).

**Method/setup updates:** DD25 admission targets (main 4.5, S2.4); DD26 gap w.r.t. J_LP,ceil with the reason (Section 3.6, Prop. S1, TS1 for J_LP); DD27 EV-GNN per profile at N=20, rho=4, 5 seeds, commit 01aa1b7a, GNN-DT not included (no public code); eMPC-HiGHS as a non-identical port; DD28 sensitivity scope; compute section with run counts (51 own PPO runs + tuning + optional runs; no measured timings, these are RESULT placeholders); parameter counts incl. small network and MLP. Availability (AIF and Section 5.4): code release with TODO link, EV2Gym 55ef368c, EV-GNN 01aa1b7a. The smoke-run numbers in the method docs are not used.

**Lint (final, `lint_final_manuscript.md`, `lint_final_supplementary.md`):** remaining blocking flags only "undefined acronyms" A1-A5 (assumption labels), SA (from SA-Attn) and S1-S10 (supplementary item labels such as "Table S3"); all justified. No banned/limited phrases, no sentence > 35 words, all means 16-26, no long paragraphs, no repeated openers. CP is now defined (conformal prediction), matching the analysis name SA-Attn-CP.

**Open issues added:** (1) Method section is still 3,037 words; trimming 4.1 or 4.7 by ~300 words would hit 8,500 exactly. (2) The pipeline figure S5 shows "optimum J^LP" and omits EV-GNN; the analysis agent may update it to J^LP,ceil. (3) Paths assume the analysis folder is `analysis/` next to `paper/` in the project (it is `analysis_repo/` in this workspace).

---

## Round 1 report (superseded where Revision 1 differs)


## Produced

| File | Content |
|---|---|
| `manuscript/main.tex` + `sections/00-10` | Full LaTeX manuscript; compiles with `latexmk -pdf` (pdflatex + bibtex), 0 errors, 0 undefined references or citations, 0 overfull boxes |
| `manuscript/main.pdf` | 32 pages in the fallback `article` class (10 pt, A4, single-spaced, no paragraph indent); 19.5k words in the PDF text including tables, appendix and references |
| `manuscript/main_anonymous.pdf` | Double-blind build (`make anon`) |
| `manuscript/refs.bib` | 85 cited entries, extracted from `lit/merged/references.bib` only (no other key can be cited: the script stops on a missing key) |
| `manuscript.md` | Markdown mirror, generated by `tools/tex2md.py` (refs resolved from `main.aux`) |
| `tools/extract_bib.py`, `tools/tex2md.py`, `manuscript/Makefile` | Build helpers |
| `lint_round0.md` to `lint_round4.md` | Style-lint history: 87 blocking flags in round 0, 6 in round 4 |
| `cover_letter.md`, `declarations.md`, `README.md` | Submission drafts and build notes |

Bibliography style: `IEEEtran.bst` is **not installed** in this TeX Live; `main.tex` uses `\IfFileExists{IEEEtran.bst}{IEEEtran}{ieeetr}`, so it switches to IEEEtran automatically once the template's `.bst` is in the folder.

## Word counts (prose words counted by the linter; math, tables, algorithms and captions of tables excluded)

| Section | Words |
|---|---|
| Abstract | 339 (limit 400) |
| 1 Introduction | ~810 |
| 2 Related Work | ~1,780 (+ Table 1) |
| 3 Problem Formulation | ~1,140 |
| 4 Proposed Method | ~3,570 (+ Algorithms 1-2) |
| 5 Experimental Setup | ~1,210 (+ Tables 2-3) |
| 6 Results (skeleton) | ~1,010 |
| 7 Discussion (draft) | ~570 (+ TODO bullet list) |
| 8 Conclusion | ~270 |
| Appendix A (proofs) | ~1,940 |
| Article Information Form | ~320 |
| Total analysed | ~13,500 |

Main text without appendix and back matter is about 11,300 words; SAUCIS guidance for experimental papers is about 25 double-spaced pages (roughly 7,000-8,000 words) and at most 7 figures plus tables. The draft has **8 figures and 6 tables** (Table 1 positioning, 2 scenarios, 3 hyperparameters, 4 main results, 5 ablation, 6 conformal). The guidance is not a hard limit ("no page limit"), but see open issue 1.

## Style lint (final round 4): remaining blocking flags, all justified

| Flag | Why it stays |
|---|---|
| A1, A2, A3, A4, A5 "used but never defined" | Assumption labels, not acronyms; the linter cannot accept a definition with one capital letter. They are stated in Assumption environments (Section 4.2/4.4) and match `method/` and Figure 1. |
| SA "used but never defined" | Comes from the method name SA-Attn (defined as "safe attention policy (SA-Attn)" in the abstract and the Introduction); the linter splits at the hyphen. Name is still a placeholder (`\method` macro, to be fixed at G2). |
| V2G "used before first definition" (non-blocking) | Only the title precedes the definition in the abstract. |

No banned or limited phrases, no em dashes, no rhetorical questions, no sentence over 35 words, no paragraph over 180 words, all section means within 16-26, no repeated consecutive openers, at most one This/These/It/There opener per paragraph. `, which` / `, while` / `; thus,` / `, thereby` chaining removed everywhere (checked by grep). Formal theorem statements were also kept under 35 words.

## Content decisions

- Novelty phrased as a combination, "to our knowledge", with a visible `\TODO` that the claim depends on the full texts of orfanoudakis2026can, song2025longterm and eland2026safe (risks R1-R3). No "first" claim for any ingredient. The projection is stated openly as the sLLF rate rule and the continuous quadratic knapsack projection (Introduction, Section 4.3) with citations.
- No literature number marked (n) in `gap_analysis.md` is used. Numbers kept: GNN-DT "reward within 5% of oracle" (journal abstract), sizes of transfer studies (25 to 100 chargers, 1,236 to 3,218 vehicles, 141 to 322 buses, 1000 charge points / nodes).
- Integrity: no experimental number, trend or outcome is stated. Results, the abstract's last sentence, three Conclusion sentences and the Discussion interpretation are `\RES{}`/`\TODO{}` placeholders. Design facts used: scenario parameters (DD24, including the occupancy-equalising spawn multipliers), parameter counts, ε_q = 0.008 kW, κ bounds of the EV models (from the method analysis, not from experiments). Numerical verification tolerances of the projection were *not* written; a `\TODO` asks for the unit-test value.
- Penalty weights' "more than a factor of 11 above the highest hourly price" is a fact about the price data (algorithm.md Alg. 5), kept.
- Limitations (Section 7.3) fully written: known departure/energy (conformal relaxes only d_i, marginal coverage), A5 closed-loop failure and fallback, single station/transformer, no voltage or network constraints, EV2Gym lossless accounting (DD1), constant efficiency (DD4), taper assumption A1 and taper multiplier for A3, simulation only with dropped late sessions, synthetic forecast noise, LP taper relaxation (gap overestimates), novelty-risk dependence.

## Numbering map (method docs -> paper)

| `method/` | Paper |
|---|---|
| Lemma N1 (equivariance) | Lemma 1 |
| Lemma 0 + threshold characterisation | Lemma 2 |
| Lemma E1 (A1 in EV2Gym) | Lemma 3 |
| Lemma 1 (one-step invariance) | Lemma 4 |
| Proposition 1 (projection) | Proposition 1 |
| Theorem 1, Corollary 1 | Theorem 1, Corollary 1 |
| Lemma 2 (fallback), cases FB-1*/FB-2/FB-1/FB-0 | Lemma 5, cases (i)/(ii)/(iii)/(iv) |
| Lemma 3 (deficit bound) | Lemma 6 |
| Proposition 2 (conformal) | Proposition 2 |
| Lemmas E2+E3 (A3 in EV2Gym) | Lemma 7 (Appendix A.2) |
| Proposition O1 (LP upper bound) | Proposition 3 (Appendix A.8) |
| Corollary 2 (EV2Gym configuration) | conditions of Lemmas 3 and 7 |

## Placeholders the author must fill

`\RES{}` (red): 1 in the abstract, 48 in Results (Tables 4-6 cells and structure sentences in 6.1-6.7), 3 in Discussion 7.1, 3 in the Conclusion. Result figures 5-8 are placeholder boxes until `analysis/figures/fig_zero_shot.pdf`, `fig_constraints.pdf`, `fig_example_day.pdf`, `fig_cost_of_guarantee.pdf` exist.

`\TODO{}` (blue):
1. Author names, e-mails, ORCID iDs, affiliations (`main.tex`).
2. Novelty sentence vs. full texts of orfanoudakis2026can, song2025longterm, eland2026safe (Introduction) and a precise description of the orfanoudakis2026can layer (Section 2.2); last sentence of Limitations.
3. Agreement of the taper closed form with EV2Gym from the unit tests (Section 4.2).
4. MPC solver for N > 25 and whether EV-GNN/GNN-DT runs as a baseline (Section 5.2).
5. Hardware, software versions and anonymised repository link (Section 5.4).
6. Interpretations in 6.2, 6.5 and 7.1 (four paragraph prompts), practical implications 7.2 (four bullets), supplementary timing table (6.7).
7. Article Information Form: contributions, acknowledgements, conflict-of-interest confirmation, funding, ethics wording confirmation, availability link, AI statement confirmation (Option A, tool "Claude (Anthropic), model claude-opus-5-5").

## Open issues

1. **Length.** About 11,300 main-text words, 8 figures and 6 tables, against SAUCIS guidance of about 25 double-spaced pages and 7 figures+tables. Candidates to move to supplementary material: Appendix A.2 (taper and Lemma 7) and A.8, Table 3 (hyperparameters), Table 6 (conformal) or Figure 7 (example day), Figure 4 (pipeline); Related Work could lose about 400 words (Section 2.3 power-system paragraph, parts of 2.2).
2. **Template.** `saucis.cls` was not available; front matter (authors, ORCID, keywords, article history) must be moved into the template's macros by hand (README).
3. **Conceptual figures 2-4** were not yet present at `analysis/figures_conceptual/`; captions describe the intended content and must be checked against the drawn figures. Figure 1 legend says "A1–A5", consistent with the paper.
4. **Method name** SA-Attn is a placeholder (`\method` macro).
5. **Novelty risks R1-R3** (orfanoudakis2026can, song2025longterm, eland2026safe) are still open; the Introduction and Section 2.5 claims are conditional on them.
6. **AI policy.** SAUCIS deems AI-generated abstract, introduction, literature review or discussion inappropriate except as preliminary drafts that the authors thoroughly revise; the authors must rewrite these sections and write all result interpretation themselves.
7. The cover letter is optional for SAUCIS; highlights are not required by SAUCIS, so none were written.
