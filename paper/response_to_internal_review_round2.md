# Response to internal review round 2 and consistency report round 2

Writing agent A5, 2026-10-22. Inputs:
- `review/internal_review_round2.md`;
- `review/consistency_report_round2.md`;
- DD43;
- `code_repo/code/SCHEMA_ADDITIONS.md` and README;
- `analysis_repo/README.md`;
- `analysis_repo/tables/STAT_METHODS.md`.

Locations: `NN` = `manuscript/sections/NN_*.tex`, `sK` = `supplementary/sections/sK_*.tex`.

**State after this round**
- **Builds:** main 21 pages, supplementary 20 pages (fallback article class). Both builds and both anonymous builds have 0 errors, 0 undefined references, 0 overfull boxes and 0 BibTeX warnings. The anonymous PDFs carry no CreationDate.
- **Synthetic test:** a build against `analysis_repo/synthetic_test` loads every generated table and figure without errors. A Unicode issue (λ, Σ in T3b) is fixed by declarations in the preamble.
- **Lint:** no sentence, paragraph, opener or phrase flags in either mirror. The remaining acronym flags are false positives: A1-A6, E1-E4, S/T item numbers, FB codes, SA, SAUCIS, DT4EVs.
- **Length:** Sections 1-8 have **7,057 running words** (8,058 including tables, captions and display equations). The abstract has 330 words by the linter count (limit 400).
- **Floats:** the main text has **7 floats in total**: Figures 1-5 and Tables 1-2, with no algorithm. Algorithm 1 moved to the supplement as Algorithm S1. The conditions box in §4.4 is an unnumbered inline tabular, not a float.

## A. Internal review round 2

| Item | Change and location, or justification |
|---|---|
| M1 / B2 novelty (R1-R4 unread) | **Author task.** No open-access copy exists (citation_fixes §3). 02 §2.5 now names all three abstract-only works, orfanoudakis2026can, song2025longterm and eland2026safe ("recent", not "2026"). The Table 1 row is labelled "EV-GNN, preprint version", with no "(preprint)" inside the cells. The author note in 02 now lists R1-R4. |
| M2 residues / N-m9 | Fig. 1 caption is rewritten (transformer limit, and reachability while an EV receives its bound). "Absorbs a bounded under-delivery" is removed (07: "a remedy not analysed here"). The Table 1 departure cell reads "per-EV bound, conditional (Thm. 1b)". The abstract says "admitted requirement" and defines it. **Fig. 1 artwork** still says "closed-form projection" and "β‖x − Π(x)‖²". The artwork belongs to the Orchestrator and should be relabelled. |
| M3 / N-M3 shared requirement | DD43(1) is implemented in the text. 03 §3.2, 05 §5.2, s3 §S3.1 and s4 §S4.1 state that every method and pass receives the shared admitted requirement: EV-GNN and the Gurobi eMPC through EV2Gym's EV object, eMPC-HiGHS for arrivals in its horizon, conformal passes re-certified with d̂^lo, no-admission passes keeping the capped request. Shortfall is always measured against the shared requirement. |
| M4 binding rate | Produced (`agg_upper/lower_active_steps`, `lb_active_frac`, T3b). Asked for in 06 §6.3, 07 and s5. N-m18 (the N=50 reference extrapolates at 100/200) is stated in s5 §S5.1. |
| M6 / N-M3 EV-GNN protocol | Disclosed in 05 §5.2, s4 §S4.1 and 07: own configuration, one load/PV draw per run, 3×10⁶ steps instead of 10⁷, best-reward checkpoint. |
| M7 / N-M1 statistics | The text now matches STAT_METHODS.md (DD43(5)), in 05 §5.4 and s4 §S4.4: seed as the unit; seeds fixed in advance (5; 3 for trainN50 and w ∈ {1,100}); **no adaptive rule**; primary test a paired t on seed means; one-sample t against deterministic baselines; Welch when seed indices differ; Welch as a robustness column; Wilcoxon on replay-level differences as the secondary test; Holm per test file (= table); bootstrap CIs, IQM and PoI; CIs are the primary evidence. The reviewer's objection to pairing by seed index is noted, but DD43(5) fixes the paired test, and the Welch column reports the unpaired check. E3 and E4 are tested in `tests_T3.csv`. |
| M8 / N-m2 oracle rows | 03 §3.3 and 07: "outer linear rows, exact except on a one-step transition band and a 1 % margin". |
| M9 economics | Unchanged. The limitation is kept (07). The V2G profit share is still not produced. |
| M10 length | 7,057 running words, 7 floats. §2 cut to about 1,000 words, §3 to about 880 and §4 to about 1,850. Algorithm 1 is moved to the supplement. **Not done:** a saucis.cls build (class unavailable). |
| M11 AI statement | Rewritten in 10, the Acknowledgments and declarations.md. AI searched and verified the literature, derived the method and proofs, designed the experiments, wrote the simulation and analysis code, drafted the text and made the conceptual figures. A separate instance of the same AI re-checked the proofs, and the statement says this is not an independent human verification. The authors executed all experiments on their own computer, and all numbers come from those runs through the scripts. Interpretation and Discussion are written by the authors, who reviewed, revised and take responsibility. The \TODO asks the authors to confirm the wording and the interface names (Claude Cowork; Claude Code/Agent SDK sessions were also used), whether they re-checked the proofs, and to rewrite the sections. |
| N-M2 / B4 E4 and promised outputs | E4 = `share_evs_stay_feasible` (DD43(2)) is used consistently as "share of EVs whose stay contains no infeasible step" (00, 01, 05, 06, 07, 08, s4, s5). It appears in T3 (pooled), T3b (per N) and Fig. 4 (caption updated), with tests in tests_T3.csv. Promises without an output are removed: the ratio-of-sums *gap* (03), the LP simultaneity count (03, s3) and the evaluation time (s4 asks only for training time). The cost of the guarantee *is* a ratio of sums (DD43(7)). Capped/reduced requests are in T3. |
| N-M4 readability | §4 is restructured. The conditions are grouped in one inline table (A1, A2, A3, A4, A6 as assumptions; A5 as the definition of a feasible step, "measured"). Theorem 1 now has three parts: (a) transformer and physical limits, (b) per EV, (c) global. Lemma 1 moved to the supplement (Lemma S3). The fallback rule is described in prose; the FB codes, the full case list, the deficit equation, actuation/rounding details, the A6 tail argument and Algorithm 1 moved to s1/s2. R and Ψ_k are removed from the main text, and the cap is defined in words (N-m1). §4.6 and §4.7 are shortened. |

| Minor | Change |
|---|---|
| N-m1 | Eq. (cap) now states E^req as the pipeline output with "≤" as a property (03). |
| N-m3 | Proposition S3 is rephrased as a property of the measurement on every simulated trajectory (s3, 03). Conformal and no-admission passes are covered, and the conformal variants get the gap (DD43). |
| N-m4 | Gurobi eMPC commands are clipped (s3, s4; DD43(1)). |
| N-m5 | A6 is checked on every replay through the fallback counts of cases (i), (ii) and (iv) (s1 §S1.6, s2 §S2.3, 04 table). m_L and m_P are drawn once per episode. The case (i) description is aligned in s1 (procedure of (iii), sub-case (iv) for a negative budget), s2 and code. |
| N-m6 | The η = 1 hypothesis stays in Lemma S4. The main text no longer states the bound as an equation. |
| N-m7 | Uniform draw per calibration episode, τ = 1 after d̂^lo, and requirement re-certified with d̂^lo (04 §4.6). |
| N-m8 | Abstract: "marginal coverage of at least the nominal level minus the rate of EVs that receive less than their bound". |
| N-m10 | The EV-GNN row label is "preprint version". eland2026safe is added to the abstract-only sentence. |
| N-m11 | The MLP is evaluated only at N ≤ 20 (05, s4, s5 TS3 "without the MLP baselines"; DD43(3)). |
| N-m12 | The cost limit of 0 is kept (DD42(1)). s4 justifies it (hard requirements, Λ_max cap) and notes that the training logs record the multipliers. |
| N-m13 | The cost of the guarantee is now ΔJ^pen (primary) and Δ profit (secondary) per DD43(7). The T5 caption point that the hard gap of a violating method can be negative is stated in s5. **Not done:** a break-even weight, which the pipeline does not compute. |
| N-m14 | The ablations remain in Table S9, because the generated T1 has fixed rows (analysis). 06 §6.5 reports them first. |
| N-m15 | **Not done:** a 24-h MPC without arrival foresight (no code). The information advantage is stated in 05 and 07. |
| N-m16 | All main-text placeholders now cite `\Tab{S-tab:..}` (they print "Table S5" etc.) instead of pipeline IDs. The supplementary placeholders use `\Tab{tab:..}`. Labels inside generated captions (DD36, Lemma 3) are an analysis-side item. |
| N-m17 | PV: "multipliers of mean 0.4 P_tr" (05) vs "about 0.38 P_tr" (Table S2), now consistent. |
| N-m19 | "Infeasible step" is defined in the abstract. |
| N-m20 | `\method` is fixed as SA-Attn and must match names.py (preamble comment). |
| N-m21 | 04: A6 uses N so that it is policy-independent. |
| N-m22 | Figures are not merged (fixed analysis outputs). The float count is met by moving Algorithm 1. |

## B. Consistency report round 2

| Item | Change |
|---|---|
| B2 | See M1 (author task). |
| B4 | See N-M2 (E4 produced per DD43(2), text aligned). |
| M-R1 IEEE style | `extract_bib.py` now braces acronyms in titles ({EV}, {RL}, …), fixes `&amp;` and maps months to BibTeX macros. 0 BibTeX warnings. **Still open:** IEEEtran.bst is unavailable (proxy), so ieeetr is used. |
| M-N1 | tests_T5 now has J^pen (DD43). 06 cites it. |
| M-N2 | See N-M2. Items without an output are removed from the text. Capped energy is in T3. |
| M-N3 | MLP only at N ≤ 20 (DD43(3)), stated in the text. |
| M-N4 | attn_proj_nolb uses β = 0.01 and fallback masking (s4, Table S4 row, 05). |
| M-N5 | Fig. S5: the local copy `supplementary/figures/figS5_pipeline.pdf` is now identical to `analysis_repo/figures_conceptual/figS5_pipeline.pdf` (md5 a6752d48…). The figchain uses it, and the caption matches. |
| M-N6 | Fig. 1 caption fixed. The artwork labels still need the Orchestrator's edit (see M2 row). |
| M-N7 | ACN: "hard infrastructure limits and best-effort energy delivery". Hard departure and transformer constraints are attributed only to diazlondono2026 (02 §2.1). |
| M-N8 | Henderson: "results over few seeds can mislead" (05, s4). The five-seed-minimum claim is removed. |
| m1 | The analysis README now lists Fig. 5 = fig6_cost_of_guarantee and Fig. S7 = fig5_example_day, which matches the paper. |
| m2 | `\pdfinfoomitdate` and an empty trailer id in anonymous builds (preamble). |
| m3, m4 | Normal(·) in s2. The auxiliary is renamed ĥ. |
| m5 | PV wording (N-m17). |
| m6, m7, m8, m10, m11, m12, m14 | Placeholder wording matched to the outputs: fallback steps per episode, Fig. S3 panels, TS1/TS2 "main methods", Fig. 4 "main methods", deficit ratio only for full-layer methods, Fig. 5 description with ΔJ^pen and Δ profit. |
| m13 | CPU-h "for evaluation, admission tables, calibration and optima" (05, s4). |
| m15 | Generator labels (J_LP,ceil subscript, "Lemma 3", "transformer violation") are analysis-side, not changed here. |
| m16 | A6 reference → Section S2.3 (`S-app:a6`). |
| m17 | One-sample and Welch cases stated (s4). |
| m18, m20 | "exact prices and arrivals inside its horizon; load/PV from noisy forecasts" (05, 07, s4). |
| m19 | "a check script released with the code" (s2, s3). |
| m21 | No-admission passes keep the capped request, and shortfall is measured against the shared requirement (05, s3). |
| m22 | The dense ablation uses the tuned w. The curriculum clause is added (s4). |
| m23 | EPS export at submission (author task). |
| m25 | Interface names flagged in the AI \TODO. |
| m26 | The cover-letter claim about SAUCIS publications is replaced by an optional \TODO. |
| m27 | Exclusion includes a non-optimal solver status (03). |
| m28 | Docs and code staleness: not paper items. |
| m29 | f^dis removed from Table S1. |
| m30-m32 | "recent works"; GNN-DT cell "near-oracle reward (journal abstract)"; the zero-gradient claim cites kasaura and the equivalence cites markgraf (02 §2.2). |
| m9 (preprints) | HetGPS, Reddy et al. and Yan/Wan are marked as preprints in the text and in Table 1. EV-GNN numbers are attributed to its preprint version. |

## C. Open items (authors or Orchestrator)
1. **Authors:** read orfanoudakis2026can, song2025longterm, eland2026safe and the published EV-GNN article; choose the final title; rewrite all AI-drafted sections and replace the result-sentence frames; write the interpretation; confirm the AI statement (interface names, own proof check); hardware details and repository link; re-run the scenario statistics and the closed-form checks; IEEEtran.bst and saucis.cls; EPS figures.
2. **Orchestrator:** relabel the Fig. 1 artwork ("closed-form", regulariser formula). Note that base.yaml has `n_max: 200` for the MLP, whereas DD43(3) and STAT_METHODS say "N_max = 20". The paper avoids stating the slot count in the main text, and s4 keeps "200".
