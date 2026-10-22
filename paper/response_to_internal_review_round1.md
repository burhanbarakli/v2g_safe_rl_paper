# Response to internal review round 1 and consistency report round 1

Writing agent A5, 2026-10-22. Inputs: `review/internal_review_round1.md` (M1-M11, m1-m24, Q1-Q14), `review/consistency_report.md` (B1-B3, M1-M14, m1-m36), `lit/merged/citation_fixes.md`, DD30-DD42, `method/*.md` (Theorem 1′ in safety_layer.md §5.1), `code_repo/code/README.md`, `SCHEMA_ADDITIONS.md`, `analysis_repo/README.md`.

Locations: `NN` = `manuscript/sections/NN_*.tex`, `sK` = `supplementary/sections/sK_*.tex`. Section numbers refer to the current PDFs (main 23 pages, supplementary 19 pages, fallback article class).

Status after this round: both documents and both anonymous builds compile with 0 errors, 0 undefined references and citations and 0 overfull boxes. The build was also tested with the synthetic analysis outputs (`analysis_repo/synthetic_test`) as `\analysisdir`: all generated tables T1-TS3 and figures load without errors. Lint: no sentence, paragraph, opener or phrase flags in either mirror. The only remaining flags are acronym false positives (A1-A6, S1-S14, T1-TS3, FB-0/1/2, SA from SA-Attn, SAUCIS, DT4EVs, V2G in the title).

Counts: Sections 1-8 have **8,535 words of running text** (no tables, captions, display equations or algorithm) and 9,533 words in total. The abstract has 284 words (limit 400). The main text has **7 floats**: Figures 1-5 and Tables 1-2, plus Algorithm 1.

---

## A. Internal review: major issues

| Item | What changed and where, or why not |
|---|---|
| **M1 Novelty / positioning** | All hedges are removed ("could read in full", "in the version we read", "unknown" cells, the novelty-risk sentence in the Discussion). 02 §2.5 keeps one honest sentence: orfanoudakis2026can and song2025longterm were available only as abstracts and are described at that level. The novelty claim appears once, in §2.5, with "To our knowledge". The Introduction points to §2.5 instead of repeating it (also m30). Table 1 now has no "unknown" cells. The can row reads "not stated in abstract". EV-GNN reads "separate model per scale (preprint)" and "optimisation baseline per scale (preprint)". Contributions are recast (01): C1 is the layer plus the two-layer theorem, C2 is attention as the vehicle with ablations, C3 is the evaluation with the J^pen gap and the hard optimum, C4 is conformal (secondary). The transfer range is no longer sold as a scale record. 01 and 04 §4.4 state that the induction extends Chen et al.'s laxity argument. **Not done:** reading R1 and R3 in full. No open-access version exists (citation_fixes §3), so library access is needed. A LaTeX comment in 02 reminds the authors to re-check R1-R3 before submission. |
| **M2 Guarantee oversold** | Title, abstract, contributions, method, conclusion, cover letter and Fig. 1 caption now follow DD30. 04 Theorem 1: (a)(b) give the transformer and physical limits for every raw-action sequence under A3 and A6 (new assumption A6, exogenous headroom). (c) is per EV for every EV with p ≥ l during its stay. (d) is the global special case under A5. The full proof is in s1 §S1.5 and covers fallback steps. A6 has probability < 2×10⁻⁹ per episode (s2 §S2.3). "Hold at every step" for departure requirements is removed (m1). The share of EVs whose stay contains no infeasible step is a primary endpoint (05 §5.3/§5.4, Table T3b). 04 adds the A3-on-hardware remark (margin on B̂, measured-energy feedback), and 07 and 08 add it as a limitation. The working title is set with a \TODO saying the authors choose the final title. |
| **M3 Admission / oracle asymmetry** | DD31: the station-level request pipeline (cap plus admission LP) runs once per replay and is shared by all methods and both optima (03 §3.3, s3 §S3.1). Shortfall is measured against the same admitted requirement for every method. DD32: the main metric is gap^pen to the soft-penalty optimum, which bounds every controller (Proposition S3 = softoracle, s3). The hard optimum J^LP,ceil (Proposition S2) is used for the cost of the guarantee. Per-episode mean with the 1 € exclusion, ratio of sums and the number of excluded episodes are in 03 §3.6. Admission reductions are reported separately (T3b). |
| **M4 Zero-shot validity** | The profile zero-shot claim is deleted. 05 §5.1 says the profiles are seen in training and that zero-shot covers N and the ρ_tr = 3 kW stress level only (DD36). Per-N binding rate and infeasible-step rate are reported (05 §5.1, T2/T3b). The in-distribution reference trained at N = 50 and the transfer gap are added (05 §5.2/§5.3). SA-Attn-CP is stated as not zero-shot (04 §4.6). **Not done:** leave-one-profile-out, sub-linear rating, N-free calibration. These were not chosen in DD36. 07 states the linear-scaling limitation. |
| **M5 Missing baselines** | Added in 05 §5.2 and s4 §S4.1: transformer-only projection (attn_proj_nolb), deploy-only projection (attn_penalty_safedeploy), Attention+PPO-Lagrangian, Deep Sets+safety layer (parameter-matched), sLLF, price-threshold rule (raw and with the layer), eMPC-HiGHS 10-step and 24-h at all N, MLP restricted to N ≤ 20 and the literature configuration, and the trainN50 reference. The dense penalty is listed as an ablation. **Not done:** log-N attention temperature and H_f > 8. These are design choices outside DD31-DD39. |
| **M6 Baseline protocols** | The false "no public code" claim for GNN-DT is corrected. 05 and s4 say: public code (DT4EVs, MIT, commit 8b327d12), not run because it needs optimiser-generated offline datasets and a port (DD42-6), stated as a limitation in 07. EV-GNN: 3×10⁶ steps, per profile at N=20, ρ=4, five seeds, reported in-distribution and zero-shot. The action mapping for EV-GNN+safety layer is in s4. eMPC-HiGHS is the single MPC at all N, and Gurobi eMPC is a cross-check on the literature configuration (DD34). **Not done:** training EV-GNN on our mixture (DD35 chose per-profile training). The asymmetry is in the Limitations. |
| **M7 Tuning / statistics** | DD37 is in 05 §5.4 and s4 §S4.3/Table S4: 12 configurations per method for 7 methods, on the training mixture, 0.5×10⁶ steps, seed 0, 24 validation episodes, w ∈ {1,10,100} for penalty methods, η_Λ for the Lagrangian method, selection by J^pen at w = 10. DD38 is in 05 §5.4 and s4 §S4.4: the seed is the unit, paired t on seed means, stratified bootstrap, Wilcoxon as a secondary test, Holm per table, IQM and probability of improvement, endpoints E1-E4, and 10 seeds if the intervals overlap. **Not done:** tuning β (DD37 fixes β and the architecture for all methods with the layer). |
| **M8 Oracle relaxation** | The taper rows are exact up to the step discretisation (Lemma S1 = taperrows with full proof, s3 §S3.2). The remaining relaxations are listed in 03 §3.6, s3 and 07. J^LP vs J^LP,ceil is in Table TS1 (s5). The information part of the gap can be read from eMPC-HiGHS-24h (perfect information within 24 h). An explicit decomposition is not written, because no analysis output exists for it. |
| **M9 V2G economics** | DD39 sensitivities (throughput cost 0.05 €/kWh, discharge factor 0.9) are in 05 §5.1, s3 (oracle objective with c^thr) and Table TS2. The complementarity check is in s3 §S3.2 (m21). 07 discusses lossless accounting. **Open:** "share of profit from V2G discharging" has no analysis output, so no placeholder asks for it. |
| **M10 Length / fit** | Running text is 8,535 words and the main text has 7 floats. Proposition 1(iii) is moved to Proposition S1 (DD40). The oracle details and ε_q are in the supplement. The §4.6 conformal part is compressed. An intuition paragraph precedes Theorem 1. Fig. 2 caption is shortened (m11). **Not done:** a build with the official saucis.cls, which cannot be downloaded here. |
| **M11 AI policy** | The AI statement is rewritten (10 and declarations.md, DD41): it covers literature search, screening and verification; method derivation and proofs, independently re-checked by a separate AI agent; experiment design; code and analysis pipeline; drafting of all sections and figures. It states that the authors reviewed and revised everything, take full responsibility, and wrote every interpretation. No AI generated, reported or interpreted the results. Tool: Claude (Anthropic), claude-opus-5-5, via Claude Cowork, October 2026. Acknowledgments repeat the disclosure. A \TODO says the authors must rewrite the AI-drafted sections and choose the final wording. **Cannot be done by an agent:** the authors' own rewrite. |

## B. Internal review: minor issues

| Item | Change |
|---|---|
| m1 | "at every step" for departure removed (00, 01, 08). |
| m2 | "no violation pays at the optimum of the penalised objective; learned policies can still violate" (s4 §S4.1). |
| m3 | Sign convention defined once in 05 §5.3 and in the Fig. 5 caption. |
| m4 | Nomenclature Table S1. Conformal scores renamed S_j → e_j, n → n_cal. Minibatch is n_mb. Inner/outer taper slopes are ϱ^in/ϱ^out (ρ_tr kept for the rating). The taper auxiliary is renamed ψ. |
| m5 | Feature table, Table S3 (s2 §S2.4). |
| m6 | 03 §3.7: POMDP with a feed-forward actor and critic. |
| m7 | J^π is defined in Eq. (objective) as the undivided episode profit in €. |
| m8 | Table 1: GNN-DT "near-optimal reward vs. soft-penalty optimum (abstract)", guarantee cell in two layers, EV-GNN gap cell made specific. |
| m9 | Table 2 placeholder header fixed. The share of EVs without infeasible steps is in T3b, because the generated T1 (analysis) has fixed columns. |
| m10 | New macros `\restable`, `\resultfig` and `\analysisdir` fall back to local `tables/` and `figures/` copies. |
| m11 | Fig. 2 caption is two short sentences. |
| m12 | Efficiencies 0.90/0.93 in 05 §5.1, s2 §S2.2 and Table S2. |
| m13 | Admission is shared by all methods (05 §5.2, DD31). |
| m14 | "optional" removed. Compute budget comes from code README (core and extended) in 05 and s4 §S4.5. Hardware is a \TODO. |
| m15 | Unbiasedness qualified to the vanilla PG / PPO surrogate gradient at θ_old (04 §4.7). |
| m16 | Floating-point guard sentence in 04 §4.3 and Alg. 1. |
| m17 | Procedural abstract sentence replaced. The abstract ends with one \RES. |
| m18 | Keyword "Safety layer" replaces "Attention mechanism". |
| m19 | Merged bib (114 keys) has archival venues and years. `refs.bib` is re-extracted (79 cited keys). Preprints are marked in the text (Wan, Yan, Markgraf). Remaining "arXiv preprint" entries are true preprints (Dalal, Reddy, Ba, Schulman PPO). |
| m20 | "Early size-independent states" (02 §2.3). |
| m21 | Complementarity argument and check (03 §3.6, s3 §S3.2). |
| m22 | Three seeds for w ∈ {1,100} stated in 05 §5.2 and Table S4. |
| m23 | Conclusion rewritten around the two-layer result and three CPO-relevant limitations. |
| m24 | Full proof of A3 in EV2Gym (Lemma S3 = a3, s2 §S2.2). The closed-form check is stated with a \TODO for the authors to re-run it. |
| Q1-Q14 | Q1: 01/04 state the laxity extension and the closely related sLLF form. Q3: eMPC-HiGHS-24h. Q4: T3b and Fig. 4. Q5: DD31. Q6: deploy-only ablation. Q7: 04 remark and 07. Q9: Deep Sets. Q14: T6/Fig. S4. Q2, Q8, Q10-Q13 are left for the authors' response letter. |

## C. Consistency report

| Item | Change |
|---|---|
| **B1** AI statement | See M11. |
| **B2** Novelty rests on abstract-only works | See M1. One honest abstract-only statement, no "unknown" cells. Full-text reading remains an author task. |
| **B3** Tuning text vs code | 05 §5.4 and s4 §S4.3 follow DD37: the training mixture for tuning runs, 24 validation episodes from the mixture, selection on J^pen. Code enforcement ("refuses final training without a tuning result") is reported in code README/DD37 and is not changed by A5. |
| M1 ablation set | 05 §5.2: β=0, curriculum, small network (trained); no admission and deploy-only (evaluation); dense penalty and w ∈ {1,100} for Attention+penalty. |
| M2 gap exclusion | 1 € threshold in 03 §3.6 matches analysis README (`NaN if |J*| < 1 €`, count reported). |
| M3 Fig. 4 content | Fig. 4 now plots against N (analysis README). The pooled version is Fig. S6. |
| M4 T3 columns | Placeholders now ask T3b for fallback cases, deficit-bound ratio and capped/reduced requests. |
| M5 cost unit | € per EV and % of \|J^LP,ceil\| (T5 and fig6, analysis README). |
| M6 sensitivity / literature outputs | TS2 and TS3 placeholders in s5. The penalty weights are in TS2 only. |
| M7 fallback mapping | (i)=FB-1*/3, (ii)=FB-2/2, (iii)=FB-1/1, (iv)=FB-0/0 in 04 §4.5 and s1 §S1.6. "Three cases, the third with a sub-case" (m14). |
| M8 methods list | 05 §5.2 lists every method that appears in names.py, including eMPC-HiGHS(-24h), Gurobi eMPC, the EV-GNN zero-shot/in-distribution variants, layer variants and SA-Attn point prediction. |
| M9 shortfall reference | One admitted requirement for all methods (DD31). |
| M10, M11 | 01 sentence split. TD3 sentence cites fujimoto plus orfanoudakis2026can with "intended to". |
| M12 | Numbers taken from the verbatim checks in citation_fixes: GNN-DT 25 → 5-100, HetGPS 1,236 → 2,239 and 3,218. EV-GNN 1000 CPs / 70 groups (preprint). |
| M13 | Working title per DD30. "Size-agnostic" used throughout. |
| M14 | Bib upgraded (see m19). **Not done:** IEEEtran.bst is not installed, and the CTAN download is blocked by the proxy (HTTP 403). `main.tex` switches automatically when the template's .bst is present. |
| m1-m8 | Citation wording fixed in 01/02 (closely related sLLF form, action-replacement layers, jin2026 "evaluated in simulation", "early", action-correction layers, "intended to"; tabas+ev2gym for penalty failures; gu/su review wording). |
| m9a-h, m10 | Terminology: "cost of the guarantee", "charger" (no "port"), "re-implementation", "perfect-information optimum/LP" in our text, "Zero-shot transfer across station sizes", "Attention+penalty" for the twin, one term "heuristic filters", `\method{}` everywhere with "safe attention-based controller". |
| m11a-f, m12, m28, m36 | Normal(·) for the Gaussian; θ, ω and e_i defined; n_cal; commanded vs reported power defined once; auxiliary variables renamed; no-taper formula qualified; deficit-bound hypotheses stated; PPO symbols in Table S4. |
| m13 | Unreferenced numbered equations removed or used (eq:tau, eq:policy, eq:phi, eq:reach and eq:lclosed are gone from the main text). |
| m15 | A6 argument based on EV2Gym's load clipping and a 6σ bound (s2 §S2.3). |
| m16 | Cap "about 10⁻⁶ kWh, lower only where needed" (03). |
| m17, m18 | Alg. 1 has non-finite sanitising, l clip, tolerance and saturation rule. The regulariser applies to feasible steps (04 Eq. reg, Alg. S1). |
| m19, m20 | Missing parameters added to Tables S2/S4 (min stay 180, E_min 5 kWh, Adam ε, log σ clamp, advantage normalisation, deterministic test action, GBR 200×3, n_train 2000, β=0 for penalty methods, small-network head 32, tolerance units). |
| m21a | Main Fig. 5 = fig6_cost_of_guarantee. Example day = Fig. S7, pooled constraints = Fig. S6. Documented in README. **The analysis README plan still lists fig5_example_day as main Fig. 5: please align it there.** |
| m21b | Supplementary tables are mapped to analysis files in one sentence (s5). The pipeline figure is cited only in s5. |
| m22 | T6 placeholders ask only for inference ms/step and training time. |
| m23, m30 | Length (see M10). Novelty claim stated once. |
| m24 | "Grid quantities and the arrival rate scale linearly with N". |
| m25 | "compare with published EV2Gym numbers only on the literature configuration (Table TS3)". |
| m26 | "exact projection computed in O(N log N)" instead of "closed-form". |
| m27 | PDF CreationDate timezone: not changed (minor; can be stripped at submission). |
| m29, m31-m34 | Mask sentences split into "First/Second"; departure "met at departure"; "two-sided sum constraint"; one status for EV-GNN and the 5-min study; SA-Attn-CP trained once at α = 0.1 and recalibrated. |
| Section 2 (AI-run statistics) | Scenario-generator statistics are labelled as design checks with a \TODO for the authors to re-measure. |

## D. Deviations and open items for the orchestrator

1. Figure numbering: the example day is Fig. S7 (file fig5_example_day.pdf), and figS6_constraints_pooled is Fig. S6. The analysis README plan should be aligned.
2. IEEEtran.bst is unavailable (proxy 403). The ieeetr fallback is still in use.
3. saucis.cls is unavailable, so the page count in the official class is unknown.
4. No analysis output exists for the share of profit from V2G discharging or for an explicit information/policy gap decomposition.
5. Author tasks: full-text check of R1-R3; final title; rewriting the AI-drafted sections; result interpretation; hardware; repository link; confirming the scenario-generator statistics and the closed-form check.
