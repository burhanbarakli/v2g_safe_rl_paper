# Internal review, round 2 (A6, referee simulation for SAUCIS)

Reviewer: A6, round 2. I did not write the paper and did not do round 1. Date: 2026-10-22.

**What I reviewed.**
- `paper_repo/manuscript` (main.tex, sections/*.tex, main.pdf with 23 pages in the fallback article class) and the mirror `manuscript.md`.
- `paper_repo/supplementary` (sections s0-s5, 19 pages).
- `review/internal_review_round1.md` and `paper_repo/response_to_internal_review_round1.md`.

**What I checked claims against.**
- `method/safety_layer.md` (Theorem 1′, §5).
- `lit/merged/citation_fixes.md` (R1-R5).
- `analysis_repo/README.md` and `analysis_repo/scripts/{tables,stats,load}.py`.
- `code_repo/code/v2gsafe/{envs/wrapper.py, eval/metrics.py, baselines/{evgnn.py, evgnn_adapter.md, mpc.py}}`.
- The SAUCIS requirements (project doc `saucis_journal_requirements.md`).

**Location format.** `NN:line` refers to `manuscript/sections/NN_*.tex`, and `sK:line` refers to `supplementary/sections/sK_*.tex`. Results are placeholders by design. I judge the design, the theory, the claims and the writing, and whether the placeholders can be filled from the pipeline as it stands.

**Short verdict.** The revision is substantial and mostly honest:
- The two-layer theorem is now stated correctly, and the main text matches the supplementary proofs.
- The soft-penalty optimum is a genuine upper bound for every compared method. Section 4 states the conditions.

Four problems remain:
1. The paper is still too long for SAUCIS, and its method section is formally heavy.
2. The closest work (R1) is still unread.
3. The AI statement contradicts itself.
4. The text and the analysis pipeline have drifted apart, in the statistics and in a pre-declared primary endpoint that nothing computes.

---

## 1. Status of round-1 major issues

| Item | Status | Evidence (location) | What remains |
|---|---|---|---|
| **M1** Novelty / positioning | **Partially resolved** | Hedges gone, and Table 1 has no "unknown" cells (02:79-94). The novelty claim appears once (02:69). The laxity extension is stated (01:30, 04:144). The contributions are recast (01:32-39). | R1 (`orfanoudakis2026can`), R2 (`eland2026safe`), R3 (`song2025longterm`) and R4 (published EV-GNN) are still unread (citation_fixes §3). The novelty sentence 02:69 rests on R1, a GNN-TD3 with a safety layer that "preserves charging requirements and enforces power limits", from the group most likely to review the paper. The EV-GNN row cites the published *Communications Engineering* article (refs.bib) but fills its cells "(preprint)" (02:84). That version is open access, so it can and must be read. R2 is unread but is not named in the abstract-only sentence (02:68). Author task, before submission. |
| **M2** Guarantee oversold | **Resolved** | Theorem 1(a)-(d) at 04:178-186. Abstract 00:12-15, C1 01:34 and conclusion 08:74-76 follow the two-layer statement. "At every step" for departures is removed. The A3 hardware remark is at 04:197-198 and 07:45-47. | Small residues are listed as N-m9: Fig. 1 caption, Table 1 "hard per EV", the "absorbs" claim, and "requested" vs "admitted" in the abstract. The headline empirical half, the share of EVs with no infeasible step, is not computed by the pipeline (N-M2). |
| **M3** Admission / oracle asymmetry | **Partially resolved** | The request pipeline is computed once per replay and shared (03:39-54, s3:5-36, wrapper.py:228-236). The soft-penalty optimum bounds every controller (Prop. S3, s3:97-107; I re-derived it, see §4.2). The 1 € exclusion and its count are produced (load.py:318). | (a) The external baselines do **not** receive the admitted requirement (N-M3). (b) The cost of the guarantee is still a raw profit difference between a controller that may under-serve and one that does not (05:64-66). Without a break-even weight or a service-adjusted difference it cannot be interpreted (N-m13). (c) The "ratio of sums" promised at 03:94 is not computed (N-M2). (d) The conformal variants are excluded from the main gap unnecessarily (N-m3). |
| **M4** Zero-shot validity | **Partially resolved** | The false profile claim is deleted (05:22). A held-out stress rating of 3 kW is used (05:21). The SA-Attn-CP calibration is said not to be zero-shot (04:239). There is an N=50 in-distribution reference (05:43, 05:67). Linear scaling is a stated limitation (07:61). | The per-N **binding rate** promised at 05:25 has no metric in `metrics.py` and no column in T2 (N-M2). It is the only quantity that tells a reader whether large-N instances are easier, so without it the transfer claim cannot be interpreted. The stress level exists only at N ∈ {20, 200}. Sub-linear rating and leave-one-profile-out were declined (acceptable with the limitation). The N=50 reference is itself zero-shot at N=100 and 200 (N-m18). |
| **M5** Missing baselines | **Resolved** (residues minor) | No-l_i projection, deploy-only, Lagrangian, Deep Sets (parameter-matched), sLLF, price-threshold rule, eMPC-HiGHS 10-step and 24-h, MLP restricted, trainN50 (05:30-56, s4:5-48). | The residues are minor: the MLP on the 25-charger literature configuration has untrained slots (N-m11), the Lagrangian cost limit is exactly 0 (N-m12), the no-l_i and deploy-only ablations appear only in Table S9 although C2 rests on them (N-m14), and no 24-h MPC runs without arrival foresight (N-m15). |
| **M6** Baseline protocols | **Partially resolved** | The GNN-DT statement is corrected with a valid reason (05:51-52, s4:46-48). The EV-GNN budget and the layer mapping are stated (05:49-50, s4:44-45). There is a single HiGHS MPC at all N (05:46-47). | The EV-GNN protocol hides two handicaps: training on a single load/PV draw per run (`evgnn_adapter.md` §2 item 3: `tr_seed: -1`), and 3×10⁶ steps against its default 10⁷. It also pursues the full-battery request rather than the admitted one (N-M3). None of this is in 05:49-50, s4:44 or 07:63-64. |
| **M7** Tuning / statistics | **Partially resolved** | Tuning is symmetric on the mixture, includes w for the penalty methods and selects on J^pen at w=10 (05:71-75, s4:78-86). The seed is the unit (05:77). The endpoints are pre-declared (05:78). | The stated test is not what the code does, and it is not a valid design: "paired t-tests on seed means" (05:79, s4:131) against Wilcoxon-on-replays primary plus Welch secondary in `stats.py:286-357`. The Holm families differ (N-M1). The "10 seeds if intervals overlap" rule is optional stopping (s4:135). E3 and E4 have no test file (only `tests_T1/T2/T5`). β and the architecture are untuned. This is acceptable if stated, and it is stated (s4:86). |
| **M8** Oracle relaxation | **Resolved** | Outer taper rows with a proof (Lemma S7, s3:53-69). J^LP vs J^LP,ceil in TS1 (s5:82-83). Complementarity remark (s3:71-73). | The main text still calls the rows "exact up to the step discretisation" (03:78, 07:66). It should say "outer rows, exact except on a one-step transition band and a 1 % margin", so the bound property is visible (N-m2). The number of LP steps with simultaneous charging and discharging (s3:73) is not produced (N-M2). |
| **M9** V2G economics | **Partially resolved** | Throughput-cost and discharge-factor sensitivities at N=20 (05:26-27, s3:51, TS2). Lossless accounting is a limitation (07:53-56). | The profit share from discharging is not produced (acknowledged). The sensitivities run at one size and one rating only. That is acceptable for SAUCIS if 07:55 stays. |
| **M10** Length / fit | **Not resolved** | 23 pages in single-spaced 10-pt article. 8,535 words of running text, 9,533 with captions and equations, 13.9k words in the PDF including back matter and 79 references. 7 floats plus Algorithm 1 (DONE.md, `pdfinfo`). | SAUCIS guidance is ≤ 25 *double-spaced* pages and ≤ 7 figures and tables for experimental papers. 23 single-spaced pages are roughly 40 double-spaced pages, and the Results and Discussion will grow by about 800-1,200 words once filled. Algorithm 1 is an eighth float. The build has still not been made with `saucis.cls`. See N-M4 and §5 change 3. |
| **M11** AI policy | **Partially resolved** | Detailed disclosure in the Article Information Form and in Acknowledgments (10:108, 10:131-142). Tool and version named. TODO to rewrite. No pre-written interpretation in 06/07. | The statement contradicts itself (§4.6). It says the AI "wrote … the analysis pipeline that computes every reported statistic, table and figure" (10:136), and the generated tables are inserted automatically (`\restable`). It then says "No AI tool generated, reported or interpreted the research results" (10:141). It also does not say who will run the experiments. "A separate AI agent independently re-checked the derivations and proofs" (10:134) reads as independent verification, which it is not. The rewrite of the abstract, introduction, related work and discussion by the authors is still outstanding. |

Round-1 minors m1-m24 are addressed, with two exceptions:
- m19 is partial: IEEEtran.bst is unavailable and ieeetr is used.
- m24 keeps an author TODO (s2:22).

The response document mislabels two items:
- "Lemma S3 = a3" should be S6.
- "Lemma S1 = taperrows" should be S7.

**Count: 3 resolved (M2, M5, M8), 7 partially resolved (M1, M3, M4, M6, M7, M9, M11), 1 not resolved (M10).**

---

## 2. New major issues

### N-M1. The statistical plan in the paper does not match the analysis code, and the stated test is not a valid design
**Where.**
- The paper: 05:77-80 and s4:127-135 say "paired t-tests on seed means … Wilcoxon … as a secondary check", with "Holm correction applied over each results table".
- The code: `analysis_repo/scripts/stats.py:286-357` runs the Wilcoxon signed-rank test on replay-level differences of seed-averaged policies as the *primary* test, with Holm families "across N within (comparison, metric, profile, ρ)" and "across all configurations". The Welch *unpaired* t-test on seed means is secondary.

**Why it matters.**
- Seeds of two different methods are independent training runs. Pairing seed k of SA-Attn with seed k of Attention+penalty has no basis, so a paired t-test on seed means is the wrong test.
- When the results come in, either the text or every p-value will be wrong. The cost of the guarantee (E2) is expected to be small, so it is exactly where the choice of test decides the conclusion.
- "If the intervals of a primary endpoint overlap, … ten seeds" (s4:135) is a data-dependent stopping rule. It inflates the type-I error unless it is pre-specified with a correction.

**Fix.**
1. State the code's design, which is sensible:
   - unit = seed;
   - primary inference = stratified-bootstrap 95 % CI of the difference in seed means, plus the Welch t-test on seed means;
   - secondary = Wilcoxon on replay-level differences of seed-averaged policies.
2. Name the Holm family exactly as implemented, or change the code to "the pre-declared primary comparisons E1-E4".
3. Either drop the adaptive 10-seed rule, or decide on 5 or 10 seeds now.
4. Add the test files for E3 (overload energy) and E4 (share of EVs without infeasible step). Today only `tests_T1/T2/T5.csv` exist.
5. With 5 seeds, the Welch test has low power. Report CIs as the primary evidence and say so.

### N-M2. The Results skeleton asks for quantities the pipeline does not compute, including a pre-declared primary endpoint
**Where.**
- **E4, the share of EVs whose stay contains no infeasible step.** It is promised in the abstract (00:18), 01:34, 05:62, 05:78 (primary endpoint), 06:61 ("per N, Table T3b"), 07:17 and 08:79.
  - No table column computes it: T3b (`tables.py:360-394`) has infeasible steps by case, the deficit ratio and capped/reduced requests.
  - `n_evs_touched_infeasible` is logged (`metrics.py:171`) and loaded (`load.py:64`) but never tabulated or tested.
  - T3b is pooled over the grid, not per N.
- **Per-N binding rate** of the forward limit (05:25): there is no metric in `metrics.py`.
- **Ratio-of-sums gap** (03:94): not computed anywhere in `analysis_repo/scripts`.
- **Number of LP steps with simultaneous charging and discharging** (s3:73): not output by `oracle.py`.
- E4 is a primary endpoint but has no main-text display. Fig. 4 shows only energy not served and overload (06:58-69).

**Why it matters.**
- The authors cannot fill 00:18, 06:61, 07:17 or 08:79 without new analyses, which the brief says the skeleton must avoid.
- The empirical half of the guarantee, "how often does the conditional part fail versus N", is the result that turns Theorem 1(c) into evidence.

**Fix (all small, before any run).**
1. In `metrics.py`, add `n_evs` and `n_evs_certified`: EVs with p_i(t) ≥ l_i(t) at every step of their stay. This is exactly the hypothesis of Theorem 1(c) and is larger than "no infeasible step".
2. Add `n_evs_no_infeasible` (= n_evs − n_evs_touched_infeasible).
3. Add `shortfall_among_certified_kwh`. It must be 0, and reporting it is a one-line empirical check of the theorem.
4. Add `binding_steps`: steps with λ* > 0 or Σp = C̃.
5. Add the ratio-of-sums gap and the LP simultaneity count.
6. Make T3b per N, or add the E4 column to T2, and add `tests_T3b.csv`.
7. Add E4 as a third panel of Fig. 4.

### N-M3. "Every method uses the same admitted requirements" is not true for the external baselines, and the EV-GNN protocol hides two handicaps
**Where.**
- The claim: 05:32, s3:8, s4:7 and the cover letter.
- `wrapper.py:213-236` reads `ev.desired_capacity` (the full battery) and stores the admitted `E_req` only in its own slot. It never writes the value back to the EV2Gym EV object.
- EV-GNN's state function `V2G_ProfitMax_with_Loads_GNN` and its reward read EV2Gym's own EV objects (`evgnn.py:46-48`). So EV-GNN targets the full battery, not the admitted requirement. The same holds for the original Gurobi eMPC on the literature configuration.
- eMPC-HiGHS uses the uncapped, unadmitted request for arrivals inside its horizon (`mpc.py:202`).
- EV-GNN's training environment keeps "the loads/PV draw of the constructor seed for all episodes of a run" (`evgnn_adapter.md` §2 item 3). It is trained for 3×10⁶ steps against a default of 10⁷.

**Why it matters.**
- The J^pen bound still holds (§4.2), but the comparison does not.
- EV-GNN buys energy that is never credited, so its profit and gap are depressed by a target mismatch rather than by the method. EV-GNN is the only external, published, size-agnostic learned baseline, and an EV2Gym-group referee will check exactly this.
- Undisclosed training handicaps undermine every zero-shot comparison with EV-GNN.

**Fix.**
1. In the EV-GNN adapter and the Gurobi-eMPC adapter, set `ev.desired_capacity` to the admitted E_req at arrival (in EV2Gym's units) before the baseline's state function runs. Do the same in `mpc.py:202` for future arrivals, using the shared table entry.
2. Report "over-delivered energy" (E(d+1) − E^req)^+ per method in T3. It is cheap and makes target mismatches visible for every method.
3. Re-seed EV-GNN's training loads/PV per episode with a thin driver, or disclose the fixed draw and the reduced budget in 05:49-50 and 07:63-64.

### N-M4. The method section is hard to follow for a SAUCIS readership, and too long
**Where.** Section 4 has 2,930 words and carries:
- six numbered assumptions (04:42, 04:147-165);
- Lemma 1, Proposition 1, Theorem 1 with four parts, Proposition 2 and a deficit-bound equation;
- Algorithm 1;
- fallback cases (i)-(iv) "the third with a sub-case", plus code labels FB-1*/FB-2/FB-1/FB-0 (04:209-217);
- the operators φ, R, Ψ_k, B, B̂, of which R and Ψ_k are used only to define the cap (03:43, 04:47).

Two of the six assumptions are not assumptions:
- A4 is enforced by construction (04:168).
- A5 is not assumed by the main result (a)-(c). It is an event that is measured.

Under A6, three of the four fallback cases cannot occur (04:216), yet all four are explained in the main text.

**Why it matters.**
- SAUCIS is a CS-generalist journal whose scope list names no energy topics.
- A reader who meets "A1-A6" in the Fig. 1 caption (04:12) will conclude that the guarantee is assumption-laden, which is the opposite of the paper's improved message.
- Together with M10, this is the most likely reason for a "too long / not accessible" decision.

**Fix (≈ 700-900 words saved in §4, no loss of content, since the supplement already holds the details).**
1. **Reduce the visible assumptions to three.**
   - A1 (monotone full-power map).
   - "S (simulator and grid conditions)" = A3 + A6.
   - "A2 (declared departure)", folded into the problem statement in 03:15.
   - State A4 as a property of the request pipeline (03:53).
   - Turn A5 into a *definition* ("a step is feasible if …"), used only in (d).
2. Keep Theorem 1 with three parts: (a)+(b) merged, (c), and (d) as a one-line corollary.
3. Move Lemma 1, the fallback cases (i), (ii) and (iv), the FB code labels and the deficit-bound equation to the supplement. Keep one sentence each: "Under S only case (iii) can occur: least-laxity-first …" and "the shortfall is bounded a priori (Lemma S4)".
4. Drop R and Ψ_k from the main text. Define the cap directly as the largest E^req with E^arr ≥ B̂(τ(t^arr)).
5. Shorten §4.6 (conformal) to the statement of Proposition 2 plus two sentences.
6. Shorten §4.7 (training) to the likelihood-ratio and Moreau points.
7. Add a 3-row "symbol at a glance" box (l_i, u_i, C̃) beside Fig. 2, so the reader does not need Table S1.

---

## 3. New minor issues

- **N-m1** 03:43: Eq. (cap) defines E^req with "≤". Define E^req as the output of the pipeline ("E^req_i is the admitted request"), and state "E^req_i ≤ min{…}" as a property.
- **N-m2** 03:78 and 07:66: write "outer (relaxing) linear taper rows, exact except on a one-step transition band and a 1 % margin (Lemma S7)". "Exact up to the step discretisation" hides that this is what makes the LPs valid upper bounds.
- **N-m3** 03:88, s3:98, s3:109, s5:53:
  - The soft-optimum bound is a property of the *measurement*, not of the controller: J^pen(π) ≤ J^pen,LP for every simulated trajectory when shortfall is measured against the shared admitted requirement.
  - Rephrase Proposition S3 that way.
  - Then report the conformal variants and the no-admission passes in the main gap, with shortfall against the shared requirement. Today `wrapper.py:220-223` measures the conformal variant against its own, lower E^req, which breaks the bound, and the no-admission pass against the cap-only E^req.
  - Say in 05:41 which reference the no-admission passes use.
- **N-m4** Propositions S2 and S3 (s3:93) use "EV2Gym never delivers more than φ(E) − E". Lemma S6 proves exact delivery only for commands ≤ u_i.
  - All wrapper methods pass the physical clip, so the step is fine for them.
  - The Gurobi eMPC adapter (s4:41) must also be clipped. Otherwise, add one line proving that delivered energy is monotone in the pilot.
- **N-m5** A6 (04:169, s2:94-100):
  - A6 is exogenous, so verify it *exactly* on every training and test replay and report "A6 held on all … replays", instead of a 6σ Gaussian-tail argument that depends on EV2Gym internals.
  - Say whether m_L is drawn once per episode. The 2×10⁻⁹-per-episode figure assumes it.
  - s2:100 ("logged as case (ii) or (iv)") and s1:161 ("negative budget in case (i) … logged FB-0") disagree with 04:211 ("procedure of case (iii) with priority on the forward limit"). Align all three.
- **N-m6** 04:218-222: the main-text deficit bound omits the η^c = η^d = 1 hypothesis of Lemma S4 (s1:176-177). Add it.
- **N-m7** Proposition 2 and 04:228-240:
  - State that each calibration session is drawn *uniformly* from its episode; the exchangeability with "a uniformly chosen EV of a test episode" (04:238) needs this.
  - State that the layer sets τ^lo = 1 for t > d̂^lo (s1:202 uses it, but the main text never says what happens after the bound).
  - State that E^req in Proposition 2 is the conformal variant's own, lower requirement.
- **N-m8** Abstract 00:15: "extends the second result to undeclared departures at a stated coverage level" omits two things: the coverage is marginal, and it is reduced by P(∃t: p_i < l_i). Write "with marginal coverage of at least 1 − α minus the rate of EVs that receive less than their bound".
- **N-m9** Residual overclaims:
  - Fig. 1 caption 04:12, "preserves feasibility under Assumptions A1-A6": write "keeps the transformer limit under S, and each EV's departure feasibility while it receives its bound".
  - 04:197-198, "absorbs a bounded under-delivery", is not proven. Write "would absorb … (not analysed here)".
  - Table 1 (02:93), "Departure energy: hard per EV (bound l_i)": write "per-EV bound, conditional (Thm 1c)".
  - Abstract 00:6, 00:10, 00:14 say "requested energy" and "departure requirement". Write "admitted requirement (the request capped to what is reachable and admissible)" once, so the guarantee is not read as relative to the driver's original request.
  - Title: "Feasibility-Preserving" is defensible only with the theorem in mind. Consider "…with a Transformer-Safe Projection…" (author's choice; the title is flagged as a working title at main.tex:32).
- **N-m10** 02:45 and 02:84: the EV-GNN row cites the published article, but its cells say "(preprint)". Read the open-access version (DOI 10.1038/s44172-025-00457-8) and fill the cells from it. Add `eland2026safe` to the abstract-only sentence in 02:68, or read it.
- **N-m11** 05:42 and s4:27-28: the MLP "serves … at N ≤ 20 and on the literature configuration". The literature configuration has N=25 > 20, so slots 21-25 never received a gradient. Train the MLP rows for TS3 on N=25, or drop them there.
- **N-m12** s4:18-22: a Lagrangian cost limit of exactly 0 on expected shortfall and overload drives Λ to Λ_max=1000, which turns the method into a large fixed penalty. Use a small positive limit, e.g. 1 % of the demanded energy, or the realised shortfall of SA-Attn per episode. Report the multiplier trajectories in Fig. S1.
- **N-m13** Cost of the guarantee (05:64-66, T5):
  - Add the break-even weight w* = ΔProfit / ΔEnergy-not-served per N. It is computable from the T3 and T5 outputs. The guarantee "pays" whenever unserved energy is valued above w* €/kWh.
  - Report J^pen differences at evaluation weights {1, 10, 100}. J^pen is linear in w, so this is post hoc and needs no retraining.
  - The hard-optimum gap of Attention+penalty (06:74) is not bounded below by zero, because Proposition S2 does not cover controllers with violations. Say so in the T5 caption.
- **N-m14** 01:35 claims the ablations "test what each part contributes", but the no-l_i projection and deploy-only variants appear only in Table S9 (06:88). Add both as rows of the main-text Table 2 (T1), which already holds Deep Sets.
- **N-m15** 05:46-47: eMPC-HiGHS-24h knows true arrivals within 24 h, so it is not the "why RL?" baseline. Add a "connected EVs only" variant of the 24-h LP; in `mpc.py:196-205` this means skipping the future-arrival loop. Without it, Q3 of round 1 remains unanswered.
- **N-m16** Generated tables leak internal labels into the paper:
  - T2 caption "(in-distribution reference, DD36)" (`tables.py:303-305`).
  - T3b notes "Lemma 3", "DD31" and "FB-1*" (`tables.py:370-376`). The paper's lemma is S4.
  - COMMON_NOTE refers to "STAT_METHODS.md" and "_values.csv files" and reports mean ± SD while the text promises CIs (`tables.py:206-209`).
  - Main-text placeholders cite pipeline IDs ("Table T2/T3b/T5/TS2", 06:44-75, 07:11-20), but in the PDF these are Tables S5, S7, S8 and S13. Use `\Tab{S-tab:t2}` inside the RES text, so the authors cannot copy "Table T2" into the paper.
- **N-m17** 05:16 vs Table S2 (s2:129): PV "mean peak 0.4 P_tr" vs "about 0.38 P_tr". Make them consistent.
- **N-m18** 05:43 and 05:67: the N=50 reference is in-distribution only at N=50. At 100 and 200, the "transfer gap" compares two extrapolations (from 10/20 and from 50). Say so in the T2 caption and in 06:45.
- **N-m19** Abstract 00:14: "aggregate-infeasible step" is undefined in the abstract. Add "(a step at which the transformer cannot give every EV its lower bound)".
- **N-m20** preamble.tex:49: `\method` = SA-Attn is marked as a placeholder ("fix at G2"). Fix the name now, because figures, the analysis labels (`names.py`) and tables must match.
- **N-m21** 04:89 and 04:164: briefly explain why A6 uses N while C̃ uses n(t): N makes A6 policy-independent, and n ≤ N suffices.
- **N-m22** Figures: Fig. 3 (gap vs N) and Fig. 4 (constraints vs N) share the x-axis. Merging them into one two- or three-panel figure (adding E4, N-M2) keeps the float count at 7 even if Algorithm 1 is counted.

---

## 4. Answers to the specific checks requested

### 4.1 Theorem 1 (= Theorem 1′ of `safety_layer.md`) and A6: main text vs supplement
- **Statements agree.** The main text (04:163-186) and the supplementary proof (s1:111-145) agree with `safety_layer.md` §5.1-5.4 on:
  - A6 (C ≥ Nε_q, C_rev ≥ 0);
  - (a) and (b) under A3 + A6;
  - (c) under A1-A4, per EV, pathwise;
  - (d) under A5.
- **I re-checked each step and found no error.**
  - Feasible steps: (a) holds because Σp̃ ≤ Σp + nε_q ≤ C̃ + nε_q = C. (b) holds because [l, u] ⊆ [p^min, P̄], which the min/max guard ensures even for individually infeasible EVs.
  - Infeasible steps under A6:
    - C̃ ≥ (N − n)ε_q ≥ 0 ≥ −C_rev excludes case (i);
    - u ≥ 0 excludes case (ii);
    - p^min ≤ 0 excludes case (iv).
    - In case (iii) the budget is exhausted, so Σp = C̃ ≥ −C_rev.
  - (c): the per-EV induction uses only p_i ∈ [l_i, u_i] during the stay, and the upper half comes from (b).
  - The two-EV example (04:200-202) is correct.
- **Clarity gaps (minor).**
  - The η = 1 hypothesis of the deficit bound is missing from the main text (N-m6).
  - The case (i) handling differs between 04:211 and s1:161 and s2:100 (N-m5).
  - The behaviour for t > d̂^lo is unstated (N-m7).
  - (c) silently needs A3 ("also" refers back to (a)). This is correct but worth one word: "If A1-A4 hold, …".
- **A6 probability.** The 6σ argument is right if m_L and m_P are drawn once per episode: P(Z > 5.97) ≈ 1.2×10⁻⁹. Because A6 is exogenous, it should simply be checked on every replay (N-m5).

### 4.2 Is the soft-penalty optimum an upper bound for every compared method?
**Yes, with the conditions spelled out.** I re-derived the proof of Proposition S3 (s3:97-107):
- Take p^± = p̃^± (billed powers), E = E^nom (billed energies), z = shortfall against the shared E^req, and o^± = realised overloads. This point is feasible for the soft LP.
  - The power rows hold because p̃ ≤ P̄(E^real) ≤ P^nom and p̃ ≥ −P^dis.
  - The outer taper row holds because p̃ ≤ ϱ_κ(E^cap − E^real) ≤ 1.01 ϱ_κ(E^cap − E^nom), since E^nom ≤ E^real (Lemma S7).
  - The floor and departure rows hold because of the rounding allowance G^max.
  - The overload rows hold by definition of o^±.
- At that point the LP objective equals J^pen(π).
- The bound uses only EV2Gym physics on the shared replay, never the controller's internals. It therefore covers the heuristics (raw and with the layer), the Lagrangian and penalty methods (with violations), eMPC-HiGHS, EV-GNN (which runs inside our wrapper on the pinned commit, `evgnn_adapter.md` §1) and the deploy-only variants.

Caveats:
1. It requires shortfall to be measured against the shared admitted requirement. That is true for all wrapper methods except the conformal and no-admission passes, which use their own E^req (`wrapper.py:220-236`). See N-m3.
2. It requires commands to be physically clipped to [p^min, u], or a monotone-delivery lemma. That holds in the wrapper; check the Gurobi eMPC adapter (N-m4).
3. The sensitivity runs need their own LP economics (c^thr, f^dis) in both J^pen and the LP. s3:51 shows this for the LP. Confirm it for the wrapper reward.
4. "Bounds every method" does not mean "same targets": EV-GNN optimises toward the full battery (N-M3).

### 4.3 Remaining overclaims
- Title, abstract, contributions and conclusion are now calibrated.
- The remaining residues are listed in N-m8, N-m9 and N-m19.
- The novelty sentence (02:69) stays a risk only through R1 (M1).
- "a perfect-information optimum that bounds every compared method" (01:36, cover letter) is true after N-m3.

### 4.4 Internal consistency
- Method names are consistent across 05, s4 and the T1 placeholder. `\method` is still a placeholder (N-m20).
- Metric names (gap^pen, energy not served, overload, cost of the guarantee) are consistent.
- The inconsistencies are the following:
  - the statistics (N-M1);
  - table IDs in placeholders vs PDF numbering, and internal labels in generated captions (N-m16);
  - PV peak (N-m17);
  - fallback case (i) (N-m5);
  - two lemma numbers in the response document (§1).
- All cross-references resolve: 0 undefined references in both builds.

### 4.5 Readability, length and floats
See N-M4 and M10. The main text has the following floats:
- Figures 1-5 and Tables 1-2, which is 7.
- Algorithm 1, an eighth float object. SAUCIS's template has a separate "Algorithms" item, so it may not count, but do not rely on that.

Running text is about 8.5k words, and the Results and Discussion are still placeholders. Target ≤ 7,000 words of running text before results are filled, as follows.

| Section | Now (words) | Target (words) | How |
|---|---|---|---|
| §2 | 1,399 | ≈ 850 | Merge 2.1 into the Introduction; compress 2.3-2.4 |
| §3 | 1,260 | ≈ 950 | The oracle construction and rounding allowance go to s3; keep the J^pen definition and one sentence per optimum |
| §4 | 2,930 | ≈ 2,100 | N-M4 |
| §5 | 1,345 | ≈ 1,050 | List the compared methods in one compact paragraph per family |

Build once with the real `saucis.cls` (9 pt one-column) before submission.

### 4.6 AI statement vs the SAUCIS AI policy
The policy (11 April 2025) makes four points:
- AI "must not be employed to produce, report, or interpret research findings".
- AI-generated substantial sections (abstract, introduction, literature review, discussion) are "inappropriate" unless they are preliminary drafts thoroughly revised by the authors.
- Disclosure goes in Methods or Acknowledgements, with the tool's name and version.

The current statement (10:131-142) is candid but not yet accurate or compliant:
1. "it wrote … the analysis pipeline that computes every reported statistic, table and figure" (10:136) together with auto-inserted tables (`\restable`) and an AI-generated `findings_numbers.md` contradicts "No AI tool generated, reported … the research results" (10:141). Write exactly what happens:
   - who executes the training and evaluation runs;
   - that the tables are produced by deterministic scripts that the authors reviewed;
   - that the authors verified the outputs and wrote all result text.

   If AI agents execute the experiments, say so, and consider asking the editor beforehand whether this counts as "producing findings".
2. "a separate AI agent independently re-checked the derivations and proofs" (10:134) should not be presented as verification. The authors must check the proofs themselves and say so.
3. "drafted all sections" (10:137) is accurate today. It is compliant only after the authors' rewrite, which the TODO at 10:142 already requires. Until then, submission is a desk-reject risk.
4. Confirm the product names actually used. The agent sessions of this workspace run as Claude Code / Agent SDK sessions, and the statement names only "Claude Cowork". List every interface used.

### 4.7 Will the Results skeleton work without new analyses?
Mostly. T1, T2, T5, T7, TS1-TS3, Fig. 3 and Fig. 5 map one-to-one onto generated outputs. The exceptions are listed in N-M2:
- E4;
- the binding rate;
- the ratio of sums;
- LP simultaneity;
- per-N T3b;
- tests for E3 and E4.

Add the break-even weight of N-m13. All are cheap to add now and expensive to add after the runs.

---

## 5. Recommendation and the five most valuable remaining changes

**Recommendation, if the experiments turn out as planned: major revision, borderline minor.**
- **Strengths.**
  - The theory is correct and now honestly stated.
  - The evaluation design is above the usual SAUCIS standard: a valid soft upper bound for all methods, a shared request pipeline, symmetric tuning and the decisive ablations.
- **What a referee would still return as major.**
  - Length and accessibility (M10, N-M4).
  - Positioning against the unread R1 (M1).
  - The asymmetric treatment of EV-GNN (N-M3).
  - Statistics that the paper describes differently from what it computes (N-M1).

With the five changes below, I would expect minor revision.

1. **Close the text-to-pipeline gaps before any run (N-M1, N-M2, N-m13, N-m16).**
   - Add E4 and the "certified EVs" metric with a zero-shortfall check, the binding rate, the ratio of sums, a per-N T3b, tests for E3 and E4, and the break-even weight.
   - Align the statistics text with `stats.py` (Welch/bootstrap on seed means, Wilcoxon secondary, exact Holm families) and drop the adaptive 10-seed rule.
   - Remove internal labels from the generated captions.
2. **Make the external baselines like-for-like (N-M3, N-m15, N-m11).**
   - Write the admitted E^req into `ev.desired_capacity` for EV-GNN, the Gurobi eMPC and the eMPC future arrivals.
   - Re-seed or disclose EV-GNN's fixed load/PV draw and its reduced budget.
   - Add a 24-h receding LP without arrival foresight.
   - Fix the MLP literature rows.
3. **Cut and simplify the main text (M10, N-M4, N-m22).**
   - Three visible assumptions; A5 as a definition; a three-part theorem.
   - Lemma 1, the fallback cases, the FB labels and the deficit equation go to the supplement.
   - §2 to about 850 words; about 7,000 words of running text.
   - Merge Figs. 3 and 4 and add the E4 panel.
   - Build with `saucis.cls`.
4. **Read R1-R4 in full and settle the novelty claim (M1, N-m10).**
   - Library access for `orfanoudakis2026can`, `eland2026safe` and `song2025longterm`; the open-access published EV-GNN.
   - Rewrite their Table 1 rows from full text and apply the gap-analysis decision rule.
5. **Make the AI statement exact and do the human rewrite (M11, §4.6).**
   - Say who runs the experiments and how tables are generated and verified.
   - Drop "independently" for the AI proof check, and have the authors verify the proofs.
   - Confirm the product names.
   - The authors rewrite the abstract, introduction, related work and discussion in their own words.
