# Internal review, round 3 (A6, referee simulation for SAUCIS, results-filled paper)

Reviewer: A6, final round. I did not write the paper. Date: 2026-10-24.

**Material reviewed.**
- `paper/manuscript` (main.pdf, 23 pp.) and `paper/supplementary` (supplementary.pdf).
- The mirrors and `paper/cover_letter.md`.
- The analysis outputs and the raw episode files. The numbers were audited separately in `review/consistency_report_round3.md`. This report gives the referee view and cites that report as "CR §x".

---

## 1. Summary of the submission

The paper proposes a safety layer for V2G charging control at a CPO station. The layer is an exact projection of raw per-charger actions onto per-EV bounds and a two-sided transformer limit, with a taper-aware lower bound that keeps each EV on a latest-start trajectory.

Theorem 1 has two layers:
- (a) the transformer limit holds at every step for arbitrary raw actions, under a one-step model and grid headroom;
- (b) every EV whose stay contains no infeasible step meets its admitted requirement.

Two size-agnostic set policies (attention, "SA-Attn", and Deep Sets) are trained with PPO with the layer in the loop on 10/20 chargers and tested zero-shot up to 200 chargers in EV2Gym. They are compared with penalty, Lagrangian, rule-based and MPC baselines, each with and without the layer, and with a soft-penalty perfect-information optimum.

After DD45 the paper is honestly reframed. It no longer claims an attention advantage, and it states that simple rules with the layer are competitive.

**Main empirical findings (verified from the raw data):**
- 0 overload steps in 102,750 layer episodes, against 3,274-3,730 overload steps for the penalty and Lagrangian attention policies.
- The episode-level departure check is 0 for all 13 non-conformal methods with the bound.
- Zero-shot transfer works in the narrow sense: within 0.04 € per charger of an N = 50-trained reference.
- Deep Sets beats attention at every N (Holm p < 1e-45).
- A price-threshold rule with the layer beats attention at every test-grid N.
- The learned policies trigger many infeasible steps: 17,678 for SA-Attn, against 1 for LLF+layer.

---

## 2. Assessment

### 2.1 Strengths

1. The theory is stated precisely and checked against the data. Every reported number traces to a generated table. I recomputed about 110 values from the raw episodes and found no value mismatch (CR §1).
2. The bound check (0 of 153,100 episodes above the soft optimum), the shared admitted requirement for every method (DD46) and the absolute gap (DD47) make the evaluation internally coherent.
3. The negative findings are reported plainly in the Abstract, Introduction and Conclusion: no attention advantage, rules with the layer competitive, learned policies cause infeasible steps. This is rare and valuable.
4. The limitations section is complete with respect to DD49: two seeds, not-run studies, redrawn training seeds, optimum tolerance sensitivity, lossless economics and the episode-level check.
5. The baselines are broad and fairly tuned. The penalty weight was selected at w = 10, the same as the evaluation weight.

### 2.2 Major issues

**M1. The paper does not report the main practical downside of its own departure bound (CR O1, O2).** This is the most important issue.

What the data show:
- The variant **without** the departure lower bound has a lower gap than SA-Attn at every N. The difference is 0.12 to 1.27 € per charger, with Holm p ≤ 1.3e-30.
- It also leaves **fewer** EVs unserved: 0.99 % vs 2.24 % of EVs, and 1,551 vs 6,158 EVs in total.
- At the stress rating it is far better: gap 0.98 vs 2.98, and 2.6 % vs 12.0 % unserved at N = 20.
- SA-Attn also leaves more EVs unserved (2.24 %) than its **penalty-trained twin without any layer** (1.86 %), the Lagrangian policy (0.20 %), the deploy-only policy (0.46 %) and the price-threshold rule with the layer (0.44 %).

What the text says: it reports only the conditional episode-level check, in which the no-bound variant looks worse because its 1,551 unserved EVs fall in episodes without infeasible steps. The Abstract leads with the conditional departure statement.

What a reader will conclude: the departure guarantee improves service. The data show the opposite for learned policies. The lower bound makes the policy ride its latest-start trajectories (p_i = l_i on 28-42 % of occupied charger-steps). The station then hits infeasible steps, and more EVs end up short.

The Discussion half-states the mechanism ("a profit-seeking policy keeps many EVs on their latest-start trajectory") but never quantifies the net effect. SA-Attn's J^pen advantage over its twin comes from the overload penalty, not from service.

A referee who reads T3 will see this immediately and will question the paper's central motivation.

Required changes:
- State these comparisons in Results.
- Discuss them in the Discussion as the price of a conditional guarantee.
- Note the confound: the no-bound variant was trained with β = 0.01 and the penalty reward, so it is not a clean ablation.
- Soften the Abstract accordingly.

The rules with the layer (LLF, AFAP, RR: 0 unserved EVs) show that the bound works when the upstream policy does not exploit it. That is the positive message to keep.

**M2. Abstract and Conclusion overgeneralise the departure check (CR C1).** "For every method with the departure bound" and "no EV was left short in an episode without infeasible steps" are contradicted by the conformal variants (81,742 such EVs at α = 0.01) and, in the Conclusion, by the no-bound variant (1,551). Results and s5 have the correct qualifier "non-conformal". This is a factual error at abstract level and must be fixed.

**M3. Statistics are reported selectively across the two inference tools (CR C8).** The primary test is a Wilcoxon signed-rank test, which tests a median shift. The CIs are bootstrap CIs of the mean. The two disagree at N = 200 in three key comparisons, and the text quotes only one side each time:
- Penalty twin vs SA-Attn gap: the CI includes 0, but Holm p = 2e-5. Only the CI is quoted.
- ΔJ^pen: the CI includes 0, but Holm p = 1.7e-6. Both are quoted, without explanation.
- E4 Deep Sets vs SA-Attn: the CI excludes 0, but Holm p = 0.94 after Holm over a 351-test family. Only the p is quoted.

For each comparison, report the difference, the CI, the Holm p and the direction (rank-biserial sign) together. State the Holm family size in Section 5, and explain once that mean and median can disagree because a few large-penalty episodes dominate the mean.

At N = 10 the median gap difference between SA-Attn and Attention+penalty is only −0.02 €, and SA-Attn is worse on 43 % of replays. The IQMs in T1 (0.712 vs 0.751) tell the same story. The SA-Attn advantage over its twin is a tail effect, which should be said.

**M4. The positioning of the learned policy at the training size (CR O3).**
- At N = 10, a training size, SA-Attn (1.290) has a **higher gap than every other method with the layer in T1/T2**. Every rule with the layer, eMPC with the layer, the zero-padded MLP with the layer, Deep Sets, the no-bound variant and the deploy-only policy are all better (Holm p ≤ 1.1e-9 for the rules).
- At N = 5 the same holds in mean.
- The training curves (Fig. S1) are flat after about 0.1 M of 3 M steps.
- SA-Attn had the worst validation J^pen of all tuned learned methods (−41.58 €, tuning `selected.yaml`).

Together these suggest under-training or a learning problem specific to the attention policy with the layer, not only "no advantage". "Simple rules are competitive" understates this. The paper should say plainly that, in this study, the attention policy did not beat a price-aware rule with the same layer at any station size. Only Deep Sets did, and only at N ≥ 20 (0.649-0.722 vs 0.774-0.793). At N ≤ 10 the price rule beat both learned set policies. It should also either diagnose the attention policy (for example the masked fallback steps or the β setting) or explicitly scope the claim to "under the stated tuning budget".

**M5. SAUCIS AI policy compliance (blocking for submission, author task).** Results and the supplementary results were reported, and the Discussion and Conclusion interpreted, by an AI drafting agent. The SAUCIS policy forbids AI to "produce, report, or interpret research findings". The TODOs ask the authors to rewrite these sections, write the practical implications and the AI statement, and fill in the author data. Until this is done the paper cannot be submitted. I count this as blocking, though it is outside the scientific assessment.

### 2.3 Minor issues

1. "the safety layer raised the penalised profit" (Abstract): a causal wording for a comparison of two differently trained policies. Use "SA-Attn scored … higher than its penalty-trained twin" (CR C2).
2. "per EV served": the denominator is admitted EVs, in the Abstract, Section 5, Results, the Fig. 5 axis and the Conclusion's "7.4-34.1 % of EVs served" (CR N2, N4).
3. Shortfall/deficit ratio 0.009: this is a configuration mean, while the episode mean is 0.019. Report the **maximum** ratio as the actual test of Lemma S5. The † on two-seed methods in T3 is wrong (CR N1).
4. "overload in 6.7 %, 7.9 %, 17.0 % of their episodes": these are configuration-weighted (episode-weighted: 7.9 / 9.2 / 17.0 %), and the sentence mixes 43-configuration counts with a 36-configuration share (CR N3).
5. "penalty-based and Lagrangian training reduced violations": no reference arm (CR C3).
6. "Deep Sets / price-threshold … lower gap at every size": not on the 25-charger literature configuration (CR C4).
7. "as Theorem 1(b) predicts" should be "consistent with", since the episode-level check is only necessary (CR C5).
8. Conformal: 0.799 < 0.80 is not quite "met". "Close to known departures" holds for the gap only, as the shortfall is 6-7× (CR C6).
9. "3 of 3,550 replays needed solver fallbacks": 41 more used the interior-point step (CR N5). Give the redraw counts of DD49(1) in s4.
10. s4 EV-GNN sentence: "run from its public code … but it was not run" is self-contradictory wording.
11. The Fig. 5 caption omits panel (b). In Fig. 3, colours are shared between methods (Deep Sets/ALAP, three orange attention variants). The Fig. 3 caption says "best rules", although ALAP/AFAP are shown.
12. Cover letter: "pre-declared primary endpoint" and "a hard-constrained optimum measures the cost of the guarantee" do not match the manuscript (CR C7).
13. Length: 23 single-spaced pages, about 40 double-spaced, against the guidance of ≤ 25. Table 2 (T1) has 42 rows; move the rows without the layer, except the penalty and Lagrangian twins, to the supplementary. Build with `saucis.cls`.
14. Markdown mirrors: garbled tables, truncated captions and "placeholder" notes. Regenerate them and the lint reports (current blocking flags: main 29, supplementary 50, almost all acronym false positives) (CR §5.4-5.5).
15. The `terminology.md` ban on "set policy" conflicts with the DD45 framing. Update the style file.

---

## 3. Is the reframed contribution publishable in SAUCIS?

**Yes, conditionally.** SAUCIS publishes applied ML and optimisation papers of moderate novelty. Its scope does not name energy, but it has published EV-related work.

The reframed paper offers three things:
- a correctly stated and empirically verified feasibility result for a taper-aware V2G safety layer;
- an honest, broad benchmark showing that the layer removes transformer overloads for any upstream controller;
- a candid negative result on learned set policies.

That is a publishable, useful contribution **if** the framing matches the data. At present it does not fully match:
- The title ("Size-Agnostic Safe Reinforcement Learning …") and the Abstract still put the learned policies and the departure guarantee at the centre.
- The data say the layer's transformer part is the robust contribution.
- The departure part helps rules but induces infeasible steps and *more* unserved EVs with learned policies.
- The attention policy never beat a price-threshold rule with the same layer. Deep Sets beat it only at N ≥ 20, by about 0.07-0.12 € per charger.

A referee will accept the honest negative findings. A referee will **not** accept the omission in M1 once it is found in the supplementary tables.

The novelty is modest, as the paper itself says: a classical projection, a laxity-type bound, and a set policy. Its value lies in the combination, the two-layer theorem and the measurement. With M1-M3 fixed, I expect SAUCIS referees to ask for a minor-to-major revision rather than to reject.

The authors should consider a title that names the layer as the contribution. Two examples, with the final choice left to the authors:
- "A Feasibility-Preserving Safety Layer for Size-Agnostic V2G Charging Control: Guarantees and an Empirical Study";
- "… Safety Layer for V2G Charging: Transformer Guarantees, Conditional Departure Guarantees and Learned Set Policies".

---

## 4. Recommendation

**Major revision** (internal recommendation: not ready for submission).

Scientifically, the paper is close. No numerical errors were found in about 140 audited numbers. But one abstract-level statement is factually contradicted by a table (M2). One key result is omitted in a way that reverses the practical reading of the departure guarantee (M1). The statistical reporting is selective (M3). The AI-policy rewrites are outstanding (M5).

After these fixes, and with the authors' own Results/Discussion text, I would expect a referee verdict of minor revision.

## 5. The five most valuable fixes

1. **Report the net effect of the departure bound on learned policies (M1).** In Results and the Discussion, state that:
   - the no-bound variant had a lower gap at every N and fewer unserved EVs (0.99 % vs 2.24 %);
   - SA-Attn left more EVs unserved than its penalty twin (2.24 % vs 1.86 %) and the Lagrangian (0.20 %);
   - rules with the layer leave almost none (0-0.44 %).

   Frame the conditional guarantee as useful with non-exploiting controllers, and as a source of infeasible steps for profit-seeking learners. Note the β/penalty confound of the no-bound variant.
2. **Fix the Abstract and Conclusion departure sentences (M2):** restrict them to "methods with the known-departure bound" and remove the "any episode run through the layer" conjunction. Replace "the safety layer raised" with a comparative statement. Replace "per EV served" with "per admitted EV".
3. **Unify the statistical reporting (M3):** for every comparison give the difference, the 95 % CI, the Holm p and the direction. State the Holm family size. Explain mean vs median, and mention the IQMs, which show that SA-Attn's edge over its twin is a tail effect.
4. **State the learning result plainly and scope it (M4):**
   - at N ≤ 10, every rule with the layer beat SA-Attn;
   - SA-Attn did not beat the price-threshold rule with the layer at any test-grid N, and Deep Sets did only at N ≥ 20;
   - the training curves plateau early.

   Either diagnose the attention policy or scope the claim to the tuning budget. Consider a layer-centred title.
5. **Make the paper submittable (M5 + housekeeping):**
   - the authors rewrite Results, Discussion and Conclusion and write the AI statement and author data;
   - fix the cover letter (C7);
   - build with `saucis.cls` and trim Table 2;
   - regenerate the Markdown mirrors and lint;
   - apply the minor wording fixes N1-N6 and C3-C6 of the consistency report.
