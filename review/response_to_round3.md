# Response to the round-3 audit (consistency_report_round3.md, internal_review_round3.md)

Writing Agent A5, 2026-10-24. Every new number comes from `analysis/tables/*.md` or `analysis/stats/tests_*.csv`, and each one is listed in `paper/results_number_sources.md`. Raw-episode recomputations from the consistency report are not used in the text, because no analysis table holds them. These are the episode mean 0.019, the episode-weighted overload shares, the 41 interior-point solves, the median differences and the share of replays.

## Consistency report

| Item | Change |
|---|---|
| C1 | **Abstract:** "For every non-conformal method with the known-departure bound, no EV was left short …". **Conclusion:** split into two sentences, with the same qualifier on the departure part. |
| C2 | Abstract: "the attention policy scored 0.04 to 3.46 euros per admitted EV more than the twin, with an interval including zero only at 200 chargers". |
| C3 | 07: "did not remove violations, with 3,730 and 3,274 overload steps". |
| C4 | 07 and 01: "at every station size of the test grid". In the Abstract and Conclusion, the price-threshold claim is also limited to the test grid, because of the tie on the literature configuration (TS3). |
| C5 | 06: "consistent with Theorem 1(b)". |
| C6 | 07 makes four points:<ul><li>coverage was within 0.002 of each target (the report's "0.001" does not hold for 0.992 and 0.902);</li><li>coverage fell below the target only at 0.80;</li><li>at α = 0.01, service collapsed;</li><li>at 0.05/0.1 the gap was close to known departures, but the shortfall was six to seven times as large (T7).</li></ul> |
| C7 | Cover letter: "pre-declared primary endpoint" deleted; "a penalty-trained twin measures the cost of the guarantee"; negative findings and O1/O2 added. |
| C8 | 05 and s4 now state:<ul><li>the Holm family (one test file, up to 351 comparisons);</li><li>the sign convention of the rank-biserial correlation;</li><li>median shift vs mean;</li><li>"inconclusive" where the test and the CI disagree.</li></ul>06 reports all three cases: (i) penalty at N = 200 with Holm 2.0e-5 and r = −0.35; (ii) ΔJ^pen at N = 200 with Holm 1.7e-6; (iii) E4 Deep Sets at N = 200 with interval [0.9, 3.6] points and Holm 0.94. It also reports the deploy-only policy at N = 200. One further disagreement was found in tests_T1 at N = 10: LLF, sLLF, AFAP, RR and ALAP with the layer have CIs above 0, but negative rank-biserial correlations (−0.39 to −0.47). These comparisons are reported as inconclusive. |
| C9 | "per learned method" (Abstract, Conclusion, cover letter). |
| N1 | 06: "as a mean over configurations of the episode means for episodes with a positive deficit". The maximum per-episode ratio is not in the tables, so it is not reported. The † in the T3 shortfall column needs a fix in `tables.py` (orchestrator). |
| N2 / N4 | "per admitted EV" in 00, 05, 06, the Fig. 5 caption, 07 and 08. **The Fig. 5 y-axis still says "per EV served"** (`figures.py` line 473), so it must be regenerated. |
| N3 | 06: "Averaged over the 36 test-grid configurations, 6.7 %, 7.9 % and 17.0 % of their episodes had overload". The counts over all configurations are in a separate sentence. |
| N5 | 07: "3 of 3,550 replays needed additional solver fallbacks, primal simplex or a tighter shortfall tolerance". |
| N6 | 07: "with intervals up to 0.05"; 08: "in the point estimates". |
| N7 | `results_number_sources.md`: new section "Supplementary s2", with 768, 1.4e-14, 6.8 %, 3.7 %, 0.21 and about 5 % (derived). |
| O1 | New Results subsection 6.4, "Effect of the departure bound". It reports:<ul><li>a lower gap for the no-bound variant at every N (0.117 to 1.266, Holm ≤ 1.3e-30);</li><li>0.99 % vs 2.24 % unserved and 0.09 vs 0.43 kWh (T3);</li><li>1,551 vs 6,158 unserved EVs (T3c);</li><li>stress 2.59 % vs 12.04 % (TS2);</li><li>p_i = l_i on 27.8-42.3 % of occupied charger-steps (T3b).</li></ul>The confound is stated: penalty reward vs profit reward. β = 0.01 is the same for both variants (s4), so β is not mentioned as a confound. O1 is also stated in the Abstract, the Introduction, the Discussion draft (new paragraph on the price of the conditional guarantee), the Conclusion and s5. |
| O2 | 6.4 compares the share of unserved EVs: SA-Attn 2.24 % and 6,158 EVs, against the penalty twin 1.86 % and 1,451, Lagrangian 0.20 %, deploy-only 0.46 % and price+layer 0.44 %. LLF, sLLF, AFAP and RR with the layer left none. The claim "the J^pen advantage comes from the overload penalty, not service" was **not** adopted: SA-Attn has less energy not served than its twin (0.43 vs 0.63 kWh per episode, T3), and the text now says so. |
| O3 | 6.1: at N = 10, every rule and eMPC-HiGHS with the layer had a lower mean gap. Price and eMPC are confirmed by the tests; the other rules are inconclusive (see C8). MLP+layer: Holm 0.014. At N = 20, the comparison is mixed: SA-Attn is lower than LLF, sLLF, AFAP and RR, and price and eMPC are lower than SA-Attn. |
| Price-threshold statement | Verified against T1, T2 and tests_T2. SA-Attn has a higher mean gap than price+layer at every test-grid N (0.107 to 1.335, Holm ≤ 3.1e-4, r > 0). Deep Sets is below price+layer only at N ≥ 20 (means only). No test of that pair exists, and the text says so. |
| §5.1 | Fig. 3 caption: "three rules with the safety layer". Fig. 5 caption: panel (b) described. The colour sharing in Fig. 3 is for the analysis side. |
| §5.4 | Lint regenerated, and `lint_final_*` updated. "very" removed. IQM is now defined in s4. S5.7 has an added sentence. |
| §5.5 | `tex2md.py` fixed:<ul><li>tables come from `analysis/tables/*.md` and are renumbered as Table N;</li><li>`%` is protected through the double `inline()` pass;</li><li>the placeholder note appears only if the figure is missing;</li><li>`\euro` is rendered;</li><li>the supplementary title regex is fixed;</li><li>separator rows use 4 hyphens, so the lint no longer reads them as em dashes.</li></ul> |

## Internal review

| Item | Change |
|---|---|
| M1 | See O1/O2. Discussion: the departure part is most useful with controllers that do not ride their lower bounds, and a profit-seeking learner can turn it into a source of infeasible steps. |
| M2 | See C1. |
| M3 | See C8. 6.1 adds the IQMs (0.712 vs 0.751; 0.719 vs 0.748) and the rank-biserial correlations (−0.24, −0.19). The SA-Attn advantage over its twin is described as coming mostly from a minority of episodes with large penalties. |
| M4 | 6.1, 6.2, 07 and 08 state the following:<ul><li>no SA-Attn win over price+layer at any test-grid size;</li><li>Deep Sets beat price+layer only from N = 20;</li><li>at N = 10, the rules had lower means.</li></ul>The claim is limited to "the stated tuning budget of 12 configurations per method". The training-curve plateau is not stated, because no table holds it. |
| M5 | Unchanged author TODOs: AI statement, Acknowledgments AI sentence, Discussion rewrite (kept under its TODO), and the Results/Conclusion check. |
| Minor 1-12 | Covered above. Minor 9: the redraw counts are in s4 and 07 (DD49). Minor 10: the EV-GNN sentence is reworded. |
| Minor 13 | Not done here: the `saucis.cls` build and the trimming of T1 are author/orchestrator tasks. The main text is about 8,190 prose words. |
| Minor 14 | Done (§5.5). |
| Minor 15 | The terminology.md "set policy" ban needs an update (orchestrator). |
| Title | Unchanged. A layer-centred title is left to the authors. |
