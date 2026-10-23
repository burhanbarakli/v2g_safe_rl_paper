# Consistency report, round 3 (A6, final round)

Reviewer: A6, independent consistency agent. I did not write the paper and made no edits to it. Date: 2026-10-24.

**Scope.**
- Main text: `paper/manuscript` (main.tex, sections/*.tex, main.pdf with 23 pages, main_anonymous.pdf with 22 pages) and the mirror `paper/manuscript.md`.
- Supplementary: `paper/supplementary` (sections s0-s5, supplementary.pdf) and the mirror `paper/supplementary.md`.
- Number sources: `paper/results_number_sources.md`.
- Analysis outputs: `analysis/tables/*`, `analysis/stats/tests_T*.csv`, `analysis/load_report.txt` and `STAT_METHODS.md`.
- Raw data: `code/results/raw/*/episodes.csv`, 156,650 rows (153,100 method episodes and 3,550 optimum episodes).
- Decisions and notes: `method/design_decisions.md` (DD45-DD49) and `RUN_NOTES_TR.md`.
- Other files checked: `paper/cover_letter.md`, the journal rules and the style linter.

**Location format.** `NN:` means `manuscript/sections/NN_*.tex`, and `sK:` means `supplementary/sections/sK_*.tex`. "md:L" is a line in `manuscript.md`.

---

## 0. Summary

| Check | Result |
|---|---|
| Measured numbers audited (Abstract, Intro, Setup, Results, Discussion, Conclusion, supplementary text) | about **140** numbers, all traced to a table cell, test file, load report or run notes |
| Recomputed directly from raw `episodes.csv` | about **110** values (≥ 25 required): gaps, transfer gaps, counts, E4, binding rates, T5 ratios, T7 coverage/shortfall/energy, T6 timings, TS1-TS3, Wilcoxon p-values |
| Value mismatches (number ≠ source) | **0** |
| Rounding errors | **0** |
| Description / unit / scope mismatches (number correct, wording or statistic misdescribed) | **7** (N1-N7, §1.3). The worst is N1, the shortfall-to-deficit-bound ratio 0.009, which is described as an episode average but is a mean over configurations; the episode mean is 0.019. N2 is the denominator "per EV served", which is really admitted EVs. |
| Overclaims / claims contradicted or not supported by the data | **9** (C1-C9, §2). The worst is C1: the Abstract and Conclusion state the departure check for "every method with the departure bound" or "any episode run through the safety layer", and T3c contradicts both (conformal: 81,742 / 4,916 / 5,491 / 8,126 / 5,469; no bound: 1,551). |
| Material omissions that change the reading of a result | **3** (O1-O3, §2.2): the departure bound's net effect on the learned policy, SA-Attn's unserved share against its penalty twin, and SA-Attn against the rules with the layer at N ≤ 10 |
| Scope / method consistency (2 seeds, statistics, DD46/DD47, not-run experiments, hardware, DD49) | Consistent. No "5 seeds" or "3 seeds" is left. One wording slip at s4 (EV-GNN "run from its public code … but it was not run"). |
| Theory vs data (Theorem 1(a)/(b), episode-level check) | Consistent for all non-conformal methods with the departure bound: 0 overload steps in 102,750 episodes, and the check column is 0 in 13 of 13 rows. The overgeneralisation in the Abstract and Conclusion is C1. |
| Cross-references / floats / captions | No undefined references in either log. 7 main floats (SAUCIS guidance ≤ 7). Two caption issues (§5). |
| Journal rules | Abstract 380 words (≤ 400). 5 keywords, alphabetical. The Article Information Form has all 8 items with author TODOs. The anonymised PDFs are clean. |
| Lint (current files) | main 29 blocking flags, all from the acronym list, 0 sentence-level issues. Supplementary 50 blocking flags: 45 acronym/V2G tokens and 5 section flags caused by long table captions in the mirror. Details in §5.4. |
| Markdown mirrors | **Defective**: generated tables are garbled, two figure captions are truncated at "95%", figure captions still say "placeholder", and the supplementary title is duplicated (§5.5) |

**Blocking for submission (from this report):** C1 (Abstract/Conclusion contradicted by T3c), O1-O2 (omissions that change the reading of the departure result), and the SAUCIS AI-policy TODOs in Results/Discussion/Conclusion (§5.3). Everything else is minor or moderate.

---

## 1. Number audit

### 1.1 Method

1. I extracted every measured number from 00, 01, 05, 06, 07, 08, s3, s4 and s5 (text, not tables) and from the figure captions.
2. I matched each number to its row in `results_number_sources.md` and then to the cited cell in `analysis/tables/*.md` or `stats/tests_T*.csv`.
3. I recomputed about 110 of them independently from the raw `episodes.csv` (script logic in Appendix A):
   - J^pen = profit − 10·(energy_not_served_shared_kwh + tr_violation_kwh).
   - gap = (oracle_pen_profit_eur − J^pen)/N, joined on (test_cfg_id, replay_id).
   - Seed-averaged policy per replay, then the mean over configurations of the replay means (STAT_METHODS.md).
   - The test grid is the 36 configurations without ρ = 3 kW and the literature configuration, unless the table says otherwise.

Global checks from raw:
- 153,100 method episodes, 30 methods, 43 configurations and 3,550 optimum episodes.
- 0 episodes with J^pen above the soft optimum + 1e-6 €, which matches `load_report.txt` and s3.
- The 102,750 episodes run through the safety layer have 0 overload steps and 0 overload kWh.

### 1.2 Numbers checked (✓ = matches the source cell; R = also recomputed from raw and equal at the printed precision)

**Abstract (00)**

| Text value | Source | Raw recompute | Status |
|---|---|---|---|
| 102,750 episodes through the layer, no overload | T3c sum of 15 rows | 102,750; Σ overload steps = 0 | ✓ R |
| 3,730 / 3,274 overload steps | T3c | 3,730 / 3,274 | ✓ R |
| check = 0 "for every method with the departure bound" | T3c | 0 for 13 non-conformal rows; >0 for 5 conformal rows | value ✓, **claim C1** |
| 0.90 / 0.87 / 0.68 / 0.79 € (N = 200) | T2 0.902 / 0.869 / 0.675 / 0.789 | 0.902 / 0.869 / 0.675 / 0.789 | ✓ R |
| 65.9 % → 92.6 % (E4) | T3b 65.86 / 92.58 | 65.86 / 92.58 | ✓ R |
| 0.04 to 3.46 € per EV "served" | T5 −0.042 / −3.462 | −0.042 / −3.462 | ✓ R; denominator wording **N2**; causal wording **C2** |
| interval includes zero only at N = 200 | T5 CI column | – | ✓ |
| coverage 0.992 / 0.950 / 0.902 / 0.799 | T7 | 0.992 / 0.950 / 0.902 / 0.799 | ✓ R |
| 10 and 20 chargers; up to 200 | design | – | ✓ |

**Results (06)**

| Text value | Source | Raw | Status |
|---|---|---|---|
| SA-Attn 1.290 / 0.973 | T1 | 1.290 / 0.973 | ✓ R |
| Attention+penalty 2.204 / 1.569 | T1 | 2.204 / 1.569 | ✓ R |
| Lagrangian 1.792 / 1.258 | T1 | 1.792 / 1.258 | ✓ R |
| −0.915 [−1.254, −0.601], −0.595 [−0.797, −0.412]; Holm 1.3e-5, 0.0014 | tests_T1 rows 0, 3 | raw Wilcoxon p 3.17e-7 / 3.93e-5 (= p_wilcoxon) | ✓ R |
| Deep Sets 0.984 / 0.722; diff 0.306 [0.218, 0.397], 0.251 [0.220, 0.286]; Holm < 1e-65 | T1, tests_T1 (3.55e-66, 3.11e-83) | 0.984 / 0.722; p 3.23e-68 / 2.78e-85 | ✓ R |
| no bound 0.776 (both) | T1 | 0.776 / 0.776 | ✓ R |
| price+layer 0.790 / 0.788 | T1 | 0.790 / 0.788 | ✓ R |
| LLF, sLLF, AFAP, RR + layer 1.095 / 1.099 | T1 | 1.095 ×4 / 1.099 ×4 | ✓ R |
| price-threshold 156.370 / 159.518 | T1 | same | ✓ R |
| eMPC-HiGHS 108.138 / 103.765 | T1 | same | ✓ R |
| "mostly from energy not served" (153.20 / 317.44; 109.46 / 210.21 kWh) | T1 ENS column | same ENS | ✓ R (10·ENS/N = 153.2 vs gap 156.4; 158.7 vs 159.5; 109.5 vs 108.1; 105.1 vs 103.8, so "mostly" holds) |
| eMPC-HiGHS+layer 1.145 / 0.846 | T1 | same | ✓ R |
| SA-Attn 0.921 / 0.881 / 0.902 | T2 | same | ✓ R |
| reference 0.884 / 0.857 / 0.869 | T2 | same | ✓ R |
| transfer gaps 0.037 [0.026, 0.050], 0.023 [0.012, 0.033], 0.033 [0.022, 0.046] | T2, tests_T2 rows 18-24 | 0.037 / 0.023 / 0.033 | ✓ R |
| Deep Sets 0.698 / 0.649 / 0.675; below the ref by 0.186-0.208 | T2 | same; −0.186 / −0.208 / −0.194 | ✓ R |
| price+layer 0.793 / 0.774 / 0.789 | T2 | same | ✓ R |
| N = 5: 2.292 / 1.663 / 0.956 | T2 | same | ✓ R |
| penalty N = 200 0.929; diff −0.027 [−0.060, 0.016] | T2, tests_T2 row 78 | 0.929; −0.027 | ✓ R (Holm p = 1.98e-5 not reported; see **C8**) |
| stress 2.979 / 1.586; DS 2.400 / 1.423 | TS2 | same | ✓ R |
| 102,750 episodes | T3c | 102,750 | ✓ R |
| 3,730 / 3,274 / 3,648 overload steps | T3c | same | ✓ R |
| 6.7 % / 7.9 % / 17.0 % of episodes with overload | T3 | config-weighted 6.7 / 7.9 / 17.0; **episode-weighted 7.9 / 9.2 / 17.0** | ✓ cell, wording **N3** |
| episode-level check 0 for non-conformal methods with bound | T3c | 0 in 13 rows | ✓ R |
| cases (i), (ii), (iv) did not occur | T3b | Σ fb1star = Σ fb2 = Σ fb0 = 0; fb1 = infeasible steps in every layer row | ✓ R |
| 17,678 / 39,771 / 6,158 / 233,016 | T3c | same | ✓ R |
| DS 12,122 / 27,949 / 5,038 | T3c | same | ✓ R |
| LLF+layer 1; price+layer 1,675 infeasible steps | T3c | same | ✓ R |
| E4 65.86 → 92.58; infeasible/episode 4.64 → 0.58 | T3b | same | ✓ R (note: T3b is test-grid only; T3d at N = 200 incl. stress gives 1,696/900 = 1.88 per episode, so the scope must stay "test grid") |
| DS E4 higher at every size, Holm ≤ 0.0043 to N = 100, 0.94 at N = 200 | tests_T3 rows 54-69 | raw p at N = 100: 3.45e-5; N = 200: 8.7e-3 | ✓ R (CI at N = 200 [−0.036, −0.009] excludes 0; see **C8**) |
| shortfall / deficit bound 0.009 | T3 | config-mean 0.0093; **episode mean 0.019 (seeds 0.018 / 0.020); ratio of sums 0.037** | ✓ cell, wording **N1** |
| ΔJ^pen −3.462 [−4.639, −2.388] / −0.042 [−0.094, 0.025] | T5 | −3.462 / −0.042 | ✓ R |
| Wilcoxon on J^pen Holm ≤ 2e-4 | tests_T5 (max 1.96e-4, N = 20) | raw p 7.0e-15 … 3.9e-5 | ✓ R |
| Δprofit 0.142 [0.075, 0.218] / 0.064 [0.039, 0.091] / −0.092 [−0.120, −0.065] | T5 | 0.142 / 0.064 / −0.092 (N = 50: −0.024, negative) | ✓ R |
| MLP ΔJ^pen −6.958 to −1.383 | T5 | −6.958 (N = 10) / −1.383 (N = 20); N = 5 −4.324 lies inside | ✓ R |
| coverage 0.992 … 0.799 | T7 | same | ✓ R |
| conformal gap 27.341 / 1.292 / 1.958; known 1.210 | T7 | same | ✓ R |
| charged energy 791.4 / 1396.5 kWh | T7 | same | ✓ R |
| deploy-only N = 200 0.930 | T2 | 0.930 | ✓ R |
| inference 7.76 / 5.99 ms | T6 | 7.76 / 5.99 (test-grid episodes; incl. stress 7.74 / 5.91) | ✓ R |
| literature 0.507 / 0.507 | TS3 | 0.507 / 0.507 | ✓ R |

**Experimental setup (05) and supplementary compute (s4)**

| Text value | Source | Raw / check | Status |
|---|---|---|---|
| 43 configurations; 100 / 50 replays; 3,550 per policy | RUN_NOTES §2 | 24×100 + 12×50 + 3×100 + 3×50 + 100 = 3,550; T3d episode counts agree | ✓ R |
| 153,100 episodes, 30 methods | RUN_NOTES, load_report | 153,100 / 30 | ✓ R |
| 12 configs × 7 methods = 84 tuning jobs; 18 training runs | RUN_NOTES | 9 trained variants × 2 seeds = 18 dirs | ✓ R |
| 0.5×10^6 / 3×10^6 steps | config.yaml | `total_env_steps: 3000000`; logs end at 2,998,272 | ✓ R |
| hardware, OS, Python 3.13.15, PyTorch 2.14.1 | RUN_NOTES §1 | – | ✓ |
| 1.6 / 7.9 / 10.2 / 2.5 h; 6 min; 298 + 43 jobs | RUN_NOTES §2 | Σ T6 training times / 2 parallel jobs ≈ 10.1 h | ✓ |
| 3 of 3,550 replays needed solver fallbacks | RUN_NOTES §3 | solver_status: Optimal 3,506; **Optimal(ipm) 41**; Optimal(primal) 2; Optimal(slack1e-8) 1 | ✓, wording **N5** |
| training time 1.15 h; 1.68 / 7.76 ms | T6 | 1.68 / 7.76 | ✓ R (ms) |

**Discussion (07) and Conclusion (08)**

| Text value | Source | Status |
|---|---|---|
| lower bound on 28 % to 42 % of occupied charger-steps | T3b 27.8 / 42.3 (raw 27.8 / 42.3) | ✓ R |
| within 0.04 € of the N = 50 reference | T2 0.037 / 0.023 / 0.033 | ✓ point estimates; the upper CI is 0.050, see **N6** |
| twin earned at most about 0.14 € more per EV | T5 0.142 | ✓ R |
| levels 0.05-0.1 close to known departures | T7 1.292 / 1.319 vs 1.210 | ✓ for the gap; but the shortfall is 2.75 / 3.09 vs 0.43 kWh (6-7×), see **C6** |
| 7.4 % to 34.1 % of EVs "served" had an infeasible step | derived 100 − 92.58, 100 − 65.86 | ✓ arithmetic; wording **N4** |
| 12 configurations per method | RUN_NOTES | ✓ |
| 3 of 3,550 | RUN_NOTES | ✓, see N5 |

**Supplementary results text (s5) and pipeline (s3)**

| Text value | Source | Raw | Status |
|---|---|---|---|
| 1,551 (no bound, check column) | T3c | 1,551 | ✓ R |
| N ≥ 50: 0.901 / 0.674 / 0.770 / 0.910 / 0.996 / 1.138 | T4 | same | ✓ R |
| total shortfall 771,853 / 12,507 / 2,568 kWh | T7 | same | ✓ R |
| point prediction 0.899 / 1.322 | T7 | same | ✓ R |
| relative J^pen gap 163.0 % / 102.5 % | TS1 | 163.0 / 102.5 | ✓ R |
| stress E4 55.30 / 65.91 vs 65.14 / 73.25 | TS2 | same | ✓ R |
| literature 0.363 / 0.456 / 0.507 | TS3 | same | ✓ R |
| 0 of 153,100 above the soft optimum | load_report | 0 | ✓ R |
| 768 combinations; 1.4×10^-14 kWh (s2) | `method/check_A1_results.txt` lines 1, 15 | – | ✓, **not listed** in results_number_sources (N7) |
| dropped spawns 6.8 % / 3.7 %, occupancy ≈ 0.21, energy within 5 % (s2, author TODO) | `method/profile_check_results.txt` CONFIRM lines (6.8, 3.7; 0.224 / 0.214 / 0.208) | – | ✓ for the first two; **not listed** in results_number_sources (N7) |

Derived numbers checked:
- 102,750 = Σ T3c episodes over the 15 layer rows.
- 7.4 / 34.1 and 28-42 %.
- 0.186-0.208.
- "mostly from energy not served".
- 3,550 per policy.

All are correct.

### 1.3 Description, unit and scope mismatches (numbers correct)

| # | Location | Problem | Evidence | Severity | Suggested fix |
|---|---|---|---|---|---|
| **N1** | 06 (md:529) "0.009 times the a-priori deficit bound on average over episodes with a positive deficit" | The cell is a mean over **configurations** of per-configuration means, so each N ≥ 100 configuration of 50 replays counts as much as one of 100, and configurations without deficit episodes drop out. The episode mean is **0.019** (per seed 0.018 / 0.020), and the ratio of sums is 0.037. The cell also carries a **†** (single deterministic run) for SA-Attn, Deep Sets and MLP+layer, although these have two seeds (T3 shortfall column). | raw recompute; T3 | Moderate | Either report "mean over configurations" or switch to the episode mean (0.019), and fix the † in `tables.py` for that column. Both readings support "far below the bound". |
| **N2** | 00 (abstract), 05 metrics, 06 §cost, Fig. 5 caption and y-axis, md:538 | "€ per EV **served**", but the denominator is **admitted** EVs, including unserved ones (STAT_METHODS; T5 caption "per admitted EV") | STAT_METHODS.md "Cost of the guarantee" | Minor-moderate (unit) | "per admitted EV" everywhere |
| **N3** | 06 (md:519) "overload in 6.7 %, 7.9 % and 17.0 % of their episodes" | The values are config-weighted means over the 36 test-grid configurations. The share of their episodes, pooled over both seeds, is 7.9 % / 9.2 % / 17.0 %. The same sentence gives overload-step counts over all 43 configurations (incl. stress and literature). | raw | Minor | "…in 6.7 %, 7.9 % and 17.0 % of episodes (mean over the test-grid configurations)" |
| **N4** | 08 "7.4 % to 34.1 % of the EVs **served** by the attention policy had one" | E4 is defined over admitted EVs, and the quantity is a mean of per-episode shares | T3b, STAT_METHODS | Minor | "of the admitted EVs" |
| **N5** | 07 limitations; s4 "3 of 3,550 replays needed solver fallbacks" | 41 further optimum episodes were solved by the interior-point step of the original chain (`Optimal(ipm)`). Only the 3 needed the two new fallbacks. | oracle `solver_status` | Minor | "3 of 3,550 replays needed additional solver fallbacks (primal simplex or a tighter shortfall tolerance)" |
| **N6** | 07, 08 "within 0.04 € per charger" | This holds for the point estimates (≤ 0.037), but the 95 % CI reaches 0.050 at N = 50 | T2 | Minor | "within 0.04 € (point estimates; CIs up to 0.05)" |
| **N7** | `results_number_sources.md` final section ("None of the numbers in the text lacks a source") | s2 numbers (768, 1.4e-14, 6.8 %, 3.7 %, 0.21, 5 %) are not listed. They are traceable to `method/check_A1_results.txt` and `profile_check_results.txt`. | – | Minor (documentation) | Add these rows |

Not in the paper, but inconsistent between documents: DD46 says "0 of 149,550 episodes exceed it", while `load_report.txt` and s3 give 153,100. The paper uses the correct 153,100.

---

## 2. Claim audit

### 2.1 Overclaims and statements contradicted or not supported

| # | Location | Statement | Data | Severity | Fix |
|---|---|---|---|---|---|
| **C1** | 00 Abstract ("For every method with the departure bound, no EV was left short in an episode without infeasible steps"); 08 Conclusion ("no transformer overload occurred in any episode run through the safety layer, **and** no EV was left short in an episode without infeasible steps") | Both are contradicted by T3c. The conformal variants run with a (conformal) departure bound and have 81,742 / 4,916 / 5,491 / 8,126 / 5,469 such EVs. The variant without the bound runs through the layer and has 1,551. Results (06) and s5 correctly say "every **non-conformal** method with the departure bound". | T3c | **Major (blocking)**: abstract-level statement contradicted by a table | Abstract: "For every method with the known-departure lower bound, …". Conclusion: "…and, for every method with the known-departure bound, no EV …". |
| **C2** | 00 "the safety layer **raised** the penalised profit by 0.04 to 3.46 euros per EV served" | This is a causal attribution to the layer, but the comparison is between two differently trained policies (SA-Attn vs the penalty-trained twin). The deploy-only arm isolates the layer at deployment, and it is not used here. Also, the low end "0.04" is the N = 200 point estimate, whose CI includes zero. | T5, T2 | Moderate | "SA-Attn scored 0.04 to 3.46 € per admitted EV higher in J^pen than its penalty-trained twin, with an interval including zero only at 200 chargers" |
| **C3** | 07 "penalty-based and Lagrangian training **reduced** violations but did not remove them" | There is no reference arm, such as an unconstrained attention policy, against which "reduced" is measured. The data show only that they did not remove violations. | T3c | Minor | "…did not remove violations (3,730 and 3,274 overload steps)" |
| **C4** | 07 "the Deep Sets policy had lower gaps … **at every size**"; "A price-threshold rule with the layer had a lower gap than the attention policy **at every size**" | Not on the 25-charger literature configuration: Deep Sets 0.516 vs SA-Attn 0.507, and the price-threshold rule 0.507 = 0.507 (TS3). True on the test grid (all Holm p ≤ 3e-4). | TS3, tests_T2 | Minor | "at every station size of the test grid" |
| **C5** | 06 "…no EV was left short in an episode without infeasible steps, **as Theorem 1(b) predicts**" | The episode-level check is a necessary-condition check. It cannot see an EV that was short in an episode that also had an infeasible step, but whose own stay was free of infeasible steps. 07 limitations state this. | 05, 07 | Minor | "consistent with Theorem 1(b)" |
| **C6** | 07 "coverage **met** its targets … levels of 0.05 to 0.1 **stayed close** to known departures" | 0.799 < 0.80 at α = 0.2, a marginal undercoverage. "Close" holds for the gap (1.292 / 1.319 vs 1.210), but not for the shortfall (2.75 / 3.09 vs 0.43 kWh per episode) or for unserved EVs in episodes without infeasible steps (4,916 / 5,491 vs 0). | T7, T3c | Minor | "achieved coverage within 0.001 of each target"; "close in gap, with a 6-7-fold shortfall" |
| **C7** | cover_letter.md | (a) "the share of vehicles whose stay contains no infeasible step is a **pre-declared primary endpoint**". The manuscript declares no endpoints, and its main metric is the gap (05 Metrics). (b) "a **hard-constrained optimum measures the cost of the guarantee**". In the paper the cost is penalty twin minus layer policy, and J^LP,ceil is only a normaliser (05, T5). | 05, T5 | Moderate (cover letter) | Delete "pre-declared primary endpoint"; replace (b) with "a penalty-trained twin measures the cost of the guarantee" |
| **C8** | 06 (statistics reported selectively) | Wilcoxon (rank-based) and the bootstrap CI of the mean disagree in three places, and the text reports only the convenient half each time:<br>(i) Attention+penalty vs SA-Attn at N = 200: CI [−0.060, 0.016] includes 0, but Holm p = 2.0e-5. Only the CI is reported, and the text says "approached".<br>(ii) T5 at N = 200: CI includes 0, but Holm p = 1.7e-6. Both are reported, without the explanation that they test different things (median shift vs mean; 63 % of replays favour SA-Attn while the mean is driven by a few large episodes).<br>(iii) E4 Deep Sets vs SA-Attn at N = 200: Holm p = 0.94, but the CI [−0.036, −0.009] excludes 0. Only the p is reported. | tests_T2 row 78, tests_T5 row 10, tests_T3 row 69; raw medians | Moderate | Report the difference + CI + Holm p + rank-biserial sign together for every comparison, and add one sentence in 05 that the Wilcoxon tests the median shift while the CI is of the mean |
| **C9** | 00 "With two training seeds per method" | Rules and eMPC run once (deterministic). | T1 † | Trivial | "per learned method" |

No statement claims or implies an attention advantage. The Intro, Abstract, Discussion and Conclusion all state that attention showed no advantage. The reframing therefore holds on this point (§2.3).

### 2.2 Omissions that change the reading of a reported result

| # | What the paper says | What the data also show (not stated anywhere in main or supplementary text) | Severity |
|---|---|---|---|
| **O1** | 06 "Attention+safety layer without departure bound reached 0.776 at both sizes". 07 "the variant without the departure bound left EVs unserved in episodes without any infeasible step". | The variant **without** the departure bound has a lower gap than SA-Attn at **every** N (diff 0.12-1.27, Holm p ≤ 1.3e-30, tests_T2). It also leaves **fewer** EVs unserved overall: 0.99 % vs 2.24 % of EVs and 0.09 vs 0.43 kWh per episode (T3); 1,551 vs 6,158 unserved EVs (T3c); 2.59 % vs 12.04 % at the stress level, N = 20 (TS2). The departure bound makes the learned policy ride its lower bounds (p_i = l_i on 28-42 % of steps vs 4-9 % without the bound, T3b). This causes infeasible steps and *more* unserved EVs. The text reports only the conditional check, in which the no-bound variant looks worse. Caveat: the no-bound variant was trained with β = 0.01 and a penalty reward (s4), so it is not a clean ablation. | **Major (blocking for an honest framing)** |
| **O2** | 06 "it left 6,158 of 233,016 admitted EVs unserved" | Its penalty twin **without** any layer left 1,451, the Lagrangian 1,013 and the deploy-only policy 516 (T3c). By share: SA-Attn 2.24 %, Attention+penalty 1.86 %, Lagrangian 0.20 %, deploy-only 0.46 %, price+layer 0.44 % (T3). In service terms, the "guaranteed" policy delivers worse than the unguaranteed twin. The J^pen advantage of SA-Attn comes from the overload penalty, not from service. | **Major** |
| **O3** | 01 "simple rules with the safety layer are competitive with both" | At the training size N = 10, **every** rule with the layer had a lower mean gap than SA-Attn (1.290): LLF, sLLF, AFAP and RR 1.095, ALAP 1.007, price-threshold 0.790, eMPC-HiGHS 1.145. All have Holm p ≤ 1.1e-9 (tests_T1). The MLP+layer did too (1.025, Holm p = 0.014). At N = 5, every rule with the layer also had a lower mean gap (raw: 0.956-1.107; eMPC-HiGHS+layer 2.183 vs 2.292). Only price, ALAP and AFAP are tested there, and AFAP has Holm p = 1.0. "Competitive" understates this at N ≤ 10. | Moderate |

### 2.3 Reframing (DD45) consistency

| Place | Consistent with DD45? | Note |
|---|---|---|
| Title (main.tex, supplementary.tex, cover letter, README) | Yes, identical working title everywhere | The supplementary mirror duplicates it ("Supplementary Material: Supplementary MaterialSize-Agnostic…"), a mirror bug |
| Abstract | Yes ("no advantage") | C1, C2 |
| Contributions (01) | Yes: two set policies, the layer as the core, "None of the ingredients is new in isolation" | 01 still calls attention "the main instantiation". Tables still label only SA-Attn "(ours)". Acceptable, but slightly at odds with "one of two set architectures compared". |
| Discussion / Conclusion | Yes | C1, C3-C6 |
| Cover letter | Mostly | C7 |
| `terminology.md` | Not updated | It still forbids "set policy", which the reframed paper now uses deliberately (6×). Update the style file, not the paper. |

---

## 3. Scope and method consistency

| Item | Status | Evidence |
|---|---|---|
| 2 seeds everywhere; no "5 seeds" / "3 seeds" / "ten seeds" / t-intervals left | ✓ | grep over both mirrors and the cover letter; s4 states that the N = 50 reference also has two seeds (RUN_NOTES §3.1: reduced from 3 to 2) |
| Statistics text = STAT_METHODS.md | ✓ | 05 §Tuning/statistics and s4 match: Wilcoxon on replay-level differences of seed-averaged policies, Holm per test file, stratified replay bootstrap, IQM at training sizes, no seed-level t (DD48). One gap: 05 does not say that Holm's family is the whole test file (up to 351 comparisons in T3), which explains p = 0.94 at N = 200 for E4. Add one clause. |
| DD46 (shared requirement, J^pen recomputed) | ✓ | 05 Metrics; s3; reproduced from raw |
| DD47 (absolute gap primary, relative only as ratio of sums in supplementary) | ✓ | 03 eq. gap, TS1, s5 |
| Not-run experiments not presented as run | ✓, with one wording slip | 05 and 07 list ablations, penalty-weight and other sensitivities, MPC variants, EV-GNN and the example day as not run. s5 says the same. The only sensitivity is ρ = 3 kW. **Slip:** s4 (supplementary.md:565) "EV-GNN … was planned as an external size-agnostic baseline, **run from its public code at commit 01aa1b7a** with its own training script, but it was not run". This reads as if it had been run; it should say "was to be run from…". The supplementary figure file is named `figS2_ablation` and the section label is `app:res-ablation`. Its content (architecture and constraint-handling variants) was run, so this is cosmetic. |
| Hardware / run times = RUN_NOTES | ✓ | All values match §1-§2 of RUN_NOTES. |
| DD49 limitations | ✓ | (1) Redrawn training seeds: 07. The counts (1 per tuning job, 2 per training job) are not given, so add them to s4. (2) Oracle tolerance sensitivity and 3 fallbacks: 07, see N5. (3) Not-run studies: 07. |
| Validation / tuning symmetry | ✓ text | Selected penalty weight w = 10 for both penalty methods (tuning/selected.yaml), the same as the evaluation weight. Fair. |

---

## 4. Theory vs results

| Statement in paper | Data | Verdict |
|---|---|---|
| Theorem 1(a): the transformer limit holds at every step for any raw actions under A3, A6 | 0 overload steps and 0.0 kWh in all 102,750 layer episodes (raw Σ tr_violation_steps = 0) | Exact |
| "the three cases that A6 excludes did not occur" (06) | Σ fb1star = fb2 = fb0 = 0 over all layer episodes; every fallback step is case (iii) (Σ fb1 = Σ infeasible steps in every row) | Exact |
| Theorem 1(b) episode-level check = 0 for methods with the departure bound | 0 for SA-Attn, Deep Sets, MLP+layer, ref N = 50, LLF, sLLF, price, AFAP, ALAP, RR, eMPC (+layer), deploy-only and no-admission (13 rows). Conformal rows > 0, as the theorem does not cover them (s5 explains why). No-bound 1,551. | Exact for 06 and s5. Overgeneralised in 00 and 08 (C1). |
| T3c caption "Theorem 1(b) covers the methods with the safety layer and the departure lower bound, not the conformal variants…" | consistent | ✓ |
| Lemma S5 deficit bound: shortfall ≤ bound | ratio 0.009 (config mean) / 0.019 (episode mean); max per-episode ratio not reported | Supported (N1). Reporting the **maximum** ratio (it must be ≤ 1) would be the actual test of the lemma. |
| Prop. S3 (soft optimum bounds every method) | 0 of 153,100 above +1e-6 € | Exact |

---

## 5. Usual checks

### 5.1 Cross-references, floats, captions

- `main.log` and `supplementary.log`: no undefined or multiply defined references and no `??` in either PDF. All `S-` references resolve via xr.
- Main floats: Table 1 (positioning), Table 2 (T1), Figures 1-5. That makes 7 floats, at the SAUCIS guidance limit. Each is referenced before it appears.
- In the main text, Table S8 (T3c) is cited before Table S6 (T3). That is fine for supplementary tables.
- Caption issues:
  - **Fig. 5** (fig6_cost_of_guarantee): the caption describes only panel (a). Panel (b), the violations of the penalty-based policies in kWh per episode, is not mentioned. The y-axis says "per EV served" (N2).
  - **Fig. 3**: Deep Sets and ALAP+layer share light blue, and SA-Attn, Attention+penalty and the no-bound variant all share orange. They are distinguished only by line style and marker, so grey-scale readability is weak. The caption says "the best rules with the safety layer", but ALAP and AFAP are shown, and they are not the best (T2 selects "the three baselines with the lowest mean gap"). That is only true pooled, so consider "three rules with the safety layer".
  - Table 2 (T1) is a 42-row table in the main text. Consider moving the rows without the layer (except penalty and Lagrangian) to the supplementary to stay within length guidance.

### 5.2 Journal rules

| Rule | Status |
|---|---|
| Abstract ≤ 400 words, no citations, no float or section references | 380 words ✓ |
| 3-5 keywords, alphabetical | 5, alphabetical ✓ |
| Article Information Form (8 items) | All present: Contributions, Acknowledgments, Conflict, Support, Ethical Approval, Availability, AI Statement, Plagiarism. Author TODOs in Contributions, Ack/AI sentence, Conflict, Support, Ethics wording, repository link, AI statement ✓ |
| Anonymised PDF | `main_anonymous.pdf` and `supplementary_anonymous.pdf` contain no names, e-mails or repository links ✓. Third-party commit hashes only. |
| Length | 23 pages single-spaced (about 40 double-spaced) against the guidance of ≤ 25 double-spaced. This is unchanged since round 2. Guidance, not a rule. |
| `saucis.cls` build | Not done (fallback article class) — author task |

### 5.3 SAUCIS AI policy (blocking for submission, author action)

- "AI tools must not be employed to produce, report, or interpret research findings." Results (06) and supplementary results (s5) were reported by an AI drafting agent. The Discussion (07) is an AI-drafted interpretation (marked "DRAFT"). The Conclusion bullets (08) are AI-drafted as well.
- TODOs at 06:9, s5:3, 07:5, 07 implications and 08 ask the authors to check and rewrite. Until they do, the paper cannot be submitted under this policy.
- The AI statement is still a TODO.

### 5.4 Lint (run now on the current mirrors with `style_thresholds.json`)

| File | Blocking flags | Composition |
|---|---|---|
| `manuscript.md` | **29** (= `paper/lint_latest_main.md`) | The acronym list only: 28 "used but never defined" tokens + "V2G before first definition" (triggered by the title). Mostly false positives: A1-A6 (assumption labels), S1-S16 (supplementary float labels), AMD/NVIDIA/RTX/GB/WSL2, SAUCIS, "DRAFT", "SA" (from SA-Attn), AI and E4 (defined in parentheses at first use). **0** sentence-level issues, every section OK. One restricted phrase: "very" (07 "a very conservative level", unquantified; replace with "α = 0.01"). |
| `supplementary.md` | **50** (`lint_latest_supplementary.md` says 51) | 44 acronym tokens (as above plus T2-T7, TS1-TS3, FB-0/1/2, DT4EV, CUDA, IQM) + V2G + 5 section flags. Four "long sentence" flags come from table captions rendered as sentences in the mirror, and one "mean" flag comes from the 3-sentence S5.7. IQM is never spelled out in the supplementary. |

The round-2 "final" lint files (`lint_final_*.md`: 14 / 15) are outdated. Regenerate `lint_final_*` from the current mirrors.

### 5.5 Markdown mirrors (`tools/tex2md.py` output)

These are defective. The LaTeX is correct, but a reviewer reading only the mirrors would be misled.
- Every generated result table (T1 in the main text; T2-TS3 in the supplementary) is garbled into a one-column fragment (`\begin{tabular}[b]{@{}r@{}}…`).
- The captions of Figures 4 and 5 are truncated at "with 95" (an unescaped `%`). supplementary.md:759 has the same problem.
- Figure captions still append "(result figure placeholder: …)" although the figures exist: 2 places in main, 5 in the supplementary.
- `\euro{}` is not rendered in the table captions.
- The supplementary title is duplicated.

---

## 6. Prioritised fix list

1. **C1**: qualify the departure-check sentences in the Abstract and Conclusion ("known-departure bound", non-conformal).
2. **O1 + O2**: in 06 add two sentences stating:
   - that the no-bound variant had a lower gap at every N and fewer unserved EVs (0.99 % vs 2.24 %);
   - that SA-Attn left more EVs unserved than its penalty twin (2.24 % vs 1.86 %) and the Lagrangian (0.20 %).

   Discuss this in 07 as the price of the conditional guarantee. Note the β/penalty confound of the no-bound variant.
3. **C8**: report difference, CI, Holm p and direction together. Explain median vs mean.
4. **C2, N2, N4**: replace "the safety layer raised" with a comparative statement of the two policies, and replace "per EV served" with "per admitted EV".
5. **N1**: describe 0.009 as a configuration mean or report the episode mean (0.019) and the maximum ratio. Fix the † on two-seed methods in T3.
6. **O3, C3-C6, N3, N5, N6**: wording fixes as listed.
7. **C7**: fix the cover letter (primary endpoint, cost-of-guarantee description).
8. Fix the `tex2md.py` table and `%` handling, regenerate both mirrors and lint, and update `results_number_sources.md` (N7) and `terminology.md` ("set policy").

---

## Appendix A. Raw recomputation (logic used)

```python
D = concat(all episodes.csv); D = D[smoke != 'true']
O = D[method=='oracle'][['test_cfg_id','replay_id','oracle_pen_profit_eur']]
M = D[method!='oracle'].drop(columns=['oracle_pen_profit_eur', ...]).merge(O, on=['test_cfg_id','replay_id'])
M.jpen = M.profit_eur - 10*(M.energy_not_served_shared_kwh + M.tr_violation_kwh)
M.gap  = (M.oracle_pen_profit_eur - M.jpen) / M.N
grid   = not ('rho3' in cfg or 'literature' in cfg)
SA     = M.groupby([method, cfg, replay]).mean()            # seed-averaged policy
cell   = SA[mask].groupby(cfg)[col].mean().mean()            # mean over configurations of replay means
counts = M.groupby(method)[...].sum()                        # T3c, both seeds summed
check  = sum(n_evs_unserved_shared where infeasible_steps == 0)
T5     = sum(pen.jpen - safe.jpen) / sum(n_evs)              # ratio of sums over matched replays
```

The scripts were run with `/home/claude/work/ext/venv/bin/python -I`. Scratch copies are `rc.py` and `rc2.py` in the A6 scratchpad and are not part of the repository.
