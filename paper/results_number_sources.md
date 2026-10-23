# Sources of the measured numbers in the paper

Every measured number in the manuscript and supplementary text, with its source. Sources are the analysis outputs in `analysis/` (git commit 93f9943 of the run repository):
- table cells, written as `table | group / row | column`, as in `tables/*.md` and `tables/provenance.json`;
- test files `stats/tests_T*.csv`;
- `load_report.txt`;
- the run notes `RUN_NOTES_TR.md` (hardware, run times).

All gaps are in € per charger and episode (DD47). Values are rounded as in the table cells. "Derived" marks a value computed from table cells, together with its computation.

## Abstract (00_abstract.tex)

| Text | Value | Source |
|---|---|---|
| test episodes run through the safety layer without overload | 102,750 | derived: sum of `T3c` Episodes over all rows with the safety layer (groups "with safety layer", "safety layer without departure bound", "conformal departure bound"); Overload steps = 0 in all these rows |
| overload steps, Attention+penalty / Attention+PPO-Lagrangian | 3,730 / 3,274 | `T3c \| without safety layer / Attention+penalty \| Overload steps`; `… / Attention+PPO-Lagrangian \| Overload steps` |
| no EV short in an episode without infeasible steps (every **non-conformal** method with the known-departure bound; round 3, C1) | 0 | `T3c \| with safety layer / * \| Check: unserved EVs in episodes w/o infeasible step` (all rows of the group) |
| gap of SA-Attn at N = 200 | 0.90 | `T2 \| N = 200 / SA-Attn (ours) \| Gap (€/charger)` = 0.902 |
| gap of the reference trained at N = 50, at N = 200 | 0.87 | `T2 \| N = 200 / SA-Attn trained at N = 50 (reference) \| Gap` = 0.869 |
| Deep Sets at N = 200 | 0.68 | `T2 \| N = 200 / Deep Sets+safety layer \| Gap` = 0.675 |
| Price-threshold+safety layer at N = 200 | 0.79 | `T2 \| N = 200 / Price-threshold+safety layer \| Gap` = 0.789 |
| E4 of SA-Attn at N = 5 / 200 | 65.9 % / 92.6 % | `T3b \| N = 5 / SA-Attn (ours) \| E4` = 65.86; `T3b \| N = 200 / …` = 92.58 |
| ΔJ^pen per admitted EV, range ("scored … more than its penalty-trained twin") | 0.04 to 3.46 € | `T5 \| Attention \| N = 200 \| Δ J^pen (€ per EV)` = −0.042 [−0.094, 0.025]; `… N = 5` = −3.462; sign reversed in the text (twin minus SA-Attn in T5); denominator = admitted EVs (T5 caption, STAT_METHODS) |
| interval includes zero only at N = 200 | – | `T5` CI column, attention rows N = 5…200 |
| price-threshold+layer lower mean gap than SA-Attn at every test-grid N | – | `T2 \| N = 5…200 / SA-Attn, Price-threshold+safety layer \| Gap`; `tests_T2.csv` attn_safe vs price_llf_safe (all diff > 0) |
| Deep Sets lower than price+layer only from N = 20 | 0.722 / 0.698 / 0.649 / 0.675 vs 0.788 / 0.793 / 0.774 / 0.789; N = 5, 10: 1.663 / 0.984 vs 0.956 / 0.790 | `T1` (N = 10, 20) and `T2` (N = 5, 50, 100, 200) Gap cells; no test of this pair exists |
| unserved EVs SA-Attn / penalty twin / no-bound variant | 2.24 % / 1.86 % / 0.99 % | `T3 \| … \| Unserved EVs (%)` |
| no-bound variant lower gap at every size | – | `tests_T2.csv` attn_safe vs attn_proj_nolb, gap_abs, N = 5…200: diff 1.266 … 0.117, all > 0 |
| E4 SA-Attn 65.9 % at 5 → 92.6 % at 200 | 65.86 / 92.58 | `T3b \| N = 5/200 / SA-Attn (ours) \| E4` |

## Results (06_results.tex)

| Text | Value | Source |
|---|---|---|
| SA-Attn gap, N = 10 / 20 | 1.290 / 0.973 | `T1 \| N = 10 / SA-Attn (ours) \| Gap`; `T1 \| N = 20 / …` |
| Attention+penalty gap, N = 10 / 20 | 2.204 / 1.569 | `T1 \| N = 10/20 / Attention+penalty \| Gap` |
| Attention+PPO-Lagrangian gap, N = 10 / 20 | 1.792 / 1.258 | `T1 \| N = 10/20 / Attention+PPO-Lagrangian \| Gap` |
| differences to Attention+penalty, CI, Holm p, rank-biserial | −0.915 [−1.254, −0.601], 1.3e-5, −0.24; −0.595 [−0.797, −0.412], 0.0014, −0.19 | `stats/tests_T1.csv`, rows attn_safe vs attn_penalty, metric gap_abs, N = 10 / 20: diff, b_lo, b_hi, p_wilcoxon_holm, rank_biserial |
| IQMs SA-Attn vs Attention+penalty, N = 10 / 20 | 0.712 vs 0.751; 0.719 vs 0.748 | `T1 \| … \| Optimality gap, IQM` |
| no-bound variant vs SA-Attn at N = 10 / 20 | 0.514 [0.372, 0.679]; 0.198 [0.145, 0.259]; Holm < 1e-45 | `tests_T1.csv` attn_safe vs attn_proj_nolb, gap_abs (Holm 2.1e-47, 3.8e-46) |
| N = 10: rules + layer below SA-Attn | price 0.790, ALAP 1.007, LLF/sLLF/AFAP/RR 1.095, eMPC 1.145 | `T1 \| N = 10 / … \| Gap` |
| N = 10: price / eMPC differences | 0.500 [0.341, 0.680], Holm 1.1e-9; 0.144 [0.072, 0.219], Holm 5.7e-59 | `tests_T1.csv` attn_safe vs price_llf_safe / empc_highs_safe, gap_abs, N = 10 |
| N = 10: other rules inconclusive, rank-biserial −0.39 to −0.47 | −0.388 (LLF, sLLF, AFAP, RR), −0.466 (ALAP); CIs exclude 0 | `tests_T1.csv` attn_safe vs llf_safe, sllf_safe, afap_safe, rr_safe, alap_safe, gap_abs, N = 10: b_lo > 0, rank_biserial < 0 |
| N = 10: MLP+layer | 1.025, Holm 0.014 | `T1`; `tests_T1.csv` attn_safe vs mlp_safe, N = 10 |
| N = 20: SA-Attn below LLF/sLLF/AFAP/RR + layer | 0.973 vs 1.099, Holm < 1e-24 | `tests_T1.csv` attn_safe vs llf_safe … rr_safe, N = 20 (diff −0.125, Holm 1.47e-25 / 1.50e-25) |
| N = 20: price / eMPC + layer lower | 0.788 / 0.846; 0.185 [0.137, 0.238] / 0.127 [0.098, 0.161]; Holm ≤ 9.8e-12 | `tests_T1.csv` attn_safe vs price_llf_safe (9.8e-12) / empc_highs_safe (2.8e-78), N = 20 |
| Deep Sets gap, N = 10 / 20 | 0.984 / 0.722 | `T1 \| … / Deep Sets+safety layer \| Gap` |
| differences to Deep Sets, Holm p | 0.306 [0.218, 0.397]; 0.251 [0.220, 0.286]; p < 1e-65 | `tests_T1.csv` attn_safe vs deepsets_safe, gap_abs, N = 10 / 20 (Holm 3.6e-66, 3.1e-83) |
| no departure bound, both sizes | 0.776 | `T1 \| … / Attention+safety layer without departure bound \| Gap` |
| Price-threshold+safety layer | 0.790 / 0.788 | `T1 \| … / Price-threshold+safety layer \| Gap` |
| LLF, sLLF, AFAP, Round Robin + layer | 1.095 / 1.099 | `T1 \| N = 10/20 / LLF+safety layer, sLLF+…, AFAP+…, Round Robin+… \| Gap` |
| raw price-threshold | 156.370 / 159.518 | `T1 \| … / Price-threshold \| Gap` |
| raw eMPC-HiGHS | 108.138 / 103.765 | `T1 \| … / eMPC-HiGHS \| Gap` |
| "mostly from energy not served" | 153.20 / 317.44 kWh, 109.46 / 210.21 kWh | `T1 \| … \| Energy not served` (× 10 €/kWh = most of J^pen) |
| eMPC-HiGHS + layer | 1.145 / 0.846 | `T1 \| … / eMPC-HiGHS+safety layer \| Gap` |
| SA-Attn gap N = 50 / 100 / 200 | 0.921 / 0.881 / 0.902 | `T2 \| N = 50/100/200 / SA-Attn (ours) \| Gap` |
| reference N = 50 / 100 / 200 | 0.884 / 0.857 / 0.869 | `T2 \| … / SA-Attn trained at N = 50 (reference) \| Gap` |
| transfer gaps | 0.037 [0.026, 0.050]; 0.023 [0.012, 0.033]; 0.033 [0.022, 0.046] | `T2 \| N = 50/100/200 / SA-Attn (ours) \| Transfer gap to N = 50 ref.` |
| Deep Sets N = 50 / 100 / 200 | 0.698 / 0.649 / 0.675 | `T2 \| … / Deep Sets+safety layer \| Gap` |
| Deep Sets below the reference by | 0.186 to 0.208 | `T2 \| N = 50/100/200 / Deep Sets+safety layer \| Transfer gap` = −0.186, −0.208, −0.194 |
| Price-threshold+layer N = 50 / 100 / 200 | 0.793 / 0.774 / 0.789 | `T2 \| … / Price-threshold+safety layer \| Gap` |
| N = 5: SA-Attn, Deep Sets, price+layer | 2.292 / 1.663 / 0.956 | `T2 \| N = 5 / … \| Gap` |
| SA-Attn vs price+layer, every test-grid N | diff 0.107 [0.065, 0.151] (N = 100) to 1.335 [0.919, 1.840] (N = 5); Holm ≤ 3.1e-4 | `tests_T2.csv` attn_safe vs price_llf_safe, gap_abs, N = 5…200 (max Holm 3.06e-4 at N = 100; all rank_biserial > 0) |
| Deep Sets vs price+layer by N | see Abstract rows | `T1`, `T2` Gap cells (not tested) |
| Attention+penalty at N = 200 and difference; Holm p; rank-biserial | 0.929; −0.027 [−0.060, 0.016]; 2.0e-5; −0.35 | `T2 \| N = 200 / Attention+penalty \| Gap`; `tests_T2.csv` attn_safe vs attn_penalty, gap_abs, N = 200 (p_wilcoxon_holm 1.98e-5, rank_biserial −0.346) |
| stress level SA-Attn N = 20 / 200 | 2.979 / 1.586 | `TS2 \| ρ_tr = 3 kW, N = 20/200 / SA-Attn (ours) \| Gap` |
| stress level Deep Sets | 2.400 / 1.423 | `TS2 \| … / Deep Sets+safety layer \| Gap` |
| stress level no-bound variant | 0.983 / 0.829 | `TS2 \| … / Attention+safety layer without departure bound \| Gap` |
| 102,750 episodes, no overload | see abstract | `T3c` |
| overload steps penalty / Lagrangian / MLP+penalty | 3,730 / 3,274 / 3,648 | `T3c \| without safety layer / … \| Overload steps` |
| episodes with overload (mean over the 36 test-grid configurations) | 6.7 % / 7.9 % / 17.0 % | `T3 \| without safety layer / … \| Episodes with overload (%)`; config-weighted per CR round 3 N3 |
| episode-level check = 0 | 0 | `T3c` check column, "with safety layer" rows |
| fallback cases (i), (ii), (iv) did not occur | 0 | `T3b` columns (i) FB-1*, (ii) FB-2, (iv) FB-0: 0 in every row |
| SA-Attn infeasible steps / EVs with one / unserved / admitted | 17,678 / 39,771 / 6,158 / 233,016 | `T3c \| with safety layer / SA-Attn (ours) \| …` |
| Deep Sets infeasible steps / affected / unserved | 12,122 / 27,949 / 5,038 | `T3c \| with safety layer / Deep Sets+safety layer \| …` |
| LLF+layer / price+layer infeasible steps | 1 / 1,675 | `T3c \| with safety layer / LLF+safety layer, Price-threshold+safety layer \| Infeasible steps` |
| E4 SA-Attn N = 5 → 200 | 65.86 % → 92.58 % | `T3b \| N = 5/200 / SA-Attn (ours) \| E4` |
| infeasible steps per episode N = 5 → 200 | 4.64 → 0.58 | `T3b \| N = 5/200 / SA-Attn (ours) \| Infeasible steps` |
| E4 Deep Sets vs SA-Attn, Holm p; N = 200 interval | ≤ 0.0043 up to N = 100; 0.94 at N = 200 with interval [0.9, 3.6] points | `tests_T3.csv` attn_safe vs deepsets_safe, share_evs_stay_feasible, p_wilcoxon_holm per N (N = 100: 0.0043; N = 200: 0.94); N = 200 b_lo/b_hi = −0.0355/−0.0090 (sign reversed, ×100) |
| shortfall / deficit bound (mean over configurations of episode means) | 0.009 | `T3 \| with safety layer / SA-Attn (ours) \| Shortfall / deficit bound`; described as a configuration mean (CR round 3 N1) |
| no-bound variant vs SA-Attn gap, every N | 0.117 [0.094, 0.143] (N = 100) to 1.266 [0.927, 1.652] (N = 5); Holm ≤ 1.3e-30 | `tests_T2.csv` attn_safe vs attn_proj_nolb, gap_abs (max Holm 1.33e-30 at N = 100) |
| unserved share / energy not served: no-bound vs SA-Attn | 0.99 % vs 2.24 %; 0.09 vs 0.43 kWh | `T3 \| … \| Unserved EVs (%)`, `Energy not served (kWh)` |
| unserved EVs summed: no-bound / SA-Attn / penalty twin | 1,551 / 6,158 / 1,451 | `T3c \| … \| Unserved EVs` |
| stress N = 20 unserved: no-bound / SA-Attn | 2.59 % / 12.04 % | `TS2 \| ρ_tr = 3 kW, N = 20 / … \| Unserved EVs (%)` |
| penalty twin unserved / energy not served | 1.86 %; 0.63 kWh | `T3 \| without safety layer / Attention+penalty \| …` |
| Lagrangian / deploy-only / price+layer unserved | 0.20 % / 0.46 % / 0.44 % | `T3 \| … \| Unserved EVs (%)` |
| LLF, sLLF, AFAP, RR + layer: no unserved EV | 0 | `T3c \| with safety layer / … \| Unserved EVs` |
| p_i = l_i share SA-Attn | 27.8 % to 42.3 % | `T3b \| N = 200 / N = 5 / SA-Attn (ours) \| p_i = l_i` |
| J^pen at N = 200: Holm p of the median shift | 1.7e-6 | `tests_T5.csv` row attn_safe vs attn_penalty, jpen_eur, N = 200 (p_wilcoxon_holm 1.69e-6) |
| ΔJ^pen attention N = 5 / 200 | −3.462 [−4.639, −2.388] / −0.042 [−0.094, 0.025] | `T5 \| Attention \| N = 5/200 \| Δ J^pen (€ per EV)` |
| Wilcoxon on J^pen, Holm | ≤ 2e-4 at every N | `tests_T5.csv` attn_safe vs attn_penalty, jpen_eur, p_wilcoxon_holm (max 2.0e-4 at N = 20) |
| Δ profit per admitted EV, N = 5 / 20 / 200 | 0.142 [0.075, 0.218] / 0.064 [0.039, 0.091] / −0.092 [−0.120, −0.065] | `T5 \| Attention \| N \| Δ profit (€ per EV)` |
| MLP ΔJ^pen range | −6.958 to −1.383 | `T5 \| MLP \| N = 10 / 20 \| Δ J^pen (€ per EV)` |
| conformal coverage | 0.992, 0.950, 0.902, 0.799 | `T7 \| … \| Coverage (episode mean)` |
| conformal gap α = 0.01 / 0.05 / 0.2, known | 27.341 / 1.292 / 1.958 / 1.210 | `T7 \| … \| Gap (€/charger)` |
| charged energy α = 0.01 vs known | 791.4 / 1396.5 kWh | `T7 \| … \| Energy charged (kWh)` |
| safety layer at deployment, N = 200; difference, Holm p | 0.930; −0.028 [−0.061, 0.014]; 9.9e-6 | `T2 \| N = 200 / Attention+penalty, safety layer at deployment \| Gap`; `tests_T2.csv` attn_safe vs attn_penalty_safedeploy, N = 200 |
| inference SA-Attn / Deep Sets, N = 200 | 7.76 / 5.99 ms | `T6 \| SA-Attn (ours) / Deep Sets+safety layer \| N = 200 (ms/step)` |
| literature configuration SA-Attn / price+layer | 0.507 / 0.507 | `TS3 \| SA-Attn (ours) / Price-threshold+safety layer \| Gap` |

## Experimental setup (05) and supplementary compute (s4)

| Text | Value | Source |
|---|---|---|
| 43 configurations, 3,550 episodes per policy | 43 / 3,550 | `RUN_NOTES_TR.md` §2; `findings_numbers.md` header (3550 oracle episodes) |
| hardware and software | AMD Ryzen 7 7700 (8 cores, 16 threads; WSL2 12 threads, 31 GB), RTX 5070 Ti 16 GB, Windows 11 Pro, WSL2 Ubuntu 24.04, Python 3.13.15, PyTorch 2.14.1 + CUDA 13.0 | `RUN_NOTES_TR.md` §1 |
| run times | preparation 1.6 h, tuning 7.9 h, training 10.2 h, evaluation 2.5 h, analysis 6 min | `RUN_NOTES_TR.md` §2 |
| 84 tuning training + 84 validation jobs; 18 training runs; 298 evaluation + 43 optimum jobs | – | `RUN_NOTES_TR.md` §2 |
| 153,100 episodes, 30 methods | – | `RUN_NOTES_TR.md` §2; `findings_numbers.md` header |
| 3 of 3,550 replays needed additional solver fallbacks | 3 | `RUN_NOTES_TR.md` §3 item 3; DD49 |
| training seeds redrawn once per tuning job, twice per training job (07, s4) | 1 / 2 | `method/design_decisions.md` DD49 (1) |
| Holm family up to 351 comparisons (05, s4) | 351 | number of rows of `stats/tests_T3.csv` |

## Discussion (07, draft) and Conclusion (08)

| Text | Value | Source |
|---|---|---|
| within 0.04 € per charger of the N = 50 reference (N = 50…200), intervals up to 0.05 | ≤ 0.037; upper CI 0.050 | `T2` transfer gap column (0.037 [0.026, 0.050], 0.023, 0.033) |
| coverage within 0.002 of each target, below it only at 0.80 | 0.992, 0.950, 0.902, 0.799 | `T7` coverage column |
| shortfall six to seven times as large at α = 0.05 / 0.1 | 2.75 / 3.09 vs 0.43 kWh | `T7 \| … \| Shortfall vs shared requirement (kWh)` |
| lower bound commanded on 28 % to 42 % of occupied charger-steps | 27.8 % to 42.3 % | `T3b \| N = 200 / N = 5 / SA-Attn (ours) \| p_i = l_i (% token-steps)` |
| twin earned at most about 0.14 € more per admitted EV | 0.142 | `T5 \| Attention \| N = 5 \| Δ profit (€ per EV)` |
| conformal levels 0.05–0.1 close to known departures | 1.292, 1.319 vs 1.210 | `T7` gap column |
| 7.4 % to 34.1 % of the admitted EVs with an infeasible step (08) | 100 − 92.58; 100 − 65.86 | derived from `T3b` E4 of SA-Attn at N = 200 / N = 5 |

## Supplementary results (s5_results.tex) and pipeline (s3)

| Text | Value | Source |
|---|---|---|
| variant without departure bound: unserved EVs in episodes without infeasible steps | 1,551 | `T3c \| safety layer without departure bound / … \| Check` |
| gaps at N ≥ 50: SA-Attn, Deep Sets, no bound, deploy-only, penalty, Lagrangian | 0.901, 0.674, 0.770, 0.910, 0.996, 1.138 | `T4 \| … \| Gap (€/charger) N ≥ 50` |
| total shortfall α = 0.01 / 0.05 / known | 771,853 / 12,507 / 2,568 kWh | `T7 \| … \| Total shortfall (kWh, all episodes)` |
| point prediction coverage / gap | 0.899 / 1.322 | `T7 \| SA-Attn-point (uncalibrated) \| …` |
| SA-Attn training time; inference N = 10 / 200 | 1.15 h; 1.68 / 7.76 ms | `T6 \| SA-Attn (ours) \| …` |
| relative J^pen gap > 100 % at N = 5, 10 | 163.0 %, 102.5 % | `TS1 \| N = 5/10 / SA-Attn (ours) \| J^pen gap vs soft optimum (%)` |
| stress E4 SA-Attn / Deep Sets, N = 20 / 200 | 55.30 / 65.91; 65.14 / 73.25 % | `TS2 \| … \| E4 (%)` |
| literature configuration: no bound / eMPC+layer / SA-Attn | 0.363 / 0.456 / 0.507 | `TS3 \| … \| Gap` |
| no episode above the soft optimum | 0 of 153,100 | `load_report.txt` line 3 |
| over all episodes no-bound variant 1,551 vs SA-Attn 6,158 unserved | 1,551 / 6,158 | `T3c \| … \| Unserved EVs` |
| inference N = 200: Deep Sets, rules + layer (max), eMPC + layer | 5.99 / 2.72 / 14.13 ms | `T6 \| … \| N = 200 (ms/step)` (max over rules with the layer: Round Robin+safety layer 2.72) |

## Supplementary s2 (scenario checks)

| Text | Value | Source |
|---|---|---|
| closed form vs EV2Gym charging routine | 1.4e-14 kWh | `method/check_A1_results.txt` line 15 (max abs error 1.421e-14 kWh) |
| combinations checked | 768 | `method/check_A1_results.txt` line 1 |
| dropped spawn attempts, public / private | 6.8 % / 3.7 % | `method/profile_check_results.txt` CONFIRM lines 23 / 25 (days = both) |
| mean occupancy about 0.21 | 0.208 / 0.224 / 0.214 | `method/profile_check_results.txt` CONFIRM lines 21 / 23 / 25 |
| energy demand per charger within about 5 % | 9.5 / 10.5 / 9.9 kWh per charger and day | derived: EVs/charger/day × mean request in CONFIRM lines 21 / 23 / 25 (0.65 × 14.6, 0.66 × 15.9, 0.42 × 23.5); max deviation from their mean ≈ 5 % |

## Numbers that could not be sourced from the analysis outputs

None of the numbers in the text lacks a source after round 3 (the s2 scenario checks were added). Raw-episode recomputations from the round-3 consistency report (episode-mean deficit ratio 0.019, episode-weighted overload shares, 41 interior-point optimum solves, median differences) are **not** used in the text, because they are not in the analysis tables. The G4 report gives slightly different values (e.g. SA-Attn gaps 1.29 / 1.64 / 1.13, coverage 0.951, total shortfall about 966,000 kWh). These were **not** used; the tables were used instead.
