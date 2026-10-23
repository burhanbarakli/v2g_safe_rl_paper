# G4 Report — Full results check (Phase 4)

Date: 2026-10-23 · Orchestrator → human author · Status: **waiting for decision**
Data: GitHub `burhanbarakli/v2g_safe_rl_paper` at commit 93f9943. 153,100 episodes, 30 methods, 43 test configurations, 3,550 oracle episodes, 18 training runs, 2 seeds per learned method. EV2Gym commit verified.

## 1. Integrity checks (done by the Orchestrator on the raw `episodes.csv` files)
- **Code changes in the run.** Three changes were made: the training-seed redraw for EV2Gym's negative-load crash (`vec_env.py`), the oracle solver fallbacks (`oracle.py`) and the seed cap (`pipeline.py`). They were reviewed. None of them changes the method, the test replays or the metrics.
- **Soft-penalty oracle as an upper bound.** After the fix in §3.1, no episode of any method exceeds the soft-penalty optimum (0 of 149,550 episodes). Before the fix, conformal episodes did exceed it.
- **Theorem 1′ (a), transformer limit.** No violation occurred in any of the 102,750 episodes run through the safety layer. The penalty-based attention policy had 3,730 violation steps, PPO-Lagrangian 3,274 and the MLP with penalty 3,648.
- **Theorem 1′ (c), departure requirement.** Every non-conformal method with a departure bound left energy unserved only in episodes that contained at least one infeasible step. This is consistent with Corollary 1. The check is at episode level, because the logs carry no per-EV flag.

## 2. Main findings (numbers only)
| | SA-Attn | Deep Sets + layer | Attn + layer, no departure bound | Price threshold + layer | LLF + layer | Attn + penalty |
|---|---|---|---|---|---|---|
| Gap to soft optimum, € per charger and episode, N = 10 / 20 / 200 | 1.29 / 1.64 / 1.13 | 0.98 / 1.28 / 0.92 | 0.78 / 0.84 / 0.79 | 0.79 / 0.88 / 0.78 | 1.10 / 1.05 / 1.07 | 2.20 / 2.11 / 0.96 |
| Infeasible steps (all episodes) | 17,678 | 12,122 | 0 (no departure bound) | 1,675 | 1 | – |
| Unserved EVs (all episodes) | 6,158 of 233,016 (2.6 %) | 5,038 | 1,551 | 368 | 0 | 1,451 |
| Transformer violation steps | 0 | 0 | 0 | 0 | 0 | 3,730 |

- **Zero-shot transfer.** SA-Attn trained at N ∈ {10, 20} has a gap of 0.92 / 0.88 / 1.13 at N = 50 / 100 / 200. The reference trained at N = 50 has 0.88 / 0.86 / 1.08 at the same sizes.
- **Conformal variant.** Coverage is 0.992, 0.951, 0.903 and 0.799 for α = 0.01, 0.05, 0.10 and 0.20, so the targets 0.99, 0.95, 0.90 and 0.80 are met. At α = 0.01 the re-certified requests are cut so strongly that the shortfall against the shared requirement reaches about 966,000 kWh in total.
- **Percentage gap.** The optimum's J^pen is small and often negative: the median is −1.03 € per charger. The percentage gap therefore exceeds 100 % and is not interpretable. The gap is now reported in € per charger.

## 3. Problems found and fixes that need no new runs
1. **Analysis bug.** For the conformal and no-admission passes, J^pen was computed with each pass's own shortfall, not with the shortfall against the shared requirement. The fix is to recompute J^pen from `energy_not_served_shared_kwh`.
2. **Primary gap metric.** The primary gap becomes an absolute value in € per charger and episode. The percentage is kept only as a ratio of sums in the supplementary material.
3. **Statistics with 2 seeds.** A paired t-test across seeds has one degree of freedom, so it is not informative. The primary analysis becomes replay-level paired tests on seed-averaged policies, using a Wilcoxon test with Holm correction plus a bootstrap stratified by configuration. Per-seed values are also reported. The low seed count is stated as a limitation.
4. **Text to update.** Seed counts, the missing extended set, EV-GNN not run, the example-day figure not produced, hardware and run times, the two new limitations (EV2Gym's negative-load seeds, sensitivity of the oracle tolerance), and the analysis-test update.

## 4. Decision needed
The results support three claims: the safety layer prevents transformer violations at every station size, the departure guarantee holds conditionally, and the policy transfers across sizes with no loss against an in-distribution reference. The results do **not** show an advantage of the attention policy over a Deep Sets policy or over a price-threshold heuristic that uses the same safety layer. Learned policies also trigger many aggregate-infeasible steps (A5 failures).

- **Option A — report as is.**
  - Reframe the paper around the safety layer and size-agnostic safe control rather than attention.
  - The title, contributions and abstract change.
  - No new runs.
- **Option B — method extension before writing.**
  - Add a capacity-aware (lookahead) lower bound to reduce A5 failures.
  - Re-train SA-Attn and Deep Sets (2 seeds each) with tuned hyperparameters, then re-evaluate.
  - Estimated cost: about 6-8 GPU-hours plus 3 h of evaluation on the author's machine.
  - Needs a method addendum and a new proof check.
