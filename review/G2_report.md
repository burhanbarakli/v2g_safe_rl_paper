# G2 Report — Method specification (Phase 2)

Date: 2026-10-22 · Orchestrator → human author · Status: **waiting for approval**

## 1. What was done
- **EV2Gym study.** Pinned at commit `55ef368c` (v2.0.0 code, 2026-09-15). The source was read and probed with small runs (`method/ev2gym_notes.md`).
- **Design decisions.** DD1-DD24 are fixed in `method/design_decisions.md`.
- **Method specification** (about 20,000 words):
  - `method/formulation.md`: system model, perfect-information LP, MDP, attention network.
  - `method/safety_layer.md`: bounds, projection, Theorem 1 with proofs, fallback, admission control, conformal extension, gradient handling.
  - `method/algorithm.md`: Alg. 1-7, hyperparameters, baselines, scenarios, compute estimate.
  - `method/nomenclature.md`: every symbol with its unit.
- **Independent proof check.** A separate agent re-derived the proofs and ran about 113,000 closed-loop steps (`review/proof_check.md`). It found no error in the proofs. It found one major numerical issue in the projection algorithm (wrong output when the aggregate constraint is tight to rounding), which is now fixed and re-tested. The minor issues are also fixed.
- **References.** 24 new verified references (conformal prediction, PPO/GAE, LayerNorm, HiGHS, statistics). The total is now 109 verified references, 67 of them from 2022 or later.
- **Style.** The style corpus is complete: 15 papers, 91,511 words. The final profile is in `style/style_profile.md`.
- **Figure 1.** Architecture and control-loop block diagram in `analysis/figures/fig_architecture.svg`, with PDF and PNG versions.

## 2. Key results of Phase 2
1. **Theorem 1 (feasibility invariance).** Under A1-A5, the transformer limit holds at every step for any raw action, during both training and deployment, and every EV reaches its required energy at departure.
   - **A1** (monotone max-charging map) was proven for EV2Gym's taper.
   - **A2** requires known departure time and energy.
   - **A3** covers the exact delivery model, with a small margin ε_q for EV2Gym's action-grid rounding.
   - **A4** (feasible request at arrival) is enforced by the arrival cap.
   - **A5** is per-step aggregate feasibility. It cannot be guaranteed by local bounds (a counterexample is given). Its failures are handled by the fallback rule (least laxity first, transformer limit kept) and are counted.
2. **Projection.** p_i = clip(x_i − λ, l_i, u_i), computed in O(N log N). It matches a QP solver to 3e-10 on random instances, and its Jacobian matches finite differences to 1.4e-8. The paper states openly that the form equals sLLF and the box-plus-sum projection.
3. **Conformal extension (Proposition 2).** A split-conformal lower bound on departure time gives P(requirement met) ≥ 1 − α in expectation within an episode, under exchangeability and A5.
4. **EV2Gym issues we work around without patching EV2Gym:**
   - the built-in optimal solution uses a soft departure penalty with a units bug, needs Gurobi and crashes on numpy 2; we use our own HiGHS LP oracle (lexicographic: first minimum shortfall, then maximum profit), whose taper relaxation makes the gap conservative;
   - current-dependent efficiency tables; we use our own constant-efficiency EV spec file;
   - action-grid rounding; we add a margin ε_q;
   - the default 05:00 start drops overnight sessions; we use per-profile start hours.
5. **Experiment design.**
   - Train on 10 and 20 chargers. Zero-shot test on 5, 10, 20, 50, 100 and 200 chargers.
   - 2 transformer tightness levels (4 and 7 kW per charger) and 3 arrival profiles (workplace, public, private), plus the 25-charger EV2Gym literature configuration.
   - 5 seeds per learned method.
   - Baselines: MLP+penalty, MLP+safety layer, Attention+penalty, AFAP, ALAP, Round Robin, LLF, eMPC, the LP optimum, and EV-GNN or GNN-DT if its code runs.
   - Estimated compute (not measured): about 35 GPU-hours or 150-200 CPU-hours for training, plus 50-70 CPU-hours for evaluation.

## 3. Choices that need your confirmation (defaults are already used)
| # | Choice | Default |
|---|---|---|
| Q1 | Accounting convention | EV2Gym native: no round-trip loss, efficiency only reduces power. Stated as a limitation |
| Q2 | Departure target | EV2Gym default (full battery), with an arrival-time feasibility cap; sensitivity run at 80 % SoC |
| Q3 | Inflexible load and PV scale | 0.4 × transformer rating |
| Q4 | Start hours | workplace 05:00, public and private 12:00; spawn multipliers 5 / 4 / 2.5 to equalise occupancy |
| Q5 | Penalty weights for the penalty-based baselines | 10 €/kWh |
| Q6 | Main algorithm | PPO; the projection is treated as part of the environment |

## 4. Next steps after approval
1. **Code infrastructure (A3):** `code/` with the safety layer, attention policy, EV2Gym wrapper, LP oracle, baselines, train/evaluate scripts, YAML configs, unit tests, `reproduce.sh` and a README for running on your computer. Only smoke tests run here; no experiment results are produced here.
2. **Analysis infrastructure (A4):** scripts that turn your raw result files into tables and figures, with statistics. These are tested on synthetic data only, which is clearly labelled and never reported.
3. **Writing (A5):** Method, Problem Formulation, Related Work, Introduction and Abstract in full, in the LaTeX template. Results and Discussion as skeletons with placeholders. No number is written before your experiments produce it.
4. **Further figures:** block diagram of the safety layer, a geometric sketch of the projection, and the overall training and evaluation pipeline.

**Decision needed:** approve the method (and Q1-Q6), or tell me what to change.
