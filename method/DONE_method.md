# DONE — Method and Theory (Agent A2)

Version 4, 2026-10-22: implementation feedback (DD25-DD29, `code_repo/code/IMPLEMENTATION_NOTES.md`). Version 3: arrival profiles. Version 2: revision after the internal proof check. Version 1: first release.

## Produced files (all in `/home/claude/work/method/`)

| File | Content |
|---|---|
| `nomenclature.md` | Every symbol with meaning, unit and the place where it is defined; condition C0 and assumptions A1-A5 |
| `formulation.md` | System model, transformer limit, perfect-information LP (Proposition O1), MDP (12 + 38 features), network spec, Lemma N1 (equivariance, padding invariance, $N$-independent parameter count) |
| `safety_layer.md` | Core theory: reach operator, Lemmas 0/E1/E2/E3, certified thresholds, bounds, Proposition 1 (projection) with floating-point-robust algorithms, Lemma 1, Theorem 1, Corollaries 1-2, A5 counterexample, fallback FB-0/1/2/1* (Lemma 2), Lemma 3, admission control, Proposition 2 (conformal), gradient handling, assumption table |
| `algorithm.md` | Alg. 1 safety layer (bounds, robust batched projection, fallback, actuation); Alg. 2 PPO with variable $N$; Alg. 3 deployment; Alg. 4 conformal calibration; Alg. 5 baselines and penalty reward; Alg. 6 hyperparameters, tuning, statistics; Alg. 7 scenario scaling across $N$ |
| `design_decisions.md` | DD15-DD24 appended (decisions confirmed by the Orchestrator) |
| `profile_check.py` → `profile_check_results.txt` | Arrival-profile statistics per profile, start hour and spawn multiplier (Alg. 7.1) |
| `projection_ref.py` | NumPy reference: `phi_ev2gym`, `thresholds`, `cap_request`, `bounds`, `is_feasible`, `sanitize`, `project` (robust), `_bisect_safe`, `project_bisect`, `jacobian`, `fallback`, `actuation` |
| `check_A1.py` → `check_A1_results.txt` | A1/A1′/A3 on EV2Gym's `EV` class (768 configurations) |
| `verify_projection.py` → `verify_projection_results.txt` | T1 (vs SLSQP, KKT, bisection), **T1b (review M1: tight A5, $|x|$ up to $10^8$, hyperplane, NaN)**, T2 Jacobian, T3 Moreau gradient, T4 closed loop with EV2Gym `EV` objects (5138 EVs) |

## Change log, version 4 (2026-10-22, implementation feedback DD25-DD29)

- **safety_layer.md S 7.4 (DD25):**
  - inside the admission LP, connected EVs use $E_i^{\mathrm{tgt}}=\min\{E_i^{\mathrm{req}},\Psi_i^{\mathrm{LP}}-10^{-6}\}$; admitted requests are never reduced;
  - explanation why the faithful LP is always infeasible with full-battery targets (the linear minorant never reaches $E^{\mathrm{cap}}$);
  - one-LP $y_{\max}$ implementation of the bisection (I-5);
  - explicit list of what admission control guarantees and what it does not;
  - Corollary 2 item 2 states the constant efficiency values (I-17).
- **formulation.md:**
  - the main optimality gap is now w.r.t. $J^{\mathrm{LP,ceil}}$ (DD26), with the $J^{\mathrm{LP}}$ gap in the supplement;
  - added why $J^{\mathrm{LP}}$ is not an upper bound inside EV2Gym (unbilled ceil gain; smoke example −2.376 vs −2.386 €);
  - the full-battery cap remark is corrected: the cap is inactive in floating point for long stays (I-3);
  - forecast details (I-12); head widths $d_m\to d_m/2$ and small-variant breakdown (I-21).
- **algorithm.md:**
  - Alg. 3 uses DD25 targets;
  - Alg. 4: predictor details, spec-class one-hot, the $-10^{-9}$ in the bound, measured calibration cost 10-15 CPU-h with caching and workers (I-7, I-22);
  - Alg. 5:
    - penalty-method rules (admission off, untightened limits, token mask; I-13);
    - eMPC adapter, and eMPC-HiGHS as a non-identical port (I-6);
    - EV-GNN per DD27 (per-profile, $N=20$, $\rho=4$, 5 seeds, not tuned); GNN-DT not included;
  - Alg. 6: tuning for methods 1-4 only, inheritance rules, validation profile cycling (I-18), head width;
  - Alg. 7:
    - loads/PV redrawn per episode via `tr_seed` (I-2);
    - EV2Gym forecast keys 100/10 for eMPC-HiGHS;
    - the literature configuration uses our spec file (I-11);
    - sensitivity scope (DD28);
    - compute section rewritten with the implementation's per-update timings (indicative) and run counts: 51 own PPO runs, plus EV-GNN 15 and dt5 5, both optional.
- **nomenclature.md:** $\Psi_i^{\mathrm{LP}}$ and $E_i^{\mathrm{tgt}}$ added; $J^{\mathrm{LP,ceil}}$ marked as the main gap reference.
- **Check of IMPLEMENTATION_NOTES' "faithful implementation choices"** (I-2, I-5, I-6, I-7, I-9, I-11 to I-19): all agree with or refine the docs. The docs were updated where they were silent or imprecise. I found no item where the code is wrong.

## Change log, version 3 (2026-10-22, Orchestrator request: arrival profiles)

- `algorithm.md` Alg. 7 rewritten:
  - three EV2Gym arrival profiles (workplace, public, private), with per-profile start hour and spawn multiplier equalising occupancy;
  - no profile flag in the observation (justified in Alg. 7.1);
  - horizon handling: EV2Gym drops non-fitting sessions instead of truncating them; about 60 % of private sessions were dropped with a 05:00 start, 3.7 % with a 12:00 start;
  - test grid: 36 configurations plus the literature configuration, 3100 test episodes per policy;
  - compute estimate: 56 training runs, about $2.1\times10^8$ env steps including tuning, about 35 GPU-h or 150-200 CPU-h for training, 50-70 CPU-h for evaluation.
- Alg. 2, 4 and 6 updated accordingly: profile mixing in training, 148 conformal calibrations, tuning on mixed profiles, 50 replays for $N\ge100$.
- New `profile_check.py` → `profile_check_results.txt`.
- DD24 added to `design_decisions.md`; formulation 4.2 notes that there is no profile feature.

## Change log, version 2

**Major issue M1 (projection algorithm with A5 tight to rounding).**
1. `project()` now:
   - overwrites the last breakpoint value with $\sum l$;
   - uses $\lambda$ strictly beyond the last breakpoint, i.e. $p=l$, when no breakpoint satisfies $\mathcal S_j\le\vartheta$;
   - checks the aggregate post-condition with tolerance $\epsilon_{\mathrm{num}}=10^{-9}$ and hands any failing row to `_bisect_safe()`.
2. `_bisect_safe()` keeps the feasible bracket end, with padded bracket ends so that $p=l$ (resp. $u$) holds exactly even for $|x|\sim10^8$.
3. `project_bisect()` accepts its polish only if it passes the post-condition.
4. The same logic is in safety_layer.md S 4.2 and in the PyTorch pseudocode of algorithm.md Alg. 1.2.
5. New test T1b: 5 families × 20000 rows. Maximum aggregate violation $\le1.0\times10^{-9}$ kW, zero box violations, agreement with bisection to $1.4\times10^{-15}$ (relative). Before the fix, 8742/20000 tight rows returned $p=u$, with up to 418 kW overshoot.

**Minor fixes.**
- **m1.1** NaN/Inf guard: `sanitize()` before the projection, an assertion in `actuation()`, and logged counters. Theorem 1 is stated for finite $x$.
- **m1.2** Fallback ordering key (laxity ascending, $l-p^{\min}$ descending, index ascending) is identical in code (`np.lexsort`) and text.
- **m1.3** The feasibility flag includes $\tilde C\ge-C_{\mathrm{rev}}$ (`is_feasible()`, Alg. 1.1, A5). The contradictory case is handled as FB-1*.
- **m2.1** Condition C0 ($P^{\mathrm{ev}}\le P^{\mathrm{cs}}$) is stated for (S7) and Lemmas E1 and E2, and added to Corollary 2.
- **m2.2** "$1/(\eta^{\mathrm c}\Delta t)$" wording.
- **m2.3** A1/A1′ roles in the DD2 cap reworded; A1′ wording made precise.
- **m2.4** Admission control re-certifies A4 through `cap_request`.
- **New remark:** certification does not replace A1 (reviewer's STEEP test).
- **m3.1** Corollary 1 proof corrected: fallback outputs satisfy $p^{\min}\le p\le u$.
- **m3.2** Corollary 1 strengthened: (b) holds at every step; the transformer limit can fail only on FB-0/FB-2/FB-1* steps.
- **m3.3** Corollary 2 configuration list completed:
  - C0;
  - efficiency constant over current levels via our spec JSON (otherwise EV2Gym draws a random scalar per EV); the layer reads $\zeta$ per EV;
  - 3 phases, zero dead-bands, `v2g_enabled: True`, $m=5$, actuation map, `empty_ports_at_end_of_simulation`.
- **m3.4** Single tolerance $\epsilon_{\mathrm{num}}=10^{-9}$ in code and text (new S 6.4, `bounds()`, Alg. 1.1).
- **m4.1-m4.3** Lemma 2(b) restricted to EVs feasible at $t$; Lemma 2(c) speaks of the commanded violation, up to $n\varepsilon_q$ realised; the policy-independent sufficient condition gains $\tilde C\ge-C_{\mathrm{rev}}$ and the induction argument.
- **m5.1** "A5 almost surely" in Proposition 2.
- **m5.2** Calibration per test configuration including $N$ (DD23, Alg. 4).
- **m5.3** The guarantee is the expected within-episode fraction; the episode-averaged coverage is reported.
- **m5.4** $\hat B(k)\le\max\{\hat B(k-1),E^{\min}\}$ added to (S10) and used in the proof.
- **m6.1-m6.3** "Unbiased for the score-function gradient (PPO surrogate at $\theta_{\mathrm{old}}$)"; mask threshold $\epsilon_{\mathrm{num}}$ and its negligible bias stated; fixed normaliser $B_{\mathrm{mb}}$ in Alg. 2.
- **m8.1-m8.6** Notation:
  - Nomenclature cross-references fixed and missing symbols added.
  - $\Delta E^{\mathrm{def}}$ used in Lemma 3.
  - Clashes removed: SoC $s\to\varsigma$, $s(\lambda)\to\mathcal S(\lambda)$, target $\theta\to\vartheta$, interpolation $\theta\to\upsilon$, breakpoints $\beta_j\to\lambda_{(j)}$, Kronecker $\delta_{ij}\to\mathbb 1[i=j]$, Lemma E2's $w\to\Omega$ and $x\to\hat\delta$, permutation $\sigma\to\varpi$.
  - Renamed results and sections: fallback cases F0/F1/F2 → FB-0/FB-1/FB-2, Lemma F1 → Lemma N1, Proposition F1 → Proposition O1, algorithm sections A1-A6 → Alg. 1-7.
  - Equations renumbered (S1)-(S16) in order of appearance.
  - Laxity in Alg. 1.1 includes $\eta^{\mathrm c}$.

**Citations added** (keys from `lit/parts/L4_verified.bib`):
- conformal: vovk2022algorithmic, papadopoulos2002inductive, lei2018distributionfree, angelopoulos2023conformal, romano2019conformalized, lindemann2023safe;
- architecture: xiong2020layer, ba2016layer, plus vaswani2017attention and lee2019set from references.bib;
- learning and tooling: schulman2016highdimensional, kingma2015adam, huang2022cleanrl, paszke2019pytorch;
- solver: huangfu2018parallelizing;
- statistics: agarwal2021deep, henderson2018deep, wilcoxon1945individual.

**Decisions recorded** as DD15-DD23 in `design_decisions.md`:

| DD | Decision |
|---|---|
| DD15 | Reward ÷ N |
| DD16 | $n\varepsilon_q$ margin + upward rounding |
| DD17 | $m=5$ |
| DD18 | Threshold tables |
| DD19 | Lexicographic oracle + ceil variant |
| DD20 | Admission control details |
| DD21 | Masking |
| DD22 | Penalty weights $w_E=w_P=10$ €/kWh |
| DD23 | Conformal calibration per configuration |

**Scenario scaling (former open issue)** is defined in `algorithm.md`, Alg. 7:
- $P_{\mathrm{tr}}=\rho_{\mathrm{tr}}N$ with $\rho_{\mathrm{tr}}\in\{4,7\}$ kW per charger (tight, moderate);
- inflexible load and PV multipliers of 0.4 (EV2Gym rescales both by $P_{\mathrm{tr}}$; the PV peak stays below $P_{\mathrm{tr}}$, so FB-2 cannot occur);
- workplace scenario with `spawn_multiplier` 5 (arrivals are per charger, hence $N$-invariant);
- 12 test configurations plus the original 25-charger/100 kW case.

Recorded as DD24 (version 3), with the three arrival profiles.

## Key numerical results (current)

- **A1/A1′:** hold in all 768 configurations, and analytically under C0 for every $\Delta t$.
- **A3:** exact for $m=5$ ($\kappa\le2.39<\kappa^{*}(5)=4.965$); fails for $m=1$, by up to 0.106 kWh.
- **Projection:**
  - SLSQP $3.4\times10^{-10}$ kW, KKT $1.4\times10^{-12}$;
  - T1b aggregate violation $\le10^{-9}$ kW;
  - Jacobian vs finite differences $2.1\times10^{-8}$;
  - Moreau gradient $3\times10^{-7}$.
- **Closed loop T4** (5138 EV2Gym EVs): 0 misses among EVs without an infeasible step, 0 transformer violations on feasible steps, 0 Lemma 3 violations. The one violating step overall is an unavoidable FB-2 step in a toy setting with large PV.

## Open issues

1. **A5 is closed-loop** and cannot be guaranteed by per-EV bounds. There is a counterexample at constant capacity, and the policy-independent sufficient condition is very conservative. Its failure frequency on the Alg. 7 scenarios is unknown until the experiments run (risk R8).
2. **Full-battery targets with the taper.** The DD2 cap is inactive in floating point for long stays and binds, often negligibly, for short ones. Caps are reported above 0.01 kWh only. The admission LP needs the DD25 targets.
3. **L4 bib keys** are in `lit/parts/L4_verified.bib` but not yet merged into `lit/merged/references.bib`. The merge is needed before A5 writes the manuscript.
4. **Missing venv packages:** `torch`, `highspy`, `scikit-learn`.
5. **EV2Gym MPC** runs only up to $N\le25$ under the restricted Gurobi licence. A HiGHS port or an academic licence is needed beyond that.
6. **EV-GNN/GNN-DT compatibility** with our commit and scenarios is unchecked.
7. **General-$\eta$ model** is covered by the theory; the reviewer's closed loop exercised lossy $\eta$ (0 violations), our own scripts did not.
8. **Projection norm.** The projection uses the Euclidean norm in kW; a weighted norm was not explored.
9. **Alg. 7 load/PV multipliers (0.4), forecast noise (10 %) and per-profile start hours** were proposed by me and adopted in the implementation configs. They should be confirmed by whoever owns the experiment design.
10. **Remaining dropped spawns** (public 6.8 %, private 3.7 %) bias those samples slightly towards shorter stays. Reported as a limitation.
11. **Compute figures** for the PPO update and the HiGHS MPC are estimates. Only EV2Gym step times were measured.
