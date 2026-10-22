# Schema additions to code_spec.md Sec. 3 (A3, for the Orchestrator and the analysis agent A4)

code_spec.md is owned by the Orchestrator; this file lists everything the code writes **beyond** Sec. 3.
All Sec. 3.1 columns are written in the given order and with the given names; the columns below follow them.
The authoritative lists in code are in `v2gsafe/eval/metrics.py`: method files use `METHOD_COLUMNS`
(= Sec. 3.1 + extras **without** the five oracle-only columns), oracle files use `ORACLE_COLUMNS` (all columns).
Round 3 (DD31-DD39) changes are marked **[R3]**; renamed: `pen_profit_eur` -> `penalised_profit_eur`,
`pen_profit_dd2_eur` -> `penalised_profit_dd2_eur`, `oracle_pen_ceil_eur` -> `oracle_pen_profit_eur`.
Checked against the analysis loader (`analysis_repo/scripts/load.py`) on a copy of the smoke output.
Booleans are written as `true`/`false`; missing values as empty fields.

## 1. episodes.csv — additional columns (after the Sec. 3.1 columns, in this order)

| column | type | meaning | filled for |
|---|---|---|---|
| energy_capped_kwh | float | sum over EVs of (E_des − E_req^DD2)^+ (DD2 arrival cap) | all |
| energy_admission_reduced_kwh | float | sum of admission-control reductions (E_req^DD2 − E_req^adm). **[R3]** identical for every method on a replay (controller-independent pre-pass, DD31); 0 for `_noadm` rows | all |
| n_admission_not_restored | int | arrivals where admission could not restore LP feasibility (request := min(E_arr, E_req)) | all |
| n_evs_unserved_dd2 | int | EVs with E(d+1) < E_req^DD2 − 0.01 kWh, arrival-capped reference (M9; Sec. 3.1 `n_evs_unserved` / `energy_not_served_kwh` use the **post-admission** E_req, the same for all methods [R3]) | all |
| energy_not_served_dd2_kwh | float | sum (E_req^DD2 − E(d+1))^+ | all |
| n_evs_touched_infeasible | int | EVs with p_i < l_i at least once (S 7.5 item 3; Theorem 1(c) condition) | all |
| **n_evs_unserved_shared** [DD43] | int | EVs with E(d+1) < E_req^shared − 0.01 kWh, E_req^shared = the replay's shared admitted requirement (same reference in every pass incl. noadm and conformal) | all incl. oracle |
| **energy_not_served_shared_kwh** [DD43] | float | sum (E_req^shared − E(d+1))^+ | all incl. oracle |
| **n_evs_stay_infeasible** [DD43] | int | EVs whose stay contains ≥ 1 infeasible (non-A5) step (A5 evaluated with the layer's bounds) | all (diagnostic for raw methods) |
| **share_evs_stay_feasible** [DD43] | float | 1 − n_evs_stay_infeasible / n_evs (**primary endpoint E4**) | all |
| **agg_upper_active_steps** [DD43] | int | steps whose commanded sum Σp ≥ C_tilde − tol (aggregate upper limit binding; = lambda > 0 on projected steps; raw methods: at or beyond the limit) | all |
| **agg_lower_active_steps** [DD43] | int | steps with Σp ≤ −C_rev + tol (lambda < 0) | all |
| **lb_active_frac** [DD43] | float | share of occupied token-steps with commanded p_i = l_i > p_i^min (departure-aware l_i) | all |
| a5_slack_min_kw | float | min over steps of C_tilde − sum l | all |
| actuation_margin_max_kw | float | max over EV-steps of (reported − commanded) power; <= eps_q = 0.008 (A3) | all |
| energy_delivery_min_kwh | float | min over EV-steps of E(t+1) − E(t) − dt p; >= −1e-9 (A3) | all |
| n_nonfinite_actions, n_bisect_handover | int | guard counters (S 7.5 item 9) | all |
| max_kappa | float | largest taper exponent kappa of the episode (Lemma E2 condition) | all |
| penalty_return_eur | float | sum of r_pen (Alg. 5) with the run's own w; **[R3]** shortfall term vs the admitted request | all |
| **penalised_profit_eur** [R3] | float | J^pen = profit − 10·energy_not_served_kwh − 10·tr_violation_kwh (EUR; w = 10 EUR/kWh fixed, DD32) | all incl. oracle (oracle: from its hard plan) |
| **penalised_profit_dd2_eur** [R3] | float | same with energy_not_served_dd2_kwh | all incl. oracle |
| n_steps | int | steps in the episode | all |
| **alpha** | float | conformal miscoverage level | conformal only (empty otherwise) |
| **conformal_coverage** | float | fraction of the episode's EVs with actual departure d >= predicted bound d_lo | conformal only |
| **n_evs_departed_before_bound** | int | EVs with d < d_lo (uncovered) | conformal only |
| coverage_frac | float | = conformal_coverage (alias read by A4's scripts) | conformal only |
| mean_d_minus_dlo | float | mean of d − d_lo over the episode's EVs, steps | conformal only |
| shortfall_departed_before_bound_kwh | float | shortfall of the uncovered EVs | conformal only |
| **oracle_profit_ceil_eur** | float | J_LP,ceil (Prop. O1(b)); **main gap reference (DD26)**; [R3] with taper (O4') and admitted requests | oracle files only |
| **oracle_pen_profit_eur** [R3] | float | soft-penalty oracle J^pen_LP,ceil (DD32): upper bound of `penalised_profit_eur` and `penalised_profit_dd2_eur` of every non-conformal method on the replay | oracle files only |
| solve_time_s, solver_status, shortfall_stage_kwh | float, str, float | oracle solve time, HiGHS status of stage 2, stage-1 optimum z* (kWh) | oracle files only (Sec. 3.4) |
| admission, safety_layer | bool | wrapper settings of the evaluated method | all non-oracle |
| replay_seed | int | EV2Gym seed of the replay | all |
| replay_fingerprint | str | hash of sessions, prices, loads, PV of the replay; `MISMATCH:<hash>` if it differs from the manifest | all |
| train_N_set | str | training sizes of learned methods, e.g. `10+20` (`train_N` = largest, int) | learned only |
| **in_distribution** [R3] | bool | DD35: test configuration belongs to the method's training mixture (learned) / is EV-GNN's training configuration N = 20, rho_tr = 4; empty for heuristics, MPC, oracle | learned, EV-GNN |

Oracle rows (`results/raw/oracle__{test_cfg_id}/episodes.csv`, method `oracle`): `profit_eur` = **J_LP**,
`oracle_profit_ceil_eur` = **J_LP,ceil**, `solve_time_s`, `solver_status`, `shortfall_stage_kwh`, plus
and every other Sec. 3.1 column filled: `n_evs`, `n_evs_capped`, `energy_charged/discharged_kwh` (net LP power),
`n_evs_unserved`, `energy_not_served_kwh`, `frac_evs_unserved`, `user_satisfaction_mean` (from the LP plan's
nominal departure energies vs the DD2-capped E_req), `n_evs_admission_reduced` = 0, violation, infeasible,
fallback and `a5_deficit_kwh` columns 0, `inference_ms_per_step_mean/_p95` = total LP time / T;
**[R3]** requests = admitted requests of the shared admission table, so `n_evs_admission_reduced` /
`energy_admission_reduced_kwh` equal the methods' values; `admission` = true; `solver_status` may read
`Optimal(ipm)` (retry with HiGHS' interior-point solver);
`proj_dist_mean`, `train_N`, `oracle_profit_eur` empty. For the analysis join (DD26) please carry
both oracle profits per (test_cfg_id, replay_id): the main gap uses `oracle_profit_ceil_eur`, the supplementary
gap `profit_eur` of the oracle rows. Sec. 3.1's `oracle_profit_eur` stays empty in method rows (join in A4).

### exp_id of evaluation variants
Rows evaluated from one training run with a different evaluation setting carry the evaluated method in the first
exp_id field (still `{method}__{train_cfg}__s{seed}__{hash8}`), e.g. `attn_safe_noadm__train_attn_safe__s0__ab12cd34`,
`attn_safe_conf_a0.05__train_attn_safe_conformal__s0__ef56ab78`; the files live in the training run directory
(`<raw>/attn_safe__train_attn_safe__s0__ab12cd34/episodes.csv`). (exp_id, test_cfg_id, replay_id) is unique.
EV-GNN rows carry the EV-GNN seed in `seed` and in their exp_id.

## 2. steps logs (Sec. 3.2, only with `--log-steps`)

`steps.parquet` (CSV fallback `steps.csv` if pyarrow is missing; pyarrow is optional):
exp_id, **method**, **test_cfg_id**, replay_id, replay_seed, t, price_ch, price_dis, P_tr, P_inf, P_pv, C, C_rev,
C_tilde, sum_p (reported), sum_p_cmd (commanded), sum_l, sum_u, n_connected, fallback_code
(−1 none, 0 FB-0, 1 FB-1, 2 FB-2, 3 FB-1*), feasible.

`steps_chargers.parquet` (fallback `steps_chargers.csv`): exp_id, **method**, **test_cfg_id**, replay_id,
replay_seed, t, i, p (commanded), p_rep (reported), l, u, E, E_req, tau.

## 3. train_log.csv — additional columns (after the Sec. 3.3 columns)

Episode statistics (`episode_return_eur_mean/std`, `frac_unserved_mean`, `tr_violation_kwh_mean`,
`proj_dist_mean`, ...) are means over a running window of the last 2E = 32 completed training episodes, so every
row is filled even when an update completes no episode (5-min runs: T = 336 > 128 rollout steps). Updates before
the first completed episode are written to `train_log_warmup.csv` instead. `n_episodes` = episodes completed in
that update.

update, n_episodes, penalty_return_eur_mean, infeasible_frac, fallback_frac, epochs_run,
n_evs_admission_reduced_mean, n_admission_not_restored_mean, sps, update_time_s, rollout_time_s,
**[R3]** lambda_short, lambda_over, cost_short_ep, cost_over_ep (PPO-Lagrangian only: multipliers after the
update's dual step and the rollout's per-episode normalised costs; empty for other methods).

## 4. Method-name strings

Emitted in the `method` column (core set: 31 names incl. `oracle` and `attn_safe_trainN50`; EV-GNN adds 2; the extended set adds
ablations, w-sensitivity, eMPC-24h and Gurobi eMPC):

| group | names |
|---|---|
| learned, main | `attn_safe`, `attn_penalty`, `mlp_safe`, `mlp_penalty` |
| **[R3]** learned baselines (DD33) | `attn_proj_nolb` (layer without departure lower bound), `attn_lagrangian` (PPO-Lagrangian), `deepsets_safe` (DeepSets + layer) |
| **[R3]** evaluation variants (DD31, DD33) | `attn_safe_noadm`, `attn_penalty_noadm` (admission off), `attn_penalty_safedeploy` (penalty checkpoint + safety layer at deployment) |
| **[R3]** reference policy (DD36; core since DD42) | `attn_safe_trainN50` (evaluated at N in {50, 100, 200}) |
| SA-Attn-CP (trained as `attn_safe_conformal`, evaluated per alpha) | `attn_safe_conf_a0.01`, `attn_safe_conf_a0.05`, `attn_safe_conf_a0.1`, `attn_safe_conf_a0.2`, `attn_safe_conf_point` (uncalibrated, Q_hat = 0) |
| learned ablations | `attn_safe_beta0`, `attn_safe_curriculum`, `attn_safe_small`, `attn_penalty_dense`, `attn_penalty_w1`, `attn_penalty_w100` |
| 5-min sensitivity policy (DD28) | `attn_safe_dt5` |
| heuristics raw / + safety layer | `afap`, `alap`, `rr`, `llf`, `sllf` [R3], `price_llf` [R3], each also `<name>_safe` |
| MPC | `empc_highs`, `empc_highs_safe` (HiGHS port, all N); extended: `empc_highs_24h`, `empc_highs_24h_safe` [R3] and `empc`, `empc_safe` (EV2Gym eMPC, Gurobi; **[R3] only `literature_N25`**, DD34) |
| EV-GNN (optional, DD27) | `evgnn`, `evgnn_safe` (one model per profile; `seed` column = EV-GNN seed) |
| oracle | `oracle` |

Pattern rules: `<name>_noadm` = evaluated with `--admission off`; `_conf_a{alpha:g}` uses Python `%g`
formatting of alpha. The training config name is the second field of `exp_id`
(`{method}__{train_or_test_cfg}__s{seed}__{hash8}`); baseline exp_ids use the test grid file stem
(e.g. `llf_safe__test_grid__s0__1a2b3c4d`). Sensitivity configurations have ids such as
`public_N20_rho4_soc0.8`, `private_N20_rho4_dt5`, **[R3]** `workplace_N20_rho4_thr0.05` (throughput cost) and
`public_N20_rho4_dpf0.9` (discharge price factor); main-grid additions `{profile}_N20_rho3`, `{profile}_N200_rho3`
(DD36); smoke ids carry `_T16`/`_T48`.

Per-case fallback counts are the Sec. 3.1 columns `fb0_steps`, `fb1_steps`, `fb2_steps`, `fb1star_steps` (DD43(2)).

**DD43(1) requirement semantics.** Every pass receives the shared admitted requirement of the replay: the wrapper
writes it into EV2Gym's spawned EV object (`desired_capacity`), so EV-GNN and the Gurobi eMPC (whose session
targets are read from the profiles at construction and then restored) see it; the HiGHS eMPC uses the table
entries for arrivals inside its horizon. Conformal passes take the shared requirement re-certified with their
d_lo (DD43 supersedes DD42(2)); `_noadm` passes keep the DD2-capped request. `user_satisfaction_mean` is computed
with EV2Gym's formula against the original E_des. MLP methods are evaluated only at N ≤ 20 (DD43(3)): no MLP rows
for N ≥ 50 or `literature_N25`.

## 5. Fallback cases (M7)

`fallback_code` in steps logs and the `fb*_steps` columns map to the paper's cases as:
paper (i) = **FB-1*** (code 3, `fb1star_steps`), (ii) = **FB-2** (code 2, `fb2_steps`), (iii) = **FB-1** (code 1,
`fb1_steps`), (iv) = **FB-0** (code 0, `fb0_steps`); −1 = no fallback (A5 step). FB-0 also occurs on a
contradictory step (C < −C_rev) when the remaining budget C_tilde − sum p_min is negative.

## 6. Admission tables [R3] (DD31)

`results/admission/<test_cfg_id>__s<replay_seed>__<hash8>.json`: one entry per arrival {loc, t_arr, d, d_ctrl,
E_arr, E_des, E_req_dd2, E_req (admitted), reduced_kwh, not_restored} plus n_lp, lp_time_s; `<test_cfg_id>.done`
marks a completed configuration. Shared by all methods and the oracle; not read by the analysis. **DD43(1)** (supersedes DD42(2)): conformal variants
(`attn_safe_conf_a*`, `attn_safe_conf_point`) use the shared table, re-certified with their own d_lo (requirement
<= shared); they are evaluated against actual departures and reported separately. **DD42(4):** oracle
`profit_eur` (non-ceil J_LP) is supplementary only; main gap = `oracle_profit_ceil_eur`.
**DD42(6):** GNN-DT (`gnndt`, `gnndt_safe`) is not emitted; public code exists (DT4EVs, pinned 8b327d12,
`v2gsafe/baselines/gnndt_adapter.md`) but needs optimiser-generated offline data and a port (2-4 person-days).
