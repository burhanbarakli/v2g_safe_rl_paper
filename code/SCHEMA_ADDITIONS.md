# Schema additions to code_spec.md Sec. 3 (A3, for the Orchestrator and the analysis agent A4)

code_spec.md is owned by the Orchestrator; this file lists everything the code writes **beyond** Sec. 3.
All Sec. 3.1 columns are written in the given order and with the given names; the columns below follow them.
The authoritative lists in code are in `v2gsafe/eval/metrics.py`: method files use `METHOD_COLUMNS`
(= Sec. 3.1 + extras **without** the four oracle-only columns), oracle files use `ORACLE_COLUMNS` (all columns).
Checked against the analysis loader (`analysis_repo/scripts/load.py`) on a copy of the smoke output.
Booleans are written as `true`/`false`; missing values as empty fields.

## 1. episodes.csv — additional columns (after the Sec. 3.1 columns, in this order)

| column | type | meaning | filled for |
|---|---|---|---|
| energy_capped_kwh | float | sum over EVs of (E_des − E_req^DD2)^+ (DD2 arrival cap) | all |
| energy_admission_reduced_kwh | float | sum of admission-control reductions | all |
| n_admission_not_restored | int | arrivals where admission could not restore LP feasibility (request := min(E_arr, E_req)) | all |
| n_evs_unserved_dd2 | int | EVs with E(d+1) < E_req^DD2 − 0.01 kWh (Sec. 3.1 `n_evs_unserved` uses the admitted E_req) | all |
| energy_not_served_dd2_kwh | float | sum (E_req^DD2 − E(d+1))^+ | all |
| n_evs_touched_infeasible | int | EVs with p_i < l_i at least once (S 7.5 item 3) | all |
| a5_slack_min_kw | float | min over steps of C_tilde − sum l | all |
| actuation_margin_max_kw | float | max over EV-steps of (reported − commanded) power; <= eps_q = 0.008 (A3) | all |
| energy_delivery_min_kwh | float | min over EV-steps of E(t+1) − E(t) − dt p; >= −1e-9 (A3) | all |
| n_nonfinite_actions, n_bisect_handover | int | guard counters (S 7.5 item 9) | all |
| max_kappa | float | largest taper exponent kappa of the episode (Lemma E2 condition) | all |
| penalty_return_eur | float | sum of r_pen (Alg. 5), EUR | all |
| n_steps | int | steps in the episode | all |
| **alpha** | float | conformal miscoverage level | conformal only (empty otherwise) |
| **conformal_coverage** | float | fraction of the episode's EVs with actual departure d >= predicted bound d_lo | conformal only |
| **n_evs_departed_before_bound** | int | EVs with d < d_lo (uncovered) | conformal only |
| coverage_frac | float | = conformal_coverage (alias read by A4's scripts) | conformal only |
| mean_d_minus_dlo | float | mean of d − d_lo over the episode's EVs, steps | conformal only |
| shortfall_departed_before_bound_kwh | float | shortfall of the uncovered EVs | conformal only |
| **oracle_profit_ceil_eur** | float | J_LP,ceil (Prop. O1(b)); **main gap reference (DD26)** | oracle files only |
| solve_time_s, solver_status, shortfall_stage_kwh | float, str, float | oracle solve time, HiGHS status of stage 2, stage-1 optimum z* (kWh) | oracle files only (Sec. 3.4) |
| admission, safety_layer | bool | wrapper settings of the evaluated method | all non-oracle |
| replay_seed | int | EV2Gym seed of the replay | all |
| replay_fingerprint | str | hash of sessions, prices, loads, PV of the replay; `MISMATCH:<hash>` if it differs from the manifest | all |
| train_N_set | str | training sizes of learned methods, e.g. `10+20` (`train_N` = largest, int) | learned only |

Oracle rows (`results/raw/oracle__{test_cfg_id}/episodes.csv`, method `oracle`): `profit_eur` = **J_LP**,
`oracle_profit_ceil_eur` = **J_LP,ceil**, `solve_time_s`, `solver_status`, `shortfall_stage_kwh`, plus
and every other Sec. 3.1 column filled: `n_evs`, `n_evs_capped`, `energy_charged/discharged_kwh` (net LP power),
`n_evs_unserved`, `energy_not_served_kwh`, `frac_evs_unserved`, `user_satisfaction_mean` (from the LP plan's
nominal departure energies vs the DD2-capped E_req), `n_evs_admission_reduced` = 0, violation, infeasible,
fallback and `a5_deficit_kwh` columns 0, `inference_ms_per_step_mean/_p95` = total LP time / T;
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
n_evs_admission_reduced_mean, n_admission_not_restored_mean, sps, update_time_s, rollout_time_s.

## 4. Method-name strings

Emitted in the `method` column (23 in the smoke run; EV-GNN adds 2; ablations and sensitivities add more):

| group | names |
|---|---|
| learned, main | `attn_safe`, `attn_penalty`, `mlp_safe`, `mlp_penalty` |
| SA-Attn without admission control (evaluation-only ablation) | `attn_safe_noadm` |
| SA-Attn-CP (trained as `attn_safe_conformal`, evaluated per alpha) | `attn_safe_conf_a0.01`, `attn_safe_conf_a0.05`, `attn_safe_conf_a0.1`, `attn_safe_conf_a0.2`, `attn_safe_conf_point` (uncalibrated, Q_hat = 0) |
| learned ablations | `attn_safe_beta0`, `attn_safe_curriculum`, `attn_safe_small`, `attn_penalty_dense`, `attn_penalty_w1`, `attn_penalty_w100` |
| 5-min sensitivity policy (DD28) | `attn_safe_dt5` |
| heuristics raw / + safety layer | `afap`, `alap`, `rr`, `llf`, `afap_safe`, `alap_safe`, `rr_safe`, `llf_safe` |
| MPC | `empc`, `empc_safe` (EV2Gym eMPC, Gurobi, N <= 25), `empc_highs`, `empc_highs_safe` (HiGHS port) |
| EV-GNN (optional, DD27) | `evgnn`, `evgnn_safe` (one model per profile; `seed` column = EV-GNN seed) |
| oracle | `oracle` |

Pattern rules: `<name>_noadm` = evaluated with `--admission off`; `_conf_a{alpha:g}` uses Python `%g`
formatting of alpha. The training config name is the second field of `exp_id`
(`{method}__{train_or_test_cfg}__s{seed}__{hash8}`); baseline exp_ids use the test grid file stem
(e.g. `llf_safe__test_grid__s0__1a2b3c4d`). Sensitivity configurations have ids such as
`public_N20_rho4_soc0.8` and `private_N20_rho4_dt5`; smoke ids carry `_T16`/`_T48`.
