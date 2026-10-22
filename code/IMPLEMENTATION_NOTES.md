# Implementation notes for the Orchestrator (A3, 2026-10-22)

Each item: what the method docs say, what the code does, and its status. The Orchestrator resolved I-3, I-4,
I-8 and I-20 with DD25-DD28 (design_decisions.md, 2026-10-22); I-21 reports the parameter-count check.
Round 3 (DD31-DD39): I-22 (controller-independent admission pre-pass, change vs S 7.4) and I-25..I-31; the former
I-22 (conformal calibration cost) is now **I-24**.

## Resolved by the Orchestrator

### I-3 Admission-control LP is structurally infeasible with full-battery targets (safety_layer.md S 7.4, DD9, DD20) — RESOLVED by DD25
* Docs: feasibility LP over all connected EVs with their admitted E_req, a *linear inner* taper minorant
  p <= rho (E_cap - E) after the first step; if infeasible even at the lower end, set the new request to
  min(E_arr, E_req) ("admission did not restore feasibility").
* Problem: the main scenarios use EV2Gym's desired capacity 1.0 (E_des = E_cap). For most EVs the DD2 cap is
  inactive in floating point (the exact taper reaches E_cap to machine precision after enough steps), so
  E_req = E_cap. The linear minorant only approaches E_cap asymptotically, so the LP can never satisfy
  E_i(d+1) >= E_cap: **the LP is infeasible whenever such an EV is connected**, independent of the new EV and of
  the transformer. The faithful rule then sets the new EV's request to E_arr (i.e. zero requirement). The same
  happens whenever a connected EV is individually infeasible after a fallback step.
* Evidence (smoke-level, not a result): workplace, N = 10, rho = 4, random policy, 3 episodes: 2 of 6-8 arrivals per
  episode got "not restored" (E_req := E_arr). Unit test `test_admission_full_battery_target_with_taper`
  reproduces it with a single connected EV and an unconstrained transformer.
* Code: both variants, switch `safety.admission_existing_targets` (configs/base.yaml) or
  `evaluate.py --admission-targets`:
  * `admitted` (DEFAULT, faithful): targets of connected EVs = admitted E_req;
  * `lp_reachable`: inside the LP only, the target of each connected EV is min(E_req, Psi^LP(E_i(t0)) - 1e-6)
    where Psi^LP is the maximum energy reachable in the LP model (exact first step, minorant afterwards).
    Admitted requests are still never reduced; only the feasibility question changes. With this option the
    new EV's request is reduced only by what the LP model cannot deliver (typically 1e-5..1e-2 kWh for full
    targets, below the 0.01 kWh reporting threshold) or by genuine capacity conflicts.
* **Resolution (DD25): `lp_reachable` is now the default** (configs/base.yaml, `config.DEFAULTS`,
  `WrapperOptions`, `AdmissionController`); `admitted` remains available. The unit test now asserts that, with a
  single full-target EV and an unconstrained transformer, the new request is reduced by < 0.01 kWh, that a
  connected full-target EV no longer blocks a new arrival, and that the `admitted` variant still shows I-3.

### I-4 J_LP is not an upper bound for EV2Gym controllers; use J_LP,ceil for the gap (formulation.md F 3, Prop. O1) — RESOLVED by DD26
* The smoke run shows a feasible controller (llf_safe, no violation, no shortfall) with profit -2.376 EUR above
  J_LP = -2.386 EUR on one replay: EV2Gym's unreported 0.01 kWh ceil gain lets real controllers buy slightly
  less energy than the LP needs. Prop. O1(b) covers this with J_LP,ceil; the test
  `test_oracle_upper_bounds_heuristics_on_tiny_replay` checks J_LP,ceil >= profit of every heuristic with
  realised shortfall <= z* and no violation.
* Code: oracle `episodes.csv` has `profit_eur` = J_LP and `oracle_profit_ceil_eur` = J_LP,ceil.
  **Resolution (DD26):** the main gap uses J_LP,ceil, the J_LP gap goes to the supplement. Oracle rows carry
  both (`profit_eur` = J_LP, `oracle_profit_ceil_eur` = J_LP,ceil) plus solve_time_s, solver_status,
  shortfall_stage_kwh; documented in SCHEMA_ADDITIONS.md.

### I-8 EV-GNN training protocol (Alg. 5 row 8) — RESOLVED by DD27
EV-GNN's script trains on one EV2Gym config per run. **Resolution (DD27):** one model per profile at N = 20,
rho = 4, 5 seeds, zero-shot on all (N, rho) of that profile. Implemented in `scripts/run_evgnn.sh` (clone at the
pinned commit `third_party/EVGNN_COMMIT` = 01aa1b7a676cca7aa077be453b28e29ea20344dd, train, evaluate raw and
+Pi); optional in the pipeline (`./reproduce.sh --with-evgnn`). EV-GNN's `train_RL_GNN.py` crashes under
numpy 2 (`np.Inf`); `scripts/evgnn_launch.py` restores the removed numpy aliases at runtime and runs the script
unmodified with runpy. The whole path (train per profile, zero-shot evaluation) ran in smoke mode. Evaluation rows
carry the EV-GNN seed in `seed` and the model directory in the exp_id hash. GNN-DT: not included (no public code).

### I-20 Sensitivity scope (DD2 desired SoC 0.8, DD13 5-min steps) — RESOLVED by DD28
`configs/test_grid_sensitivity.yaml` (3 profiles x {SoC 0.8, 5 min} at N = 20, rho = 4) and
`configs/ablations/train_attn_safe_dt5.yaml` (gamma = 0.99^(1/3), T = 336). Optional pipeline stage
`./reproduce.sh --stage sensitivity` (after `train`): builds the test sets, trains the dt5 policy (5 seeds),
evaluates the four main learned methods + all baselines on the SoC-0.8 configurations and the dt5 policy +
baselines on the 5-min configurations, solves the oracle LPs and merges. Smoke version:
`configs/test_grid_sensitivity_smoke.yaml`.

## Open / informational

### I-1 code_spec's "N = 5, T = 16" smoke settings vs EV2Gym's session model
EV2Gym re-draws all sessions of an episode until at least one fits the horizon (`load_ev_profiles`) and drops
every session whose departure falls within 4 steps of the end. With T = 16 (4 h) and the default minimum stay of
180 min nearly nothing fits; for the private profile (stays 12-15 h) nothing ever fits and **EV2Gym loops
forever**. Smoke settings therefore use min_time_of_stay = 30 min, start 14:00 and the public/workplace profiles
(`configs/smoke.yaml`, `configs/test_grid_smoke.yaml`). The wrapper guarantee test runs at T = 16 *and* T = 48.

## Faithful implementation choices worth knowing (no decision needed unless you disagree)

* **I-2 Loads/PV per replay.** EV2Gym's `tr_seed: -1` re-seeds loads/PV only with the *constructor* seed, so all
  resets of one env would share one load/PV draw (and the 10-column house sample). The wrapper sets
  `env.tr_seed = episode seed` before each reset (attribute, no patch). Test replays are (configuration, seed)
  pairs; `scenarios.fingerprint` hashes sessions, prices, loads, PV and rating; manifests store it and every
  episode row reports `replay_fingerprint` (MISMATCH on divergence). EV2Gym replay pickles are not used.
* **I-5 Admission bisection.** Feasibility is monotone in the new EV's request, so one LP maximising its
  departure energy gives y_max and the 12-step bisection of S 7.4 is evaluated with the exact predicate
  y <= y_max (+1e-6). Same output as 12 LP solves (unit test), 1 LP per arrival instead of up to 14.
  Admission LP horizon = max(d_i); C_hat uses realised values at t0, our noisy forecasts for t0+1..t0+8 and the
  last forecast held afterwards; P_tr(t) from the transformer schedule; margin n(t0) eps_q on all steps.
* **I-6 eMPC.** EV2Gym's `eMPC_V2G.get_action` crashes under numpy 2 (solution vector of shape (n, 1)), and with
  dict efficiency tables it uses the efficiency in percent (90) as a fraction. Without modifying EV2Gym, our
  adapter subclasses eMPC_V2G with the identical formulation and a 1-D solution vector, and divides the
  efficiency by 100. eMPC uses one efficiency (first EV) for all EVs and returns zero actions when infeasible
  (published behaviour). The HiGHS port (`empc_highs`) differs: net-power LP (exact for c_dis = c_ch), per-EV
  limits, exact first-step bounds (taper), wrapper's E_req for connected EVs, elastic departure constraint
  (1000 EUR/kWh); it uses EV2Gym's own load/PV forecasts, which our scenario YAML makes unbiased
  (forecast_mean 100, forecast_std 10; EV2Gym's defaults 30/5 are biased). Both read future arrivals within the
  horizon from `env.EVs_profiles` (information advantage, flagged in Alg. 5).
* **I-7 Conformal (Alg. 4).** EV objects do not carry the model name; the "EV model" one-hot uses the 6 distinct
  (capacity, AC power) spec classes (Tesla M3/MY and ID.3/Enyaq have identical specs). "Arrival step mod N_day"
  uses the episode step t_arr. Predictor: `GradientBoostingRegressor(loss='quantile', alpha)` (200 trees,
  depth 3); `HistGradientBoostingRegressor` was 25-150x slower here because of OpenMP contention with PyTorch.
  Sessions are drawn with EV2Gym's own `EV_spawner` after replicating reset's seeding/date logic (same law, much
  cheaper than full resets). d_lo = max(t_arr, ceil(q_hat - Q - 1e-9)). One calibration per (configuration,
  alpha) as DD23 requires, even though the session law does not depend on rho_tr. Training configurations of
  SA-Attn-CP (N 10/20 x profiles x rho) coincide with test configurations and reuse their calibrations.
  Cost: EV2Gym's spawner fills a 101-entry efficiency dict per EV (~4 ms/EV): ~1 h per N = 200 configuration,
  ~10-15 CPU-h in total (Alg. 7.4 estimated < 1 CPU-h; see I-24). Coverage columns (empty unless conformal is
  active): `alpha`, `conformal_coverage` (= `coverage_frac`), `n_evs_departed_before_bound`, `mean_d_minus_dlo`,
  `shortfall_departed_before_bound_kwh` (SCHEMA_ADDITIONS.md).
* **I-9 episodes.csv.** All Sec. 3.1 columns in the given order, then extra columns (`metrics.EXTRA_COLUMNS`).
  `train_N` = largest training N (20); `train_N_set` = "10+20". `n_evs_capped` and `n_evs_admission_reduced`
  use the 0.01 kWh threshold; `n_evs_unserved`/`energy_not_served_kwh` refer to the admitted E_req;
  `n_evs_unserved_dd2`/`energy_not_served_dd2_kwh` to the DD2-capped E_req (S 7.5 item 4 asks for both).
  `proj_dist_mean` = mean over A5 steps of ||x_occ - Pi(x)||_2 / N. `inference_ms_*` = arrival processing
  (cap, admission LP) + bounds + policy + projection/fallback, 1 torch thread; arrival steps make the p95 large.
  `infeasible_steps`, `a5_deficit_kwh` are also computed for raw methods (diagnostic; fb* = 0 there).
  Extra method names: `attn_safe_noadm`, `attn_safe_conf_a{alpha}`, `attn_safe_conf_point`, `empc_highs[_safe]`,
  `evgnn[_safe]`, `<heuristic>_noadm` (if `--admission off` is used with a '_safe' baseline).
* **I-10 Small network count.** See I-21 (resolved: the small ablation now uses head width 32 = 228,067).
* **I-11 Literature configuration** uses our constant-efficiency spec file (needed for Corollary 2), not EV2Gym's
  current-dependent tables; otherwise V2GProfitMax.yaml (25 chargers, 100 kW, workplace x5, 05:00, no loads/PV).
* **I-12 Forecasts (DD6).** true value x (1 + 0.1 eps), eps ~ N(0,1) drawn independently for every (t, k), clipped
  at 0; beyond the episode end the last true value is held. Since round 3 the noise of the forecast made at step t
  comes from an RNG keyed by (replay seed, t, quantity), so forecasts are a function of (replay, t) only (needed by
  the admission pre-pass, I-22). Future prices in the observation beyond T are held.
* **I-13 Penalty baselines (Alg. 5).** No safety layer (physical clip). Round 3 (DD31): admission on (the shared
  pre-pass requests), r_pen uses the shortfall vs the admitted request (= the DD2 cap when admission is off) and
  the untightened C(t), C_rev(t); the dense variant subtracts w_E dt sum (l - p)^+ with the commanded p.
  Actor-loss token mask for raw methods: occ & (u - p_min > eps_num); no fallback mask (w = 1).
* **I-14 PPO details (Alg. 2).** approx_kl early stop uses the mean KL of the epoch's minibatches; advantages are
  normalised per minibatch over all transitions; the entropy of a row is the mean over its influential tokens
  (0 if none); the beta term uses the stored (l, u, C_tilde, C_rev) on A5 rows only; float64 projection.
  Rollout actions are sampled with a dedicated torch generator; minibatch permutations with another one.
* **I-15 Resume.** checkpoint_last.pt every 50 updates (models, optimiser, RNG states, env seed streams); episodes
  in progress at the interruption are discarded, so a resumed run is valid but not bitwise identical.
* **I-16 Scenario mutation.** Training envs switch profile/rho per episode by updating `env.config` keys read at
  reset (hour, scenario, spawn multiplier, days, transformer, loads, PV, ev); N, dt, T are fixed per env. A test
  checks that a mutated env reproduces a fresh env's episode (fingerprint).
* **I-17 DD4 spec file.** constant efficiency per model = the 3-phase table value at the current level closest
  to the EV's nominal AC current (90 %, Hyundai Kona 93 %); `v2gsafe/specs/ev_specs_const_eta.meta.json`
  records the original tables. C0 and kappa <= kappa*(5) hold at 5 and 15 min (max kappa 2.39 at 15 min).
* **I-18 Tuning (Alg. 6).** SUPERSEDED by I-27 (DD37). Former protocol: validation: 20 episodes at N = 10, rho = 4, profiles cycled (7/7/6), validation seed
  range; score = mean profit (safety-layer methods) or mean r_pen return (penalty methods). Methods 1-4 are
  tuned; SA-Attn-CP and the attn_safe ablations inherit attn_safe's selection, penalty ablations attn_penalty's.
* **I-19 Training-time conformal.** SA-Attn-CP uses tau_lo in features, bounds and the DD2 cap during training,
  calibrated per training configuration at alpha = 0.1; evaluation at each alpha uses the test-configuration
  calibration of that alpha.

### I-21 Parameter counts vs formulation.md Sec. 5 (checked)
Main networks match exactly: actor 610,370, critic 676,481 (total 1,286,851), MLP actor 792,976. For the small
variant (d_m = 64, n_L = 2, n_H = 4, FFN 4 d_m = 256) the breakdown is:

| block | actor | critic |
|---|---|---|
| embeddings W_c (12->64) 832, W_g (38->64) 2,496, type embeddings 128 | 3,456 | 3,456 |
| 2 pre-LN blocks x 49,984 (LN 128, MHA in 12,480 + out 4,160, LN 128, FFN 16,640 + 16,448) | 99,968 | 99,968 |
| final LN | 128 | 128 |
| pooling: query 64 + MHA 16,640 | – | 16,704 |
| head with hidden width **64** (64->64->2 / 64->64->1) | 4,290 | 4,225 |
| head with hidden width **32** (64->32->2 / 64->32->1) | 2,146 | 2,113 |

Totals: hidden 64 -> 107,842 + 124,481 = **232,323**; hidden 32 -> 105,698 + 122,369 = **228,067**. The
document's number is therefore right for heads of width d_m/2 (the same ratio as the main 128 -> 64 heads); the
earlier code used a fixed width 64. `configs/ablations/train_attn_safe_small.yaml` now sets `head_hidden: 32`, and
`tests/test_models.py` asserts 228,067. If the paper describes the small heads as "d_m -> 64", the text should say
"d_m -> d_m/2" instead (no change to the main network).

### I-24 Conformal calibration cost and speed-up (formerly I-22)
Measured: EV2Gym's session spawner costs ~4 ms per EV (it fills a 101-entry efficiency table per EV), so 7,000
episodes take ~1 h per N = 200 configuration and **~10-15 CPU-h** for all 37 configurations (Alg. 7.4 estimated
< 1 CPU-h). Implemented speed-ups: (i) the session data sets are generated once per configuration for all alpha
values; (ii) they are cached in `results/conformal/sessions/<cfg>_sessions_t<n>_c<n>.npz` (re-calibration with
another alpha/predictor takes seconds); (iii) `--workers K` splits session generation over K processes (identical
result; worthwhile for large N, each worker builds its own env in ~4 s); the pipeline additionally runs one job per
configuration in parallel (`--jobs`). Not done (would change DD23): sharing sessions across rho_tr levels.

### I-22 Controller-independent admission pre-pass (DD31) — change vs safety_layer.md S 7.4
* **Docs (S 7.4, Alg. 3 step 1c):** at each arrival the admission LP starts from the *measured* energies
  E_i(t0) of the connected EVs. Their energies depend on the controller, so the admitted requests (and with them
  the shortfall reference of `energy_not_served_kwh`) differed between methods; penalty baselines ran without
  admission.
* **Code (DD31):** admission is a pre-pass over each replay, computed once and shared by all methods and the
  oracle (`v2gsafe/safety/admission_prepass.py`). Arrivals are processed in time order (charger index as
  tie-break); the arriving EV enters with its nominal arrival-time state (E_arr, E_des, d or the conformal d_lo,
  EV parameters); the capacity profile is the one S 7.4 prescribes (realised at t0, our forecasts made at t0, last
  forecast held; forecasts are a function of (replay, t), I-12); the energies of the EVs admitted earlier are read
  from a **nominal plan**: after each decision an elastic version of the same LP (slacks on targets and on the
  aggregate band, weight 1e5; objective: maximise sum_i sum_k min(E_i(k), target_i), i.e. reach targets as early
  as possible) is solved and its trajectories E_i(t) are stored; the next arrival reads E_i(t0') from them. The
  decision rule itself is unchanged: DD2 cap, LP with `lp_reachable` targets (DD25), y_max + exact 12-step
  bisection (I-5), requests of admitted EVs never reduced, "not restored" -> min(E_arr, E_req), re-certification
  with `cap_request`. Fast path: if the elastic plan with the unreduced request has zero slack, the request is
  feasible (a zero-slack plan is a feasible point of the admission LP) and no y_max LP is needed.
  The rule is causal (decision j uses information up to its arrival only), so an operator could run it online.
  Tables are cached per (test_cfg_id, replay seed, eps_q, targets, forecast noise, departure mode) in memory and
  in `results/admission/<cfg>__s<seed>__<hash8>.json` (pipeline stage `admission`); the wrapper looks the
  admitted request up by (charger, arrival step) and checks E_arr and d_ctrl against the table.
* **Why A4 re-certification is preserved:** the per-EV certificate (A4: E_arr >= B_hat(tau_arr; E_req)) depends only
  on the EV's own arrival state, its admitted request and its own threshold table. E_arr is exogenous (the same in
  every method) and the admitted request is re-certified by `cap_request` exactly as before (y -> E_req' <= y with
  E_arr >= B_hat[K](E_req')), so Proposition 1 and the departure guarantee hold for every EV not touched by an
  infeasible step, for every controller. What changes is only which y is offered to `cap_request`: before, y
  depended on the realised closed-loop energies of the other EVs, now on the nominal plan. The LP is still a
  forecast-based heuristic for the *joint* feasibility (A5 is still checked per step by the layer); the nominal
  plan can be more or less optimistic than the realised trajectory of a particular controller.
* **Consequences:** `n_evs_admission_reduced`, `energy_admission_reduced_kwh`, `n_admission_not_restored` and the
  admitted requests are identical across methods per replay (unit test
  `test_admission_is_controller_independent_and_certified`); shortfall columns `n_evs_unserved` /
  `energy_not_served_kwh` refer to the post-admission request for all methods, the `_dd2` columns to the
  arrival-capped request (M9). Penalty, Lagrangian and EV-GNN methods now also run with admission (their raw
  controllers ignore it apart from the features and the shortfall reference). `--admission off` (`_noadm`) is
  run only for attn_safe and attn_penalty (DD31). Conformal methods use their own table (d_lo instead of d: the
  controller does not know d), so their requests can differ from the others'.
* **Cost:** at most 3 LPs per arrival (plan; if the plan has slack: y_max and the plan with the reduced request),
  once per replay instead of once per method; measured times in README "Compute".

### I-23 Schema alignment with the analysis agent (A4), 2026-10-22
Checked by running A4's `make_all.py` on an untagged scratch copy of the smoke output. Changes on A3's side:
(i) method `episodes.csv` files omit the oracle-only columns (`oracle_profit_ceil_eur`, `solve_time_s`,
`solver_status`, `shortfall_stage_kwh`), which A4 parses as required numerics when present; (ii) oracle rows
fill every Sec. 3.1 column (LP-plan shortfall/satisfaction, zeros for violation/fallback columns, LP time / T as
inference time); (iii) evaluation variants of a training run get their own exp_id
(`attn_safe_noadm__train_attn_safe__s0__<hash>`), making (exp_id, test_cfg_id, replay_id) unique; (iv) train-log
episode statistics use a running window of the last 32 episodes (T > rollout length at 5 min), updates before the
first completed episode go to `train_log_warmup.csv`; (v) conformal columns `alpha`, `conformal_coverage`,
`n_evs_departed_before_bound` plus A4's `coverage_frac`, `mean_d_minus_dlo`; steps logs carry `method` and
`test_cfg_id`. A4's loader then accepts all files; its summary step fails on the smoke grid (A4 to check).
Details: SCHEMA_ADDITIONS.md.

### I-25 Oracle with taper and soft penalty (DD32)
* Taper law verified in EV2Gym `ev.py::_charge` (pinned commit): with the maximum pilot, below the transition
  SoC the SoC rises linearly (capped by P_nom), above it `new_soc = 1 + exp(m dsoc/(s_tr - 1)) (soc - 1)`, i.e.
  E' = E_cap - (E_cap - E) e^{-kappa}, kappa = m P_nom dt / (E_cap (1 - s_tr)); this is our phi (S7). Hence
  Pbar(E) = (E_cap - E)(1 - e^{-kappa})/dt **exactly** in the upper region, and 1/dt bounds the ratio everywhere.
* Oracle constraint (O4'): p^+(t_arr) <= Pbar(E_arr) (exact, column bound) and p^+(t) <= rho_out (E_cap - E(t)) for
  t > t_arr with rho_out = sup_E Pbar(E)/(E_cap - E) (`bounds.taper_outer_rho`: max of a 20,000-point grid
  sup and the analytic upper-region value, +1 %, capped at 1/dt). It is an **outer** (relaxing) linearisation:
  every power EV2Gym can deliver satisfies it (test `test_taper_outer_rho_dominates_ev2gym_taper`), so J_LP,ceil
  stays an upper bound. Rounding-aware (ceil) model: E_LP <= E_true (EV2Gym's unreported ceil gain), so
  rho_out (E_cap - E_LP) >= rho_out (E_cap - E_true) >= Pbar(E_true) >= p_rep: the relaxation survives the gain.
* Sessions use the **admitted** requests of the replay's admission table (DD31); `n_evs_admission_reduced` /
  `energy_admission_reduced_kwh` of oracle rows are the table's. Prices: EV2Gym's (discharge prices include the
  discharge price factor, DD39); throughput cost c_thr on p^+ + p^- (DD39). Hard lexicographic oracle unchanged
  otherwise (stage 1 min shortfall, stage 2 max profit; ceil variant with the same z* bound).
* Soft-penalty oracle `oracle_pen_profit_eur` = max profit - 10 sum z - 10 dt sum (o^+ + o^-) with the aggregate
  band relaxed by o^+/o^- and the ceil relaxation (E_req - G, E_min - G). For every method with the same
  admitted requests: its trajectory is feasible in this LP with z <= its shortfall and o = its overload, so
  `oracle_pen_profit_eur >= penalised_profit_eur >= penalised_profit_dd2_eur` (test
  `test_oracles_bound_every_controller_on_tiny_replays`, all heuristics raw and safe, incl. random actions,
  throughput cost and dpf 0.9). Not a bound for conformal variants (other admission table, I-22) — none is
  claimed. Per-episode `penalised_profit_eur` (and `_dd2`) is written for every method (w = 10 EUR/kWh fixed).
* **Observation (no decision needed, informs DD26):** with the taper rows, the non-ceil hard J_LP becomes very
  sensitive to full-battery requests (E_req = E_cap after the DD2 cap): the linear taper approaches E_cap only
  geometrically, so stage 2 with sum z <= z* + 1e-6 forces long expensive final charging, and on a smoke-level
  check (one N = 200 replay; not a result) J_LP moved by several EUR between the tolerances 1e-6 and 1e-4 kWh,
  while J_LP,ceil (targets relaxed by G = 0.01 kWh per step) and the soft oracle are stable. EV2Gym itself
  reaches E_cap only through floating-point saturation of the same exponential. The main gap (J_LP,ceil, DD26)
  and the soft bound are unaffected; the supplementary J_LP should be read with this in mind.
* Solver robustness: these near-degenerate stage-2 LPs occasionally end with HiGHS status 'Unknown' under dual
  simplex; every oracle LP is retried once with HiGHS' interior-point solver, and `solver_status` then reads
  `Optimal(ipm)`.

### I-26 PPO-Lagrangian `attn_lagrangian` (DD33(3))
Attention policy, no safety layer (physical clip, like the penalty baselines), admission on, profit reward.
Costs per step: c_E = departure shortfall vs the admitted request (kWh, at departures), c_P = dt x overload
(kWh, either direction, realised C); normalised like DD15 as c / (N P_ref dt). Critic with 3 outputs (value,
2 cost values; +260 parameters), cost GAE with the same (gamma, lambda), combined advantage
(A_r - lam_E A_E - lam_P A_P)/(1 + lam_E + lam_P) (then the usual per-minibatch normalisation), value loss over
the 3 heads. Dual ascent once per rollout before the policy update: lam <- clip(lam + lr_lambda (J_c - d), 0,
lam_max) with J_c = mean normalised cost per step x T (per-episode cost) and limits d = 0 (any expected cost raises
lam). lam is logged (`lambda_short`, `lambda_over`, `cost_short_ep`, `cost_over_ep` in train_log.csv) and stored
in checkpoints. Scale: lam = 40 corresponds to the penalty weight w = 10 EUR/kWh at c_ref = 0.25 EUR/kWh.
**Decision point:** lr_lambda grid {0.5, 2, 8} (default 2) was chosen for this normalisation; d = 0.

### I-27 Tuning protocol (DD37) — replaces I-18
Equal budget K = 12 configurations per tuned method (default point + 11 sampled without replacement from the
method's grid, fixed seed per method): common grid lr {1e-4, 3e-4, 1e-3} x minibatch {256, 512, 1024} (the
default 512 is in the grid) x ent_coef {0, 1e-3, 1e-2}; + w in {1, 10, 100} (w_E = w_P) for penalty-reward methods
(attn_penalty, mlp_penalty, attn_proj_nolb); + lr_lambda {0.5, 2, 8} for attn_lagrangian. Tuned methods:
attn_safe, attn_penalty, mlp_safe, mlp_penalty, attn_proj_nolb, attn_lagrangian, deepsets_safe (7 x 12 runs x
0.5e6 steps, seed 0, trained on the method's training mixture). Validation: 24 episodes cycled over the
training-mixture configurations, `val_seed` range (disjoint from training and test replays). **Selection
criterion (decision point):** mean validation J^pen with fixed w = 10 EUR/kWh (= `penalised_profit_eur`) for every
method, so that a tuned penalty weight cannot win by lowering its own penalty. Variants inherit the selection of
their base (pipeline.TUNED_METHODS: conformal/ablations/dt5/trainN50 <- attn_safe; penalty_dense <- attn_penalty;
w1/w100 <- attn_penalty PPO keys only, their w is fixed by design). `reproduce.sh` refuses to start final training
without `results/tuning/<base>/selected.yaml` unless `--hparams FILE` is given (`configs/hparams_defaults.yaml` =
explicit Alg. 6 defaults; such runs should be reported as untuned).

### I-28 New baselines (DD33, DD34)
* `attn_proj_nolb`: safety layer with l_i = p_i^min (no departure lower bound; `safety.lower_bound: false`), r_pen
  reward (departure-shortfall penalty), beta = 0.01. Features still contain the departure-aware l_i (token 10) so
  the observation is identical to SA-Attn; metrics (`a5_deficit_kwh`, touched EVs) use the departure-aware bound.
* `attn_penalty_safedeploy`: attn_penalty checkpoints evaluated with the safety layer (`--deploy-safety`).
* `deepsets_safe`: per-token MLP on [token; global] (3 layers, width 448), mean-pooled context, head on
  [phi_i; context] (448*2 -> 224 -> 2); critic on [mean phi; MLP(global)]. Same safety layer/reward/PPO as SA-Attn.
  DD42(3): widened to match the attention model within +-10 %: actor 626,530 (+2.6 % vs 610,370), critic 643,777
  (-4.8 % vs 676,481) (test `test_deepsets_width_matches_attention_parameter_counts`).
* `sllf[_safe]`: sLLF of [chen2022smoothed] as summarised in lit/parts/L1_notes.md (closed form clip around a common
  level found by bisection, total = min(P, sum of caps)); our instantiation: rbar_i = P_nom,i, laxity vs the
  certified thresholds chi0_i = (E_i - B_hat_i(tau_i - 1))/(P_nom dt), caps u_i, P = max(0, C_tilde); charging only.
* `price_llf[_safe]`: LLF when the price is below the median of past prices + 8-step forecast, p_min above, idle on
  ties (simple price-aware rule, documented in the class).
* `empc_highs_24h[_safe]`: HiGHS eMPC with a 24-h horizon (extended set). DD34: `empc_highs` runs on every N of the
  grid; the Gurobi eMPC (`empc`, `empc_safe`) only on `literature_N25` (extended set, needs gurobipy).

### I-29 in_distribution and GNN-DT (DD35)
`in_distribution` (bool, method rows): learned methods: test configuration is one of the configurations of the
method's training mixture (profiles x rho_tr x training N, same dt/horizon/economics); EV-GNN: N = 20, rho_tr = 4,
default economics (its per-profile training configuration); empty for heuristics/MPC/oracle. GNN-DT: the public
code (github.com/StavrosOrf/DT4EVs) was cloned and pinned at 8b327d122efe13678df1dc74c42c673ce33541d9
(`third_party/GNNDT_COMMIT`); it ships its own modified EV2Gym copy, Gurobi-generated offline datasets and
Python-3.8-era pins, so only instructions and a stub are provided (`v2gsafe/baselines/gnndt_adapter.md`,
`scripts/run_gnndt.sh`, `--with-gnndt`); estimated 2-4 person-days to port.

### I-30 Test grid and stages (DD36, DD39)
`configs/test_grid.yaml`: + rho_tr = 3 kW at N in {20, 200} for all profiles (100/50 replays) -> 43 configurations,
3,550 episodes per policy. `configs/train_attn_safe_trainN50.yaml` (16 envs at N = 50, 3 seeds), **core set since
DD42(5)**: trained in stage `train`, evaluated in stage `evaluate` at N in {50, 100, 200} of the main grid (incl.
rho 3 at 200); `./reproduce.sh --stage reference` re-runs only these jobs. DD39 economics variants are
evaluation-only (no retraining, DD42(7)).
Sensitivity grid: + throughput cost 0.05 EUR/kWh (`throughput_cost`, charged on |p_rep| in reward, profit,
eMPC and oracle objectives) and discharge price factor 0.9 (EV2Gym `discharge_price_factor`), N = 20, rho 4, three
profiles, evaluated with the main learned policies (no retraining) and all baselines; ids `*_thr0.05`, `*_dpf0.9`.

### I-31 Fallback-case mapping (M7)
Paper (i) = FB-1* (code 3), (ii) = FB-2 (code 2), (iii) = FB-1 (code 1), (iv) = FB-0 (code 0); FB-0 also occurs on a
contradictory step whose tightened budget C_tilde is negative (no EV can be served at its lower bound). Written into
SCHEMA_ADDITIONS.md.

### I-32 Round-2 review fixes (DD43)
* (1) **Shared requirement for every pass.** `SafeV2GEnv.reset` always loads the shared admission table of the replay
  (true departures). With admission on, the requirement is the shared admitted value (conformal passes: re-certified
  with their own K from d_lo, so <= shared; this supersedes DD42(2)'s separate conformal table); with `_noadm` it
  stays the DD2 request. At arrival the wrapper writes the requirement into the spawned EV2Gym EV
  (`ev.desired_capacity`; the profile objects and fingerprints are untouched), so baselines that read EV2Gym state
  (EV-GNN's state function, Gurobi eMPC for connected EVs) target it. The Gurobi eMPC reads all session targets from
  `env.EVs_profiles` at construction: `EMPCGurobi.reset` temporarily sets them to the table values and restores them
  (test checks Cxf and the fingerprint); its commands are clipped to [-1, 1] and to [p_min, u]. The HiGHS eMPC uses
  the table entries for arrivals inside its horizon. New columns `n_evs_unserved_shared`,
  `energy_not_served_shared_kwh` (vs the shared requirement, every pass). `user_satisfaction_mean` keeps EV2Gym's
  formula against the original E_des (computed by the wrapper because `desired_capacity` is overwritten).
* (2) New columns `n_evs_stay_infeasible` (EVs connected during >= 1 step with A5 false; A5 with the layer's
  bounds, i.e. p_min for attn_proj_nolb), `share_evs_stay_feasible` (E4), `agg_upper_active_steps` /
  `agg_lower_active_steps` (commanded sum within 1e-6 relative of C_tilde / -C_rev or beyond; on projected steps this
  is lambda > 0 / < 0), `lb_active_frac` (occupied token-steps with p_i = l_i > p_i^min, departure-aware l_i).
  Per-case fallback counts are the existing `fb*_steps`.
* (3) MLP baselines are evaluated only at N <= 20 (`evaluate.MLP_EVAL_MAX_N`; the pipeline drops larger-N jobs
  with a logged note; evaluate.py itself skips such configurations with a message). The network keeps N_max = 200
  slots (792,976 actor parameters as in formulation.md); only slots <= 20 are ever trained. Consequence: no MLP rows
  on `literature_N25` (algorithm.md row 7 listed it; DD43(3) wins).
* (4) attn_proj_nolb beta = 0.01 (unchanged, `configs/train_attn_proj_nolb.yaml`).
* (5) Fixed seeds: 5 per method, 3 for attn_safe_trainN50 and w1/w100; no adaptive rule exists in the code. The
  statistical tests are in the analysis code (A4), not here.
* (6) EV-GNN protocol recorded in `meta.json` (`evgnn_training`: steps as run, EV-GNN default 1e7, config file,
  train N, "one load/PV draw per training run", model dir/name, commit) and stated in README Sec. 3.

### I-33 MLP baseline size (Orchestrator, DD44)
N_max = 20 (evaluated only at N ≤ 20, DD43-3) and hidden width 640: actor 614,440 parameters (attention actor 610,370, +0.7 %), critic 589,441 (attention critic 676,481, −12.9 %). Replaces 2 × 256 at N_max = 200 (792,976). The earlier figures in this file and in DONE.md refer to the old setting.
