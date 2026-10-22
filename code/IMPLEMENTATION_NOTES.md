# Implementation notes for the Orchestrator (A3, 2026-10-22)

Each item: what the method docs say, what the code does, and its status. The Orchestrator resolved I-3, I-4,
I-8 and I-20 with DD25-DD28 (design_decisions.md, 2026-10-22); I-21 reports the parameter-count check.

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
  ~10-15 CPU-h in total (Alg. 7.4 estimated < 1 CPU-h; see I-22). Coverage columns (empty unless conformal is
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
  at 0; beyond the episode end the last true value is held. Future prices in the observation beyond T are held.
* **I-13 Penalty baselines (Alg. 5).** No safety layer (physical clip), admission off, r_pen uses the DD2 cap
  and the untightened C(t), C_rev(t); the dense variant subtracts w_E dt sum (l - p)^+ with the commanded p.
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
* **I-18 Tuning (Alg. 6).** Validation: 20 episodes at N = 10, rho = 4, profiles cycled (7/7/6), validation seed
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

### I-22 Conformal calibration cost and speed-up
Measured: EV2Gym's session spawner costs ~4 ms per EV (it fills a 101-entry efficiency table per EV), so 7,000
episodes take ~1 h per N = 200 configuration and **~10-15 CPU-h** for all 37 configurations (Alg. 7.4 estimated
< 1 CPU-h). Implemented speed-ups: (i) the session data sets are generated once per configuration for all alpha
values; (ii) they are cached in `results/conformal/sessions/<cfg>_sessions_t<n>_c<n>.npz` (re-calibration with
another alpha/predictor takes seconds); (iii) `--workers K` splits session generation over K processes (identical
result; worthwhile for large N, each worker builds its own env in ~4 s); the pipeline additionally runs one job per
configuration in parallel (`--jobs`). Not done (would change DD23): sharing sessions across rho_tr levels.

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
