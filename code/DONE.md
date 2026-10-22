# DONE — Implementation agent A3 (2026-10-22; revised after the Orchestrator decisions DD25-DD28)

Code infrastructure for the experiments, built and tested in the cloud workspace (Python 3.13, torch 2.14 CPU,
2 cores, 7 GB, EV2Gym at the pinned commit). **No experiment result was produced or is reported here**; every
smoke output lives in `results/smoke/` and is tagged `smoke = true`.

## Files (root = `code/`)

| path | content |
|---|---|
| `README.md` | install (venv/conda/GPU), smoke test, full pipeline, run counts and compute, outputs, resume, multi-machine, disk, troubleshooting, what to send back, indicative timings |
| `IMPLEMENTATION_NOTES.md` | resolved items (I-3 → DD25, I-4 → DD26, I-8 → DD27, I-20 → DD28), I-21 parameter-count breakdown, I-22 conformal cost, all code-level choices |
| `SCHEMA_ADDITIONS.md` | columns beyond code_spec Sec. 3 (episodes, oracle, steps, train_log) and the exact method-name strings |
| `pyproject.toml`, `requirements.txt` (tested pins), `environment.yml`, `Makefile`, `.gitignore` | packaging (`make install / test / smoke / reproduce / tables / figures / bench`) |
| `third_party/EV2GYM_COMMIT`, `scripts/install_ev2gym.sh` | pinned EV2Gym commit; clone + checkout + `pip install -e` + undeclared EV2Gym deps (tested in a fresh Python 3.10 venv) |
| `third_party/EVGNN_COMMIT`, `scripts/run_evgnn.sh`, `scripts/evgnn_launch.py` | DD27: EV-GNN at pinned commit 01aa1b7a, one model per profile at N = 20, rho 4, 5 seeds, zero-shot evaluation (`evgnn`, `evgnn_safe`); numpy-2 launcher (EV-GNN files untouched) |
| `reproduce.sh` → `v2gsafe/pipeline.py` | stages scenarios, conformal, tune, tune_select, train, evaluate, evgnn (optional `--with-evgnn`), oracle, merge, analysis, and the optional composite `--stage sensitivity` (DD28); `--smoke --stage --jobs --device --tune --seeds --shard i/M --list`; resumable |
| `train.py`, `evaluate.py` | CLIs (code_spec Sec. 1) |
| `configs/` | `base.yaml` (Alg. 6), `train_attn_safe`, `train_attn_penalty`, `train_mlp_safe`, `train_mlp_penalty`, `train_attn_safe_conformal`, `ablations/` (beta0, curriculum, small, penalty_dense, penalty w1/w100, dt5), `test_grid.yaml` (36 + literature, 3100 episodes), `test_grid_sensitivity.yaml` (DD28) + smoke version, `smoke.yaml` (overlay), `test_grid_smoke.yaml` |
| `v2gsafe/constants.py`, `config.py`, `utils/` | constants, YAML config with inheritance/overlays/strict keys/hash, seeding, meta.json, CSV |
| `v2gsafe/specs/` + `scripts/make_ev_specs.py` | constant-efficiency spec JSON (DD4) + metadata; C0 / Lemma E2 checks |
| `v2gsafe/envs/` | `scenarios.py` (profiles, DD24 scaling, seed ranges, test grid, replay manifests with fingerprints, light session sampler), `wrapper.py` (SafeV2GEnv: DD2 cap, admission and conformal hooks, Alg. 1.1 bounds, 12+38 token features, projection/fallback/physical clip, actuation, DD7 metric, r and r_pen, step logs), `forecasts.py` (DD6), `vec_env.py` (sync/subprocess) |
| `v2gsafe/safety/` | `bounds.py` (S7, S8/S10 thresholds, cap_request, S11 bounds, taper minorant), `projection.py` (NumPy port + batched PyTorch Alg. 1.2 + Jacobian S16 + guards), `fallback.py` (FB-0/1/2/1*), `actuation.py` (S 6.3), `admission.py` (S 7.4 LP, HiGHS), `conformal.py` (Alg. 4), `lp.py` (highspy helper) |
| `v2gsafe/models/` | attention actor/critic (F 5; 610,370 / 676,481 parameters, exact), MLP baseline (792,976) |
| `v2gsafe/algos/` | `ppo.py` (Alg. 2: mixed-N padding, raw-sample log-prob, DD21 masks with fixed normaliser, Moreau beta term, DD15 reward scaling, GAE, clipping, entropy, grad clip, LR decay, target-KL, checkpoints, resume, curriculum), `buffers.py` |
| `v2gsafe/baselines/` | `heuristics.py` (AFAP/ALAP/RR/LLF, raw and +Pi), `mpc.py` (EV2Gym eMPC adapter, numpy-2-safe; HiGHS port), `oracle.py` (lexicographic LP + J_LP,ceil), `evgnn.py` + `evgnn_adapter.md` |
| `v2gsafe/eval/` | `evaluate.py` (episodes.csv exactly per Sec. 3.1 + extras, step logs, inference timing, parallel, resume), `metrics.py` (schema, accumulator), `merge.py` |
| `v2gsafe/tuning.py` | Alg. 6 tuning grid, validation, selection |
| `scripts/benchmark.py` | env-step / PPO-update timings |
| `tests/` | 61 test functions (80 test cases), `tests/reference/projection_ref.py` = verbatim copy of the A2 oracle |

## Tests

`python -m pytest -q tests`: **80 passed, 0 failed** (3 min 25 s, Python 3.13), after the DD25-DD28 changes. An
earlier version of the suite (76 tests) also passed on **Python 3.10.20** in a fresh venv installed only from `pyproject.toml`
ranges + `scripts/install_ev2gym.sh` (76 passed; numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, torch 2.14.1).
All files byte-compile with Python 3.10/3.11/3.12.

Coverage of code_spec Sec. 4 and the task list:
* projection: 10,200 rows incl. tight A5 (upper/lower), |x| up to 1e8, hyperplane, NumPy and PyTorch: bounds and
  aggregate limits within 1e-9; NumPy port bit-identical to `projection_ref`; PyTorch within 1e-9 (relative on
  the adversarial families); padding/masks with garbage; NaN/Inf guard and counters; Jacobian vs finite
  differences; Moreau gradient of the beta term;
* bounds: phi vs reference and vs EV2Gym's `EV._charge`; thresholds and DD2 cap vs reference, certified (S10),
  A4; S11 bounds vs reference and the S12 closed form; Lemma 1 through actuation + EV2Gym `EV.step`; taper
  minorant is inner; C0 / kappa conditions; arrival-cap example;
* fallback identical to reference, Lemma 2 properties, LLF order; actuation identical to reference, realised
  power in [p, p + eps_q] through EV2Gym's charger;
* admission LP small cases (feasible, reduced = brute-force 12-LP bisection, not restored with `admitted`);
  **DD25 default `lp_reachable`: a single full-target EV with an unconstrained transformer is reduced by
  < 0.01 kWh and a connected full-target EV no longer blocks a new arrival**; conformal quantile,
  **coverage on synthetic exchangeable data >= 1 - alpha - 0.02** for alpha in {0.05, 0.1, 0.2}, save/load,
  session cache reproduces the calibration exactly;
* models: shapes for N in {5, 10, 50, 100, 200}, permutation equivariance/invariance (1e-10, float64), padding
  invariance (garbage, extra padding, station of size n), init, parameter counts, MLP padding;
* PPO: one update on mixed-N batches (attention and MLP) runs and is bit-deterministic under a seed; masked
  tokens / fallback steps do not change the actor loss or gradient; GAE vs manual; checkpoint save/load;
* wrapper with real EV2Gym (N = 5, T = 16 and T = 48, tight transformer, random raw actions): **zero transformer
  violations on A5 steps, zero unserved EVs among EVs never touched by an infeasible step**, bounds always hold,
  actuation margin <= 0.008 kW; determinism and scenario mutation (fingerprints); observation features; reward
  = EV2Gym profit;
* synthetic closed loop (T4 style, 5 and 15 min, 739 EVs): Theorem 1(a)-(c), Corollary 1, Lemma 3 with zero violations;
* evaluate.py writes all Sec. 3.1 columns (conformal/oracle extras empty, step logs with method and test_cfg_id)
  and resumes; small attention network = 228,067 parameters (I-21); oracle J_LP,ceil >= every heuristic with
  realised shortfall <= z* on tiny replays; eMPC-HiGHS runs; **train.py smoke run (N in {3, 5}, 2,048 steps)
  completes**, and an interrupted run resumes to the same length.

## Smoke runs (2 cores; outputs in `results/smoke/`, 9.8 MB, never reported)

* `./reproduce.sh --smoke --jobs 2 --with-evgnn`: **exit 0 in 501 s**, 0 failed jobs. Stages scenarios,
  conformal (4 test + 8 training configurations x 4 alpha, tiny n), train (5 learned methods, 2,048 steps each),
  evaluate, **evgnn** (EV-GNN trained per profile with its own script through `scripts/evgnn_launch.py`, then
  zero-shot evaluation raw and +Pi), oracle, merge; analysis skipped (`../analysis/scripts/make_all.py` is not in
  this tree).
* `./reproduce.sh --smoke --jobs 2 --stage sensitivity`: **exit 0 in 169 s** (SoC-0.8 and 5-min smoke
  configurations, dt5 policy trained with T = 48, main policies + baselines + oracle evaluated, merged).
* Final content: 38 run directories, **262 episode rows, 26 method names, 6 test configurations**, every row
  `smoke = true`, 0 fingerprint mismatches, 0 duplicate (exp_id, test_cfg_id, replay_id). On the 146
  safety-layer episodes: 0 transformer-violation steps; 3 unserved EVs, all in SoC-0.8 episodes and all touched by
  FB-1 steps, with shortfall below the Lemma 3 deficit bound (Corollary 1 / Lemma 3 behaviour).
* **A4 compatibility**: on an untagged scratch copy of this output, A4's `make_all.py` loads everything
  (250 episodes, 25 methods, 6 configurations, 12 oracle episodes, 6 training logs) after three fixes on A3's side
  (oracle-only columns only in oracle files; oracle rows fill all Sec. 3.1 columns; unique exp_ids for evaluation
  variants; windowed train-log statistics). A4's next step (`stats.summary_table`) then fails on the smoke grid
  with an empty summary (KeyError on the summary columns), presumably because the smoke configurations are not in
  its canonical grid; that is for A4 to check, not a schema problem.

## Indicative timings (this machine; details in README Sec. 4)

N = 10 / 50 / 200: wrapper observe 0.36 / 1.08 / 3.34 ms, step 0.59 / 1.37 / 3.73 ms, observe with admission LP
0.70 / 5.5 / 66 ms (mean), actor forward 1.3 / 2.1 / 5.3 ms. PPO update (2048 transitions, 10 epochs, N = 10):
16.6 s; per epoch 9.4 s at N = 50; at N = 200 a 512-minibatch epoch exceeds 7 GB RAM (38.6 s per epoch with
minibatch 64). Training a 3e6-step run on 2 cores: ~7 h.

## Known limitations

* Smoke runs exercise every code path but are far too small to say anything about performance.
* Training at full scale, the conformal calibration at full size, EV-GNN training and Gurobi-eMPC at N = 25
  were not run here (compute). eMPC (Gurobi) was run on the smoke grid only (N <= 5).
* EV2Gym loops forever if no session of a configuration can fit the horizon (only relevant for unusual
  horizons; see I-1). EV2Gym's session spawner is slow (~4 ms per EV), which dominates conformal calibration.
* `requirements.txt` pins the versions tested on Python 3.13; on 3.10-3.12 use the pyproject ranges (tested on 3.10).
* The penalty-based and conformal methods depend on the I-3 decision only through the safety-layer variants
  (penalty methods run without admission control).

## Status of the earlier open issues

1. I-3 → DD25 implemented (`lp_reachable` default; `admitted` kept as option).
2. I-4 → DD26: oracle files carry `profit_eur` (J_LP) and `oracle_profit_ceil_eur` (J_LP,ceil), solve_time_s,
   solver_status, shortfall_stage_kwh; the gap reference is for A4 (SCHEMA_ADDITIONS.md).
3. I-8 → DD27 implemented and smoke-tested end to end (`--with-evgnn`); GNN-DT not included.
4. I-20 → DD28 implemented as `--stage sensitivity`, smoke-tested.
5. I-10/I-21: the document's 228,067 is right for heads of width d_m/2; the small ablation now uses width 32.
6. I-22: conformal cost ~10-15 CPU-h measured; sessions cached per configuration and parallelisable
   (`--workers`); sharing across rho_tr would need a DD23 change (not done).
7. Schema alignment with A4: method files no longer contain empty oracle-only columns, oracle rows fill every
   Sec. 3.1 column, conformal columns `alpha`, `conformal_coverage`, `n_evs_departed_before_bound` (+ A4's
   `coverage_frac`, `mean_d_minus_dlo`); verified by running A4's `make_all.py` on an untagged copy of the smoke
   output (see the smoke section). For DD26, A4's oracle join must also carry `oracle_profit_ceil_eur`.
