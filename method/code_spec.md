# Code infrastructure specification (Orchestrator → A3, A4)

Binding interface contract between the implementation (A3) and the analysis (A4) agents. The experiments are run by the human author on another computer. Here we only build and smoke-test. **No result produced here may be reported in the paper.** Smoke-test outputs are written under `results/smoke/` and are tagged `"smoke": true`.

Method source of truth: `method/formulation.md`, `method/safety_layer.md`, `method/algorithm.md`, `method/design_decisions.md` (DD1-DD24), `method/nomenclature.md`, `method/projection_ref.py` (NumPy reference implementation; reuse its logic, keep it as the test oracle).

## 1. Repository layout (root = `code/` inside the project; Python package `v2gsafe`)

```
code/
  README.md                 how to install and run on the author's machine (CPU and GPU), expected run times
  pyproject.toml            package metadata, pinned deps; also requirements.txt and environment.yml
  requirements.txt
  environment.yml
  reproduce.sh              end-to-end: (1) build scenarios, (2) train all learned methods × seeds, (3) evaluate all methods on fixed test sets, (4) run the oracle, (5) call analysis scripts. Flags: --smoke, --stage <name>, --jobs N, --device cpu|cuda
  Makefile                  targets: install, test, smoke, reproduce, tables, figures
  third_party/EV2GYM_COMMIT  contains 55ef368c25a58476a9eeaed5111f112a63a38ca8; install script clones github.com/StavrosOrf/EV2Gym at this commit (no patches)
  v2gsafe/
    __init__.py
    config.py               dataclass/YAML config loading, defaults from algorithm.md Alg. 6, config hashing
    specs/ev_specs_const_eta.json   our constant-efficiency EV specs (DD4), derived from EV2Gym's file by a script
    scripts/make_ev_specs.py
    envs/
      scenarios.py          build EV2Gym YAML configs for (profile, N, rho_tr, dt, desired_soc, seed) per Alg. 7 / DD24; generates and caches test replays
      wrapper.py            SafeV2GEnv: wraps EV2Gym; token observations (12 charger + 38 global features, formulation §4.2); bounds; actuation (round-up to action grid); per-step logging; admission control hook; conformal hook
      forecasts.py          noisy forecasts (DD6)
    safety/
      bounds.py             reachable energy, thresholds at arrival, l_i, u_i, p_min, arrival cap (DD2)
      projection.py         robust batched projection (NumPy + PyTorch), Jacobian; must match projection_ref.py
      fallback.py           FB-0/1/2/1* rules
      admission.py          admission-control LP (HiGHS via highspy)
      conformal.py          split-conformal departure lower bound (Alg. 4)
    models/
      attention_policy.py   actor (encoder, shared head), critic (attention pooling), masking
      mlp_policy.py         MLP baseline with zero padding to N_max
    algos/
      ppo.py                PPO with variable N (Alg. 2), raw-sample log-prob, beta regulariser, masking (DD21)
      buffers.py
    baselines/
      heuristics.py         AFAP, ALAP, Round Robin, LLF (all pass through the same actuation; variants with/without safety layer as in Alg. 5)
      mpc.py                adapter to EV2Gym eMPC (Gurobi, N<=25) and our HiGHS port for larger N
      oracle.py             perfect-information LP (formulation §3, lexicographic, rounding-aware variant), HiGHS
      evgnn_adapter.md      how to run EV-GNN / GNN-DT public code on our scenarios (instructions only unless it runs)
    eval/
      evaluate.py           runs any method on a test set, writes the result files of §3
      metrics.py            per-episode metrics
    utils/  (seeding, logging, git hash, timing)
  train.py                  CLI: python train.py --config configs/xxx.yaml --seed S --out results/raw/<exp_id>
  evaluate.py               CLI: python evaluate.py --method <name> --checkpoint ... --test-set ... --out ...
  configs/
    base.yaml
    train_attn_safe.yaml, train_attn_penalty.yaml, train_mlp_safe.yaml, train_mlp_penalty.yaml, train_attn_safe_conformal.yaml
    test_grid.yaml          36 configurations + literature config (Alg. 7)
    smoke.yaml              tiny: N in {3,5}, T=16, 2k steps
tests/                      pytest (at project root `tests/`, or `code/tests/`; choose `code/tests/` and symlink nothing)
```

## 2. Experiment identifiers
`exp_id = {method}__{train_or_test_cfg}__s{seed}__{config_hash8}`. Methods: `attn_safe`, `attn_penalty`, `mlp_safe`, `mlp_penalty`, `attn_safe_conf_a{alpha}`, `afap`, `alap`, `rr`, `llf`, `afap_safe`, `alap_safe`, `rr_safe`, `llf_safe` (heuristic + safety layer variants if defined in Alg. 5), `empc`, `oracle`, `evgnn` (optional).
Each run directory `results/raw/{exp_id}/` contains: `config.yaml` (resolved), `meta.json` (git hash of the project repo, EV2Gym commit, package versions, hostname, device, start/end time, wall-clock seconds, smoke flag), `train_log.csv` (learned methods), `checkpoint_*.pt`, and evaluation outputs (§3).

## 3. Result file schema (A4 reads exactly these)

### 3.1 `episodes.csv` (one row per evaluated episode)
| column | type | meaning |
|---|---|---|
| exp_id, method, seed | str, str, int | |
| train_N | int or empty | N used for training (learned methods) |
| test_cfg_id | str | e.g. `workplace_N50_rho4` |
| profile | str | workplace / public / private |
| N | int | test station size |
| rho_tr | float | kW per charger |
| dt_min | int | step length |
| replay_id | int | index of the fixed test episode |
| profit_eur | float | Σ r(t), realised, € |
| energy_charged_kwh, energy_discharged_kwh | float | |
| n_evs | int | EVs that arrived and were admitted |
| n_evs_capped | int | requests reduced by the arrival cap by > 0.01 kWh (DD2) |
| n_evs_admission_reduced | int | requests reduced by admission control |
| n_evs_unserved | int | EVs with E(d+1) < E_req − 0.01 kWh |
| energy_not_served_kwh | float | Σ_i (E_req − E(d+1))^+ |
| frac_evs_unserved | float | n_evs_unserved / n_evs |
| user_satisfaction_mean | float | EV2Gym-style mean E(d+1)/E_desired |
| tr_violation_steps | int | steps with transformer violation > 1e-6 kW (DD7 metric) |
| tr_violation_kwh | float | Σ_t violation(t)·Δt |
| tr_violation_max_kw | float | |
| infeasible_steps | int | steps with A5 false |
| fb0_steps, fb1_steps, fb2_steps, fb1star_steps | int | fallback activations |
| a5_deficit_kwh | float | Lemma 3 deficit |
| proj_dist_mean | float | mean ‖x − Π(x)‖ / N (learned methods) |
| inference_ms_per_step_mean, inference_ms_per_step_p95 | float | policy + safety layer, wall clock |
| oracle_profit_eur | float or empty | filled by the analysis join |
| smoke | bool | |

### 3.2 `steps.parquet` (optional, only for example-day figure; flag `--log-steps`)
columns: exp_id, replay_id, t, price_ch, price_dis, P_tr, P_inf, P_pv, C, C_rev, sum_p, sum_l, sum_u, n_connected, fallback_code, and per-charger p_i (long format: separate `steps_chargers.parquet` with t, i, p, l, u, E, E_req, tau).

### 3.3 `train_log.csv`
columns: exp_id, env_steps, wall_clock_s, episode_return_eur_mean, episode_return_eur_std, frac_unserved_mean, tr_violation_kwh_mean, policy_loss, value_loss, entropy, approx_kl, clip_frac, beta_reg, proj_dist_mean, lr.

### 3.4 Oracle
`results/raw/oracle__{test_cfg_id}/episodes.csv` with the same columns (method = `oracle`), plus `solve_time_s`, `solver_status`, `shortfall_stage_kwh` (stage-1 objective).

## 4. Quality bar
- Unit tests required by the orchestrator prompt: projection satisfies bounds and aggregate limits on 10,000 random inputs (incl. tight A5 and |x| up to 1e8); agreement with projection_ref.py; feasibility invariance on synthetic episodes; masking: outputs for unmasked tokens invariant to padding/masked content; permutation equivariance of the actor and invariance of the critic; policy output shape correct for N ∈ {5, 10, 50, 100, 200}; wrapper smoke test with real EV2Gym (N=5, T=16) showing zero transformer violations and zero unserved EVs on A5-feasible steps under random raw actions; oracle LP solves a tiny instance and its profit ≥ every heuristic's profit on the same replay (sanity, smoke only).
- `make test` must pass in the cloud workspace with the venv at `/home/claude/work/ext/venv` (Python 3.13, torch 2.14 CPU) and should also work with Python 3.10-3.12 on the author's machine (do not use 3.13-only syntax).
- Deterministic seeding (numpy, torch, EV2Gym's seed), configurable device, no hard-coded absolute paths.
- Never write results into `results/raw/` from tests; tests use temp dirs.
