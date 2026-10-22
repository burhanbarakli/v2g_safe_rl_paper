# v2gsafe — code for "Attention-Based Safe RL for Scale-Agnostic and Constraint-Guaranteed V2G Charging Control"

This directory contains everything needed to run the experiments on your own machine: the safety layer
(certified bounds, robust projection, fallback, admission control, conformal departure bound), the attention
and MLP policies, PPO with a variable number of chargers, all baselines (heuristics with and without the safety
layer, EV2Gym's eMPC and a HiGHS port, the perfect-information oracle LP, an EV-GNN adapter), the evaluation
pipeline that writes the result files read by the analysis scripts, and a test suite.

Method documents: `../method/` (formulation.md, safety_layer.md, algorithm.md, design_decisions.md DD1-DD29).
Code-level choices: `IMPLEMENTATION_NOTES.md`. Output columns beyond code_spec Sec. 3 and all method names:
`SCHEMA_ADDITIONS.md`. Delivery summary: `DONE.md`.

**Nothing in `results/smoke/` may be reported**: smoke runs are tiny (N in {3, 5}, 16 steps, 2k env steps)
and are tagged `smoke = true` in every output file.

---

## 1. Install

Requirements: Linux or macOS, Python 3.10-3.13, git, about 10 GB free disk. A GPU is optional (training is
about 4x faster); evaluation, the oracle and conformal calibration are CPU-only.

### venv (recommended)

```bash
cd code
python3 -m venv ../venv && source ../venv/bin/activate
pip install -r requirements.txt            # exact versions tested (Python 3.13); on 3.10-3.12 use the next line
# pip install -e ".[dev,parquet]"          # version ranges from pyproject.toml
bash scripts/install_ev2gym.sh             # clones EV2Gym at the pinned commit into ../ext/EV2Gym, pip install -e
pip install --no-deps -e .                 # makes `v2gsafe` importable from anywhere (optional)
make test                                  # about 3.5 minutes on 2 CPU cores; all tests must pass
```

### conda

```bash
conda env create -f environment.yml && conda activate v2gsafe
bash scripts/install_ev2gym.sh && make test
```

### GPU

Install the CUDA build of PyTorch that matches your driver *after* the steps above, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu124`, and pass `--device cuda`
(`./reproduce.sh --device cuda`, `python train.py ... --device cuda`). Only the PPO networks use the GPU; the
16 environment processes run on CPU cores.

### EV2Gym

EV2Gym is used **unmodified** at commit `55ef368c25a58476a9eeaed5111f112a63a38ca8`
(`third_party/EV2GYM_COMMIT`). `meta.json` of every run records the installed commit and whether it matches.
EV2Gym imports pandapower, numba, multicopula, psutil, networkx and `pkg_resources` (setuptools < 81) without
declaring them; the install script adds them. Gurobi is optional (only for the original eMPC baseline).

---

## 2. Smoke test (about 8 minutes on 2 cores)

```bash
./reproduce.sh --smoke --jobs 2        # or: make smoke
```

Runs every stage on tiny settings (N in {3, 5}, T = 16, 2 replays per configuration, 2,048 env steps per learned
method) and writes to `results/smoke/`. Expected end of the log: `[pipeline] stage merge ... OK merge`; the
analysis stage is skipped if `../analysis/scripts/make_all.py` does not exist. A single learned method:

```bash
python train.py --config configs/train_attn_safe.yaml --overlay configs/smoke.yaml --seed 0 --smoke
```

---

## 3. Full pipeline

```bash
./reproduce.sh --jobs 4 --device cuda          # all stages except tuning
./reproduce.sh --tune --jobs 4 --device cuda   # with the hyperparameter tuning of Alg. 6 (recommended order:
                                               #   --stage scenarios,conformal,tune,tune_select, then the rest)
./reproduce.sh --stage train --list            # print the commands of the pending jobs only
```

Stages (`--stage a,b,c`; every stage is resumable, completed jobs are skipped):

| stage | what | output |
|---|---|---|
| scenarios | 37 test configurations (36 + literature), replay seeds and fingerprints | `results/test_sets/*.json` |
| conformal | split-conformal calibrations, 37 configurations x alpha in {0.01, 0.05, 0.1, 0.2} (+ training configs at 0.1) | `results/conformal/*.pkl` |
| tune / tune_select | 4 methods x 18 configurations x 0.5e6 steps, 20 validation episodes, selection | `results/tuning/<method>/selected.yaml` |
| train | 5 main methods x 5 seeds, 4 ablations x 5 seeds, penalty-weight sensitivity 2 x 3 seeds | `results/raw/<exp_id>/` |
| evaluate | every checkpoint and baseline on 3,100 test episodes | `results/raw/<exp_id>/episodes.csv` |
| oracle | lexicographic LP (+ rounding-aware variant) per test replay | `results/raw/oracle__<cfg>/episodes.csv` |
| merge | rebuild every `episodes.csv` from its part files | |
| analysis | `python ../analysis/scripts/make_all.py --raw results/raw --out ../analysis` (skipped if missing) | `../analysis/` |

`--jobs N` runs N jobs of a stage in parallel. A training job itself starts 16 environment processes, so use
`--jobs ≈ cores/16` for training (or `--jobs 1` with a GPU), and `--jobs ≈ cores` for evaluate/oracle/conformal.
The final training runs automatically use `results/tuning/<method>/selected.yaml` if it exists (otherwise the
Alg. 6 defaults). Admission control uses `safety.admission_existing_targets: lp_reachable` (DD25; the literal
S 7.4 variant `admitted` is kept as an option, IMPLEMENTATION_NOTES I-3).

Optional stages:

```bash
./reproduce.sh --stage sensitivity --jobs 4      # DD28, after 'train': SoC 0.8 + 5-min configurations (N = 20,
                                                 # rho 4, 3 profiles); trains the dt5 policy (5 seeds), evaluates
./reproduce.sh --with-evgnn ...                  # DD27: adds the EV-GNN stage (scripts/run_evgnn.sh) to 'all'
bash scripts/run_evgnn.sh --device cuda          # or run it directly: clone at the pinned commit
                                                 # (third_party/EVGNN_COMMIT), train one model per profile at
                                                 # N = 20, rho 4, 5 seeds, evaluate zero-shot (evgnn, evgnn_safe)
```

EV-GNN needs `torch_geometric`, `wandb` (offline), `tqdm`, `icecream`; the script installs them unless
`EVGNN_SKIP_INSTALL=1`. Do not install EV-GNN's own `requirements.txt` (it pins ev2gym 1.0.0 and torch 2.2).

### Single commands

```bash
python train.py --config configs/train_attn_safe.yaml --seed 0                      # -> results/raw/<exp_id>/
python train.py --config configs/train_attn_safe.yaml --seed 0 --resume             # continue after interruption
python evaluate.py --checkpoint results/raw/<exp_id>/checkpoint_final.pt --test-set configs/test_grid.yaml --jobs 8
python evaluate.py --checkpoint .../checkpoint_final.pt --test-set configs/test_grid.yaml --admission off
python evaluate.py --checkpoint <attn_safe_conformal run>/checkpoint_final.pt --test-set configs/test_grid.yaml \
                   --conformal-alpha 0.01 0.05 0.1 0.2
python evaluate.py --method llf_safe --test-set configs/test_grid.yaml --jobs 8      # baselines: afap alap rr llf
python evaluate.py --method empc_highs --test-set configs/test_grid.yaml             #   (+_safe), empc, empc_highs
python evaluate.py --method afap --test-set configs/test_grid.yaml --only workplace_N20_rho4 --log-steps
python -m v2gsafe.baselines.oracle --grid configs/test_grid.yaml --test-sets results/test_sets --out results/raw
python -m v2gsafe.safety.conformal --grid configs/test_grid.yaml --out results/conformal
```

Methods and exp_ids follow `code_spec.md` Sec. 2: `{method}__{train_or_test_cfg}__s{seed}__{config_hash8}`.
Extra evaluation variants: `attn_safe_noadm` (no admission control), `attn_safe_conf_a{alpha}` and
`attn_safe_conf_point` (uncalibrated), `empc_highs[_safe]` (HiGHS port), `evgnn[_safe]` (see
`v2gsafe/baselines/evgnn_adapter.md`). Sensitivity test configurations (desired SoC 0.8, 5-min steps) are in
`configs/test_grid_sensitivity.yaml` (evaluate with `--test-set configs/test_grid_sensitivity.yaml`); the 5-min
training config is `configs/ablations/train_attn_safe_dt5.yaml`.

---

## 4. How many runs, how long

Counts from algorithm.md Alg. 7.4; timings are the A2 estimates plus the indicative measurements below.

| item | count | env steps | estimate |
|---|---|---|---|
| main training (5 methods x 5 seeds) | 25 | 7.5e7 | |
| ablations (4 x 5 seeds) + penalty weights (2 x 3 seeds) | 26 | 7.8e7 | |
| SA-Attn-CP is in the main 5; EV-GNN is external (adapter) | | | |
| **training total** | **51 runs** (+5 EV-GNN = 56 in Alg. 7.4) | 1.5e8 | ~0.5 h/run on GPU, 2-3 h/run on a 16-core CPU (Alg. 7.4); **about 30 GPU-h or 120-180 CPU-h** |
| tuning (4 methods x 18 x 0.5e6) | 72 runs | 3.6e7 | about 1/6 of a run each, ~6 GPU-h |
| evaluation (81 learned evaluations + 12 baseline policies, 3,100 episodes each) | ~93 | | 0.5 CPU-h per policy (Alg. 7.4) + admission LPs; **~50-80 CPU-h**, parallel |
| oracle LPs | 3,100 | | 0.01-10 s each, ≤ 3 CPU-h |
| conformal calibrations (37 configs, 7,000 session draws each, 4 alphas) | 148 | | EV2Gym's session spawner costs ~4 ms per EV: ~1 h per configuration at N = 200, **~10-15 CPU-h total** (measured; Alg. 7.4 estimated < 1 CPU-h). Sessions are generated once for all alphas and cached in `results/conformal/sessions/`; the stage runs one job per configuration (`--jobs`), and `python -m v2gsafe.safety.conformal ... --workers K` splits one configuration over K processes |
| optional: sensitivity (DD28) | 5 dt5 runs + 6 test configs | 1.5e7 | ~3 GPU-h + a few CPU-h |
| optional: EV-GNN (DD27) | 3 profiles x 5 seeds | 4.5e7 | depends on EV-GNN's TD3 (GPU recommended) |

Indicative timings measured in the cloud workspace (2 CPU cores, no GPU, PyTorch 2.14, see `make bench`):

| N | wrapper observe (arrivals, bounds, tokens) | wrapper step (layer + actuation + EV2Gym) | observe with admission LP (mean) | actor forward, 1 thread | PPO update, 2048 transitions |
|---|---|---|---|---|---|
| 10 | 0.36 ms | 0.59 ms | 0.70 ms | 1.3 ms | 16.6 s (10 epochs, minibatch 512) |
| 50 | 1.08 ms | 1.37 ms | 5.5 ms | 2.1 ms | 9.4 s per epoch (minibatch 512) |
| 200 | 3.34 ms | 3.73 ms | 66 ms | 5.3 ms | 38.6 s per epoch with minibatch 64; minibatch 512 ran out of memory on the 7 GB machine |

Means over one 112-step workplace episode with random actions (many early steps have no EV). Training uses
N in {10, 20}: on this 2-core machine one update (128 rollout steps of 16 envs + 10 epochs) takes about 18 s,
i.e. **about 7 h per 3e6-step run on 2 cores**; with a GPU and >= 16 cores expect the 0.5 h of Alg. 7.4.
The admission LP dominates the controller time at N = 200 at arrival steps (p95 of `inference_ms_per_step`).
Smoke pipeline (`./reproduce.sh --smoke --jobs 2`): 7.5 minutes on 2 cores. `make test`: about 3.5 minutes.

## 5. Where outputs go

```
results/
  test_sets/<test_cfg_id>.json         replay seeds + fingerprints (common random numbers for all methods)
  conformal/<cfg_id>_a<alpha>.pkl      calibrations
  tuning/<method>/<tag>/ ...           tuning runs, validation.json; <method>/selected.yaml
  raw/<exp_id>/                        one directory per run (code_spec Sec. 2):
      config.yaml meta.json train_log.csv checkpoint_last.pt checkpoint_final.pt
      episodes.csv                     one row per evaluated episode (code_spec Sec. 3.1 + extra columns)
      parts/                           per-configuration part files (episodes.csv is rebuilt from them)
      steps.parquet steps_chargers.parquet   only with --log-steps
  raw/oracle__<test_cfg_id>/episodes.csv     oracle results (code_spec Sec. 3.4)
  logs/<stage>/<job>.log               stdout/stderr of every pipeline job
  smoke/...                            the same structure for smoke runs (never reported)
```

Disk: ~31 MB per attention run (two checkpoints), ~4-5 GB for all training and tuning runs, ~0.5 GB of
episode files, < 1 GB of calibrations and logs. Use `--log-steps` only for the example-day figure (~50 MB per
policy and configuration).

## 6. Resuming

Re-run the same command. `train.py` resumes from `checkpoint_last.pt` (saved every 50 updates; the episodes in
progress at the interruption are discarded, RNG streams are restored, so a resumed run is valid but not bitwise
identical to an uninterrupted one). `evaluate.py` skips configurations whose part file is complete. The
pipeline skips finished training runs, calibrations and oracle configurations.

## 7. Several machines

Split any stage with `--shard i/M` (job k runs on machine k mod M), e.g. training on 3 machines:

```bash
./reproduce.sh --stage scenarios,conformal          # once, then copy results/test_sets and results/conformal
machine0$ ./reproduce.sh --stage train --shard 0/3 --device cuda
machine1$ ./reproduce.sh --stage train --shard 1/3 --device cuda
machine2$ ./reproduce.sh --stage train --shard 2/3 --device cuda
```

Copy all `results/raw/*` directories to one machine, then run `--stage evaluate` (also shardable), `oracle`,
`merge`, `analysis`. Replay fingerprints in `episodes.csv` (`replay_fingerprint`; `MISMATCH:...` if a machine
generated a different episode) make cross-machine inconsistencies visible.

## 8. Troubleshooting

* **Gurobi**: optional. Without `gurobipy` the `empc` jobs are skipped; with the pip licence (size-limited,
  2000 variables) EV2Gym's eMPC runs only for N ≤ 25 (the pipeline restricts it accordingly). `empc_highs` is
  the open-source port for all N.
* **`pkg_resources is deprecated` warnings**: harmless (EV2Gym); install `setuptools<81`.
* **EV2Gym hangs at reset**: EV2Gym re-draws sessions until at least one EV fits the horizon; a configuration in
  which no session can fit (e.g. the private profile with a 4-h horizon) loops forever. Only the smoke settings
  are close to this; they use the public/workplace profiles.
* **Slow conformal stage**: dominated by EV2Gym's session spawner; run it with `--jobs` = number of cores.
* **Out of memory with many jobs**: each training job holds 16 EV2Gym processes (~250 MB each).
* **sklearn HistGradientBoosting very slow**: OpenMP contention with PyTorch; the default predictor is
  `GradientBoostingRegressor` (no OpenMP).

## 9. What to send back

From `results/` (paths relative to `code/`):

* `raw/**/episodes.csv` (all methods, the oracle, all variants) — required
* `raw/**/train_log.csv`, `raw/**/meta.json`, `raw/**/config.yaml` — required
* `logs/` (zipped) and `tuning/*/selected.yaml`, `tuning/*/*/validation.json` — required if tuning was run
* `test_sets/*.json` — required (replay definitions and fingerprints)
* `raw/**/steps*.parquet` — only for the example-day figure
* `raw/**/checkpoint_final.pt` — optional (≈ 15 MB each), useful for re-evaluation
* the output of `make test` and `python scripts/benchmark.py` on your machine

## 10. Code map

```
v2gsafe/constants.py       constants (eps_num, eps_q, P_cs, normalisation)       nomenclature.md
v2gsafe/config.py          YAML config, inheritance, Alg. 6 defaults, config hash
v2gsafe/specs/             constant-efficiency EV spec JSON (DD4) + make script v2gsafe/scripts/make_ev_specs.py
v2gsafe/envs/scenarios.py  scenarios, profiles, seeds, test grid, replay manifests   Alg. 7, DD24
v2gsafe/envs/wrapper.py    SafeV2GEnv: arrivals, bounds, observation, layer, actuation, metrics   Alg. 1, 3
v2gsafe/envs/forecasts.py  noisy load/PV forecasts   DD6
v2gsafe/envs/vec_env.py    16 parallel envs (sync / subprocess)
v2gsafe/safety/            bounds.py (S2-S12), projection.py (S13-S16, NumPy+PyTorch), fallback.py (S 7.2),
                           actuation.py (S 6.3), admission.py (S 7.4), conformal.py (S 8, Alg. 4), lp.py (HiGHS)
v2gsafe/models/            attention_policy.py (F 5), mlp_policy.py (Alg. 5)
v2gsafe/algos/             ppo.py (Alg. 2), buffers.py
v2gsafe/baselines/         heuristics.py, mpc.py, oracle.py (F 3), evgnn.py + evgnn_adapter.md
v2gsafe/eval/              evaluate.py, metrics.py (episodes.csv schema), merge.py
v2gsafe/tuning.py          Alg. 6 tuning protocol;  v2gsafe/pipeline.py  reproduce.sh driver
train.py evaluate.py       CLIs;  scripts/benchmark.py  timings;  tests/  pytest (reference/projection_ref.py = oracle)
```
