# GNN-DT baseline: adapter instructions (DD35, optional, not implemented)

GNN-DT [orfanoudakis2025gnndt] (Graph Neural Network enhanced Decision Transformer) is an *offline* RL method:
a decision transformer with GCN state/action embeddings, trained on stored trajectories of EV2Gym episodes and
conditioned on a return-to-go at run time. Code: github.com/StavrosOrf/DT4EVs, inspected at the pinned commit
`8b327d122efe13678df1dc74c42c673ce33541d9` (2026-01-19, `third_party/GNNDT_COMMIT`). The code is not vendored;
`bash scripts/run_gnndt.sh --clone` clones it at that commit. `./reproduce.sh --with-gnndt` runs only this stub
(it prints a pointer to this file and writes no result rows).

## Why only instructions (cost estimate)

* The repository ships its **own modified copy of EV2Gym** (`DT4EVs/ev2gym/`: extra Gurobi models
  `PST_V2G_profit_max*.py`, a different `heuristics.py`, `ev_specs_v2g_enabled2024.json` and price CSV). Its
  state/reward functions (`utils.py`: `PST_V2G_ProfitMax_state`, `PST_V2G_ProfitMaxGNN_state`,
  `PST_V2G_ProfitMax_reward`) and its model loader import from that copy, not from our pinned EV2Gym
  (55ef368c). Running it on our scenarios requires porting the state functions to our EV2Gym and our constant
  efficiency spec file (DD4).
* Trajectory generation for the best-performing datasets ("optimal", "mixed_bau_*") calls the Gurobi oracle
  `mo_PST_V2GProfitMaxOracleGB` (`generate_trajectories.py`); a Gurobi licence is required, or the dataset must be
  generated from our HiGHS oracle plan / eMPC instead (a deviation from the paper's protocol).
* Pinned requirements (`req.txt`: numpy 1.21, pandas 1.3, transformers 4.6.1, tokenizers 0.10.3) do not install
  on Python 3.12/3.13; the model code imports `transformers` (GPT2) and `torch_geometric`.
* Effort estimate: 2-4 person-days for the port plus dataset generation (about 1-2 CPU-days for 10^3-10^4
  trajectories per profile at N = 20) and DT training (about 0.5-1 GPU-day per profile and seed).

## Recipe (if the baseline is wanted)

1. `bash scripts/run_gnndt.sh --clone`; create a separate venv with Python 3.10, `pip install torch
   torch_geometric transformers==4.* gurobipy` (do not install their `req.txt` pins on top of ours).
2. Port `PST_V2G_ProfitMaxGNN_state` / `PST_V2G_ProfitMax_state_to_GNN` (DT4EVs `utils.py`) to read our
   EV2Gym env (one port per charger, `fx_node_sizes = {ev: 5, cs: 4, tr: 2, env: 6}` as in `DT/load_model.py`).
3. Write the training configurations with our scenario writer (as EV-GNN, DD27):
   `python -m v2gsafe.envs.scenarios write-yaml --profile <p> --N 20 --rho 4 --out <DT4EVs>/config_files/ours_<p>.yaml`.
4. Generate trajectories on **training seeds** (`SEED_RANGES['train']`, never test replays) with
   `generate_trajectories.py --dataset optimal|mixed_bau_50 --n_trajectories ...` (their `--env` switch only knows
   "25"/"250": pass `--config_file` through a small launcher as `scripts/evgnn_launch.py` does for EV-GNN).
5. Train `train_DT.py --model_type gnn_act_emb --K 3 --embed_dim 128 --n_layer 3 --n_head 1` (their defaults) per
   profile and seed; keep `saved_models/<run>/{vars.yaml, model.best}`.
6. Adapter: a `Controller` like `v2gsafe/baselines/evgnn.py` that keeps the last K (state, action, return-to-go)
   tuples, calls `model.get_action(...)` (see `DT/evaluation/evaluate_episodes.py::evaluate_episode_rtg`, target
   return and `scale` from their evaluation script), maps the normalised per-port action in [-1, 1] to
   x_i = a_i * P_cs * zeta_i (reported kW) and updates the return-to-go with OUR reward r(t)/scale. Register
   `gnndt` / `gnndt_safe` in `v2gsafe/eval/evaluate.py:resolve` and set `info["train_N"] = 20` so that
   `in_distribution` is true on N = 20, rho_tr = 4 and false elsewhere (as for EV-GNN).
7. Evaluate on the test grid with `--admission-cache results/admission` (same admitted requests, DD31).
