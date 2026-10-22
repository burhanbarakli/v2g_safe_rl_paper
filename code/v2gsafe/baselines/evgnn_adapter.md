# EV-GNN / GNN-DT on our scenarios (algorithm.md Alg. 5 row 8; optional baseline)

Status (2026-10-22, checked by A3 in the cloud workspace):

* **EV-GNN** (github.com/StavrosOrf/EV-GNN, commit `01aa1b7a676cca7aa077be453b28e29ea20344dd`, 2025-10-16,
  MIT licence) was cloned and inspected. Its state function `utils/state.py:V2G_ProfitMax_with_Loads_GNN` and its
  `TD3/TD3_ActionGNN.py` policy **run on our pinned EV2Gym commit** with torch 2.14 and torch_geometric 2.8.1
  (inference with an untrained policy at N = 10 and zero-shot at N = 50 through our adapter, 0 errors).
  Training was NOT run here (no GPU; 1e7 default steps). Its code is not vendored; you clone it yourself.
* **GNN-DT** [orfanoudakis2026graph]: no public repository could be located from this workspace
  (`github.com/StavrosOrf/GNN-DT` and variants need authentication = not public / not existing; web search is
  disabled here). If the code becomes available, the same adapter pattern applies (state function + `select_action`).

## 1. Install (separate venv recommended)

EV-GNN's `requirements.txt` pins `ev2gym==1.0.0` and `torch==2.2.2`. **Do not install those pins**: use our
venv (pinned EV2Gym commit, our torch) and add only what EV-GNN imports:

```bash
git clone https://github.com/StavrosOrf/EV-GNN.git ../ext/EV-GNN      # any path; pass it below
pip install torch_geometric==2.8.1 wandb tqdm icecream                  # wandb is imported unconditionally
export WANDB_MODE=offline                                              # train_RL_GNN.py logs to wandb by default
```

## 2. Training on our scenarios (same budget as our learned methods)

EV-GNN's `train_RL_GNN.py` chooses reward and state by the **config file name**: a name containing
`V2G_ProfixMaxWithLoads` selects EV-GNN's default V2G reward `ProfitMax_TrPenalty_UserIncentives` (penalty-based,
"its own default reward", Alg. 5) and the state `V2G_ProfitMax_with_Loads_GNN`. Generate our EV2Gym YAML with
that prefix inside EV-GNN's `config_files/` (it reads `./config_files/<name>`, so run it from the EV-GNN folder):

```bash
cd ../ext/EV-GNN
python -m v2gsafe.envs.scenarios write-yaml --profile workplace --N 20 --rho 4 \
       --out config_files/V2G_ProfixMaxWithLoads_ours_workplace_N20_rho4.yaml     # PYTHONPATH=<code dir>
python train_RL_GNN.py --policy TD3_ActionGNN --config V2G_ProfixMaxWithLoads_ours_workplace_N20_rho4.yaml \
       --max_timesteps 3000000 --seed 0 --device cuda --save_model --name ours_N20
# -> saved_models/ours_N20-XXXXXX/{kwargs.yaml, model.best_actor, model.best_critic, ...}
```

Differences to our training protocol that the paper must state (ORCHESTRATOR DECISION, IMPLEMENTATION_NOTES I-8):

1. EV-GNN's script trains on **one** configuration per run (no per-episode mixture of N in {10, 20}, profiles and
   rho_tr). Options: (a) train one run per seed on `workplace_N20_rho4` only; (b) train sequentially on several
   YAMLs with `--load_model`; (c) write a mixture driver (not done here). Budget: 3e6 env steps per run, 5 seeds.
2. TD3 (off-policy), its own hyperparameters (tuning grid of Alg. 6 not applied unless you run it).
3. Our EV2Gym YAML uses `tr_seed: -1`; EV2Gym then keeps the loads/PV draw of the constructor seed for all
   episodes of a run (our wrapper re-seeds per episode; EV-GNN's training env does not).
4. Its action is the normalised EV2Gym current; we convert it to reported kW (x = a P_cs zeta).

## 3. Evaluation on our fixed test replays

```bash
python evaluate.py --method evgnn      --test-set configs/test_grid.yaml --evgnn-repo ../ext/EV-GNN \
       --evgnn-model-dir ../ext/EV-GNN/saved_models/ours_N20-XXXXXX/ --out-root results/raw
python evaluate.py --method evgnn_safe --test-set configs/test_grid.yaml --evgnn-repo ../ext/EV-GNN \
       --evgnn-model-dir ../ext/EV-GNN/saved_models/ours_N20-XXXXXX/ --out-root results/raw
```

`evgnn` = raw (physical clip only), `evgnn_safe` = "+Pi" (projection, fallback, admission control). The adapter
(`v2gsafe/baselines/evgnn.py`) sets the policy's `action_dim` to the test station size at every step, so the
same weights run zero-shot at N in {5, ..., 200}. For several seeds, run one evaluation per model directory; the
model directory enters the exp_id hash, so the runs land in different directories.
