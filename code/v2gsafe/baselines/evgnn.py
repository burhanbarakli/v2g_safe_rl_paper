"""Optional evaluation adapter for EV-GNN [orfanoudakis2025scalable] policies (algorithm.md Alg. 5 row 8).

Loads a TD3_ActionGNN model trained with the public EV-GNN code (github.com/StavrosOrf/EV-GNN; NOT vendored,
the repository path is passed at run time) and exposes it as a controller of our evaluator. At every step the
EV-GNN state is built from the EV2Gym env with EV-GNN's own ``V2G_ProfitMax_with_Loads_GNN`` state function,
the deterministic actor gives normalised per-port EV2Gym actions a in [-1, 1], and we convert them to the raw
action x_i = a_i * P_cs * zeta_i (reported kW, as for eMPC). The raw variant ('evgnn') then uses the physical clip,
'evgnn_safe' the safety layer ("+Pi"). Requires torch_geometric (and EV-GNN's other imports) in the environment.
See ``evgnn_adapter.md`` for training instructions on our scenarios.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from ..constants import P_CS
from .heuristics import Controller


class EVGNNController(Controller):
    name = "evgnn"

    def __init__(self, repo: str | None = None, model_dir: str | None = None, model_name: str | None = None,
                 device: str = "cpu", policy=None):
        model_name = model_name or os.environ.get("EVGNN_MODEL_NAME", "model.best")
        repo = repo or os.environ.get("EVGNN_REPO")
        model_dir = model_dir or os.environ.get("EVGNN_MODEL_DIR")
        if repo is None:
            raise RuntimeError("set EVGNN_REPO to a clone of github.com/StavrosOrf/EV-GNN")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from utils.state import V2G_ProfitMax_with_Loads_GNN  # EV-GNN module (data, not vendored)
        self.state_fn = V2G_ProfitMax_with_Loads_GNN
        if policy is None:
            import yaml
            from TD3.TD3_ActionGNN import TD3_ActionGNN
            kw = yaml.load(open(Path(model_dir) / "kwargs.yaml"), Loader=yaml.FullLoader)
            kw["device"] = device
            kw["load_path"] = None
            policy = TD3_ActionGNN(**kw)
            policy.load(str(Path(model_dir) / model_name))
        self.policy = policy

    def act(self, obs, env):
        self.policy.action_dim = env.N                       # GNN actor is size-agnostic; output length = N
        state = self.state_fn(env.env)
        a = np.asarray(self.policy.select_action(state), float).reshape(-1)[:env.N]
        zeta = np.array([(s.params.zeta_c if a_i > 0 else s.params.zeta_d) if s else 1.0
                         for s, a_i in zip(env.slots, a)])
        return np.where(obs.occ, a * P_CS * zeta, 0.0)
