#!/usr/bin/env python
"""Run EV-GNN's train_RL_GNN.py unmodified under numpy >= 2 (DD27).

EV-GNN (commit in third_party/EVGNN_COMMIT) uses aliases removed in numpy 2.0 (``np.Inf``); we restore them at
runtime and execute the script with runpy, so EV-GNN's files stay untouched.
    python scripts/evgnn_launch.py <EV-GNN dir> [train_RL_GNN.py arguments ...]
"""
import os
import runpy
import sys

import numpy as np

for alias, value in (("Inf", np.inf), ("NaN", np.nan), ("float_", np.float64), ("infty", np.inf)):
    if not hasattr(np, alias):
        setattr(np, alias, value)

repo = os.path.abspath(sys.argv[1])
os.chdir(repo)
sys.path.insert(0, repo)
sys.argv = [os.path.join(repo, "train_RL_GNN.py")] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
