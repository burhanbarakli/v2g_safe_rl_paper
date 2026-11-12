"""|mu| of residual policies on a few test episodes (deviation from the rule in logit units)."""
import sys, glob
import numpy as np, torch
from v2gsafe.eval.evaluate import MethodSpec, resolve
from v2gsafe.envs.scenarios import load_test_grid, build_test_set
from v2gsafe.envs.wrapper import SafeV2GEnv
torch.set_num_threads(2)
grid = [s for s, n in load_test_grid("configs/test_grid.yaml") if s.dt_min == 15 and s.rho in (4.0, 7.0)
        and s.N in (10, 50) and getattr(s, "name", None) is None and not getattr(s, "throughput_cost", 0)]
for ck in sys.argv[1:]:
    spec = MethodSpec(method="x", checkpoint=ck, admission_cache="results/admission", device="cpu", torch_threads=2)
    ctrl, name, opts, info = resolve(spec)
    actor = ctrl.actor if hasattr(ctrl, "actor") else None
    mus, stds = [], []
    for scfg in grid:
        man = build_test_set(scfg, 2, "results/test_sets")
        env = SafeV2GEnv(scfg, opts)
        for rep in man["replays"][:2]:
            obs = env.reset(rep["seed"], scfg)
            if hasattr(ctrl, "reset"): ctrl.reset(env)
            while True:
                with torch.no_grad():
                    mu, ls = actor(torch.as_tensor(obs.tokens)[None], torch.as_tensor(obs.glob)[None],
                                   torch.as_tensor(obs.occ)[None])
                m = obs.occ & (obs.u - obs.l > 1e-6)
                mus += list(mu[0].numpy()[m]); stds += list(np.exp(ls[0].numpy()[m]) if ls.ndim == 2 else [float(np.exp(ls.mean()))] * int(m.sum()))
                out = env.step(ctrl.act(obs, env))
                if out.done: break
                obs = env.observe()
    mus = np.abs(np.array(mus))
    print(name, "n", mus.size, "|mu| mean %.2f median %.2f p90 %.2f" % (mus.mean(), np.median(mus), np.quantile(mus, .9)),
          "std %.2f" % np.mean(stds))
