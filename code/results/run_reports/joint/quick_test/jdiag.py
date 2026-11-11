"""Why do A5 failures remain under the joint bound? First infeasible step per episode."""
import sys, glob, numpy as np
import v2gsafe.envs.wrapper as W
from v2gsafe.eval.evaluate import MethodSpec, resolve
from v2gsafe.envs.scenarios import ScenarioConfig, load_test_grid, build_test_set
from v2gsafe.constants import H_F

rec = {}
orig = W.joint_lower_bound
def wrapped(E, tau, B, pnom, l_dep, u, C_hat, Ctil, dt_h):
    l = orig(E, tau, B, pnom, l_dep, u, C_hat, Ctil, dt_h)
    rec["shrink"] = bool(np.sum(l) < np.sum(np.minimum(u, np.maximum(l_dep, l))) - 1e-9 or
                         np.sum(np.minimum(u, np.maximum(l_dep, l))) > Ctil)
    rec["C_hat1"] = float(C_hat[0]); rec["tau"] = np.array(tau)
    return l
W.joint_lower_bound = wrapped

ck = sys.argv[1]; cfgname = sys.argv[2]; nrep = int(sys.argv[3])
grid = {s.cfg_id: s for s, n in load_test_grid("configs/test_grid.yaml")}
scfg = grid[cfgname]
spec = MethodSpec(method="learned" if ck != "-" else "price_llf_safe", checkpoint=None if ck == "-" else ck,
                  admission_cache="results/admission")
ctrl, name, opts, info = resolve(spec)
opts.admission_cache = "results/admission"; opts.lower_bound = "joint"
env = W.SafeV2GEnv(scfg, opts)
man = build_test_set(scfg, nrep, "results/test_sets")
for rep in man["replays"][:nrep]:
    obs = env.reset(rep["seed"], scfg); ctrl.reset(env)
    hist = []
    while True:
        occ = obs.occ
        hist.append(dict(t=obs.t, n=int(occ.sum()), Ctil=obs.Ctil, sl=float(obs.l_dep[occ].sum()),
                         slj=float(obs.l[occ].sum()), feas=obs.feasible, shrink=rec.get("shrink", False),
                         Ch1=rec.get("C_hat1", np.nan), arrivals=set(np.flatnonzero(occ))))
        out = env.step(ctrl.act(obs, env))
        if out.done: break
        obs = env.observe()
    bad = [h for h in hist if not h["feas"]]
    if not bad: continue
    t0 = bad[0]["t"]
    prev = [h for h in hist if t0 - 6 <= h["t"] <= t0]
    print(f"replay {rep['replay_id']}: {len(bad)} infeasible steps, first t={t0}")
    for i, h in enumerate(prev):
        new = h["arrivals"] - (prev[i-1]["arrivals"] if i else set())
        print(f"  t={h['t']:3d} n={h['n']} Ctil={h['Ctil']:6.2f} Chat(prev->t)={prev[i-1]['Ch1'] if i else float('nan'):6.2f}"
              f" sum_ldep={h['sl']:6.2f} sum_lj={h['slj']:6.2f} shrink={h['shrink']} new={sorted(new)}")
