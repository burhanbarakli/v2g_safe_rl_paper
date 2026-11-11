"""Quick test-time check of the joint lower bound: same replays, lower_bound True vs 'joint'."""
import sys, json, multiprocessing as mp, glob
import numpy as np
from v2gsafe.eval.evaluate import MethodSpec, resolve, run_episode
from v2gsafe.envs.scenarios import load_test_grid, build_test_set
from v2gsafe.envs.wrapper import SafeV2GEnv

def job(a):
    meth, ck, cfgd, nrep, lb, jf = a
    from v2gsafe.envs.scenarios import ScenarioConfig
    scfg = ScenarioConfig.from_dict(cfgd)
    spec = MethodSpec(method=meth, checkpoint=ck, admission_cache="results/admission", device="cpu", torch_threads=1)
    ctrl, name, opts, info = resolve(spec)
    opts.admission_cache = "results/admission"
    opts.lower_bound = lb
    opts.joint_features = jf
    import os
    opts.joint_margin = float(os.environ.get("JM", 0)); opts.joint_cap_frac = float(os.environ.get("JC", 0))
    man = build_test_set(scfg, nrep, "results/test_sets")
    env = SafeV2GEnv(scfg, opts)
    out = []
    for rep in man["replays"][:nrep]:
        m = run_episode(env, ctrl, rep["seed"], scfg)
        jp = m["profit_eur"] - 10 * (m["energy_not_served_shared_kwh"] + m["tr_violation_kwh"])
        out.append(dict(cfg=scfg.cfg_id, profile=scfg.profile, N=scfg.N, rho=scfg.rho, rep=rep["replay_id"],
                        jpen=jp / scfg.N, profit=m["profit_eur"] / scfg.N,
                        ens=m["energy_not_served_shared_kwh"] / scfg.N, inf=int(m["infeasible_steps"] > 0)))
    return out

if __name__ == "__main__":
    meth, ck, nrep, lb, jf, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] == "1", sys.argv[6]
    ck = None if ck == "-" else ck
    lb = "joint" if lb == "joint" else True
    import torch; torch.set_num_threads(1)
    grid = [s for s, n in load_test_grid("configs/test_grid.yaml")
            if s.dt_min == 15 and s.rho in (4.0, 7.0) and s.N in (5, 10, 20, 50) and getattr(s, "name", None) is None
            and not getattr(s, "throughput_cost", 0)]
    jobs = [(meth, ck, s.to_dict(), nrep, lb, jf) for s in grid]
    with mp.get_context("spawn").Pool(11) as p:
        rows = [r for rr in p.map(job, jobs) for r in rr]
    json.dump(rows, open(outp, "w"))
    print(outp, len(rows), "jpen", np.mean([r["jpen"] for r in rows]), "inf", np.mean([r["inf"] for r in rows]))
