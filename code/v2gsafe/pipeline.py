"""End-to-end pipeline driver used by ``reproduce.sh`` (job lists, resume, parallel jobs, machine sharding).

Stages: scenarios -> conformal -> tune (optional) -> train -> evaluate -> oracle -> merge -> analysis.
Every job is a subprocess command; completed jobs are skipped (resume = re-run the same command). With
``--shard i/M`` only every M-th job (offset i) of a stage is run, so several machines can split a stage
(e.g. training seeds/methods); copy results/raw/** back to one machine before 'merge' and 'analysis'.

    python -m v2gsafe.pipeline --stage train --jobs 4 --device cuda
    python -m v2gsafe.pipeline --stage all --smoke --jobs 2
    python -m v2gsafe.pipeline --stage train --list            # print the commands only
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ALPHAS = [0.01, 0.05, 0.1, 0.2]

MAIN_CONFIGS = ["configs/train_attn_safe.yaml", "configs/train_attn_penalty.yaml", "configs/train_mlp_safe.yaml",
                "configs/train_mlp_penalty.yaml", "configs/train_attn_safe_conformal.yaml"]
ABLATION_CONFIGS = ["configs/ablations/train_attn_safe_beta0.yaml", "configs/ablations/train_attn_safe_curriculum.yaml",
                    "configs/ablations/train_attn_safe_small.yaml", "configs/ablations/train_attn_penalty_dense.yaml"]
SENS_CONFIGS = ["configs/ablations/train_attn_penalty_w1.yaml", "configs/ablations/train_attn_penalty_w100.yaml"]
TUNED_METHODS = {"attn_safe": "attn_safe", "attn_penalty": "attn_penalty", "mlp_safe": "mlp_safe",
                 "mlp_penalty": "mlp_penalty", "attn_safe_conformal": "attn_safe", "attn_safe_beta0": "attn_safe",
                 "attn_safe_curriculum": "attn_safe", "attn_safe_small": "attn_safe",
                 "attn_penalty_dense": "attn_penalty", "attn_penalty_w1": "attn_penalty",
                 "attn_penalty_w100": "attn_penalty", "attn_safe_dt5": "attn_safe"}
HEURISTIC_METHODS = ["afap", "alap", "rr", "llf"]


@dataclass
class Job:
    name: str
    cmd: list[str]
    done: callable = field(default=lambda: False, repr=False)


@dataclass
class Ctx:
    smoke: bool
    device: str
    seeds: list[int]
    sens_seeds: list[int]
    results: Path
    grid: str
    overlay: list[str]
    ablations: bool
    tune: bool
    empc_max_N: int


def make_ctx(a) -> Ctx:
    smoke = a.smoke
    res = ROOT / ("results/smoke" if smoke else "results")
    return Ctx(smoke=smoke, device=a.device, seeds=[0] if smoke else list(range(a.seeds)),
               sens_seeds=[0] if smoke else list(range(3)), results=res,
               grid="configs/test_grid_smoke.yaml" if smoke else "configs/test_grid.yaml",
               overlay=["configs/smoke.yaml"] if smoke else [], ablations=(not smoke) and not a.no_ablations,
               tune=a.tune, empc_max_N=25)


def _smoke_flags(c: Ctx) -> list[str]:
    return ["--smoke"] if c.smoke else []


# ------------------------------------------------------------------------------------------ stages
def stage_scenarios(c: Ctx) -> list[Job]:
    cmd = [PY, "-m", "v2gsafe.envs.scenarios", "build-test-sets", "--grid", c.grid, "--out",
           str(c.results / "test_sets")]
    return [Job("scenarios", cmd)]


def _grid_ids(c: Ctx, grid: str | None = None, dt: int | None = None) -> list[tuple[str, int]]:
    from v2gsafe.envs.scenarios import load_test_grid
    return [(s.cfg_id, s.N) for s, _ in load_test_grid(ROOT / (grid or c.grid)) if dt is None or s.dt_min == dt]


def stage_conformal(c: Ctx) -> list[Job]:
    from v2gsafe.config import load_config
    from v2gsafe.envs.scenarios import TrainMixture
    from v2gsafe.safety.conformal import calibration_path
    conf_dir = c.results / "conformal"
    extra = []
    if c.smoke:
        extra = ["--n-train", "20", "--n-cal", "50"]
    jobs = []
    for cid, _ in _grid_ids(c):
        jobs.append(Job(f"conformal_{cid}", [PY, "-m", "v2gsafe.safety.conformal", "--grid", c.grid, "--only", cid,
                                             "--alpha", *map(str, ALPHAS), "--out", str(conf_dir), *extra],
                        done=lambda cid=cid: all(calibration_path(conf_dir, cid, a).exists() for a in ALPHAS)))
    # training configurations of SA-Attn-CP (alpha of the training config)
    cfg = load_config(ROOT / "configs/train_attn_safe_conformal.yaml", overlays=[ROOT / o for o in c.overlay])
    from v2gsafe.algos.ppo import train_mixture
    mix = train_mixture(cfg)
    Ns = sorted({int(g["N"]) for g in cfg["envs"]["groups"]})
    alpha = cfg["conformal"]["alpha"]
    for scfg in mix.all_configs(Ns):
        cid = scfg.cfg_id
        jobs.append(Job(f"conformal_train_{cid}", [PY, "-m", "v2gsafe.safety.conformal", "--scenario-json",
                                                   json.dumps(scfg.to_dict()), "--alpha", str(alpha),
                                                   "--out", str(conf_dir), *extra],
                        done=lambda cid=cid: calibration_path(conf_dir, cid, alpha).exists()))
    return jobs


def _selected_overlay(c: Ctx, method: str) -> list[str]:
    base = TUNED_METHODS.get(method)
    if base is None:
        return []
    p = c.results / "tuning" / base / "selected.yaml"
    return [str(p)] if p.exists() else []


def _train_job(c: Ctx, config: str, seed: int, extra_overlay: list[str] | None = None,
               sets: list[str] | None = None, out: str | None = None) -> Job:
    from v2gsafe.config import config_hash8, exp_id, load_config
    overlays = [str(ROOT / o) for o in c.overlay] + list(extra_overlay or [])
    cfg = load_config(ROOT / config, list(sets or []), overlays)
    if out is None:
        overlays += _selected_overlay(c, cfg["method"])
        cfg = load_config(ROOT / config, list(sets or []), overlays)
        out = str(c.results / "raw" / exp_id(cfg["method"], cfg["_name"], seed, config_hash8(cfg)))
    cmd = [PY, "train.py", "--config", config, "--seed", str(seed), "--out", out, "--device", c.device,
           *_smoke_flags(c)]
    for o in overlays:
        cmd += ["--overlay", o]
    for s in sets or []:
        cmd += ["--set", s]
    return Job(f"train_{Path(out).name}", cmd, done=lambda out=out: (Path(out) / "checkpoint_final.pt").exists())


def stage_tune(c: Ctx) -> list[Job]:
    from v2gsafe import tuning
    jobs = []
    for config in MAIN_CONFIGS[:4]:
        for j in tuning.jobs(config, str(c.results / "tuning"), smoke=c.smoke):
            jobs.append(_train_job(c, config, 0, sets=j["sets"], out=j["out"]))
    return jobs


def stage_tune_select(c: Ctx) -> list[Job]:
    jobs = []
    tdir = c.results / "tuning"
    for run in sorted(tdir.glob("*/*/checkpoint_final.pt")):
        rd = run.parent
        jobs.append(Job(f"validate_{rd.parent.name}_{rd.name}", [PY, "-m", "v2gsafe.tuning", "validate", "--run",
                                                                  str(rd)] + (["--n", "2"] if c.smoke else []),
                        done=lambda rd=rd: (rd / "validation.json").exists()))
    return jobs


def stage_train(c: Ctx) -> list[Job]:
    jobs = [_train_job(c, cfg, s) for cfg in MAIN_CONFIGS for s in c.seeds]
    if c.ablations:
        jobs += [_train_job(c, cfg, s) for cfg in ABLATION_CONFIGS for s in c.seeds]
        jobs += [_train_job(c, cfg, s) for cfg in SENS_CONFIGS for s in c.sens_seeds]
    return jobs


def _eval_cmd(c: Ctx, extra: list[str], cids: list[str], grid: str | None = None) -> list[str]:
    return [PY, "evaluate.py", "--test-set", grid or c.grid, "--only", *cids, "--device", c.device,
            "--test-sets-dir", str(c.results / "test_sets"), "--conformal-cache", str(c.results / "conformal"),
            "--out-root", str(c.results / "raw"), *_smoke_flags(c), *extra]


def _groups_by_N(ids, max_N: int | None = None) -> dict[int, list[str]]:
    """Test configurations grouped by station size (one EV2Gym env per job is reused across the group)."""
    out: dict[int, list[str]] = {}
    for cid, N in ids:
        if max_N is None or N <= max_N:
            out.setdefault(N, []).append(cid)
    return out


def stage_evaluate(c: Ctx) -> list[Job]:
    jobs = []
    groups = _groups_by_N(_grid_ids(c))
    try:
        import gurobipy  # noqa: F401
        has_gurobi = True
    except ImportError:
        has_gurobi = False
    baselines = [m for h in HEURISTIC_METHODS for m in (h, h + "_safe")] + ["empc_highs", "empc_highs_safe"]
    for m in baselines:
        for N, cids in groups.items():
            jobs.append(Job(f"eval_{m}_N{N}", _eval_cmd(c, ["--method", m], cids)))
    if has_gurobi:
        for m in ("empc", "empc_safe"):
            for N, cids in _groups_by_N(_grid_ids(c), c.empc_max_N).items():
                jobs.append(Job(f"eval_{m}_N{N}", _eval_cmd(c, ["--method", m], cids)))
    for ck in sorted((c.results / "raw").glob("*/checkpoint_final.pt")):
        name = ck.parent.name
        method = name.split("__")[0]
        variants = [[]]
        if method == "attn_safe":
            variants.append(["--admission", "off"])
        if method.endswith("_conformal"):
            variants = [["--conformal-alpha", str(a)] for a in ALPHAS] + [["--conformal-uncalibrated"]]
        for v in variants:
            for N, cids in groups.items():
                jobs.append(Job(f"eval_{name}_{'_'.join(v)}_N{N}",
                                _eval_cmd(c, ["--checkpoint", str(ck), *v], cids)))
    return jobs


def stage_oracle(c: Ctx, grid: str | None = None) -> list[Job]:
    jobs = []
    for cid, _ in _grid_ids(c, grid):
        out = c.results / "raw" / f"oracle__{cid}" / "episodes.csv"
        jobs.append(Job(f"oracle_{cid}", [PY, "-m", "v2gsafe.baselines.oracle", "--grid", grid or c.grid, "--only", cid,
                                          "--test-sets", str(c.results / "test_sets"), "--out", str(c.results / "raw"),
                                          *_smoke_flags(c)], done=lambda out=out: out.exists()))
    return jobs


# ------------------------------------------------------------------- optional stages (DD27, DD28)
SENS_TRAIN = "configs/ablations/train_attn_safe_dt5.yaml"
SENS_MAIN_METHODS = ("attn_safe", "attn_penalty", "mlp_safe", "mlp_penalty")


def _sens_grid(c: Ctx) -> str:
    return "configs/test_grid_sensitivity_smoke.yaml" if c.smoke else "configs/test_grid_sensitivity.yaml"


def stage_sens_scenarios(c: Ctx) -> list[Job]:
    return [Job("sens_scenarios", [PY, "-m", "v2gsafe.envs.scenarios", "build-test-sets", "--grid", _sens_grid(c),
                                   "--out", str(c.results / "test_sets")])]


def stage_sens_train(c: Ctx) -> list[Job]:
    sets = ["scenario.T=48"] if c.smoke else []          # 5-min smoke horizon: 4 h (T = 16 fits no session)
    return [_train_job(c, SENS_TRAIN, s, sets=sets) for s in c.seeds]


def stage_sens_evaluate(c: Ctx) -> list[Job]:
    """SoC-0.8 configurations: main policies + baselines; 5-min configurations: dt5 policy + baselines (DD28)."""
    g = _sens_grid(c)
    soc = _groups_by_N(_grid_ids(c, g, dt=15))
    dt5 = _groups_by_N(_grid_ids(c, g, dt=5))
    jobs = []
    baselines = [m for h in HEURISTIC_METHODS for m in (h, h + "_safe")] + ["empc_highs", "empc_highs_safe"]
    for groups, tag in ((soc, "soc"), (dt5, "dt5")):
        for m in baselines:
            for N, cids in groups.items():
                jobs.append(Job(f"sens_eval_{m}_{tag}_N{N}", _eval_cmd(c, ["--method", m], cids, g)))
    for ck in sorted((c.results / "raw").glob("*/checkpoint_final.pt")):
        method = ck.parent.name.split("__")[0]
        groups = soc if method in SENS_MAIN_METHODS else dt5 if method == "attn_safe_dt5" else {}
        for N, cids in groups.items():
            jobs.append(Job(f"sens_eval_{ck.parent.name}_N{N}", _eval_cmd(c, ["--checkpoint", str(ck)], cids, g)))
    return jobs


def stage_sens_oracle(c: Ctx) -> list[Job]:
    return stage_oracle(c, _sens_grid(c))


def stage_evgnn(c: Ctx) -> list[Job]:
    """Optional EV-GNN baseline (DD27): scripts/run_evgnn.sh (train per profile at N = 20, rho 4; evaluate)."""
    cmd = ["bash", "scripts/run_evgnn.sh", "--device", c.device, "--seeds", str(len(c.seeds))]
    return [Job("evgnn", cmd + (["--smoke"] if c.smoke else []))]


def stage_merge(c: Ctx) -> list[Job]:
    return [Job("merge", [PY, "-m", "v2gsafe.eval.merge", "--raw", str(c.results / "raw")])]


def stage_analysis(c: Ctx) -> list[Job]:
    script = ROOT.parent / "analysis" / "scripts" / "make_all.py"
    if not script.exists():
        print(f"[pipeline] analysis script {script} not found; skipping the analysis stage")
        return []
    return [Job("analysis", [PY, str(script), "--raw", str(c.results / "raw"), "--out", str(ROOT.parent / "analysis")])]


STAGES = {"scenarios": stage_scenarios, "conformal": stage_conformal, "tune": stage_tune,
          "tune_select": stage_tune_select, "train": stage_train, "evaluate": stage_evaluate,
          "oracle": stage_oracle, "merge": stage_merge, "analysis": stage_analysis, "evgnn": stage_evgnn,
          "sens_scenarios": stage_sens_scenarios, "sens_train": stage_sens_train,
          "sens_evaluate": stage_sens_evaluate, "sens_oracle": stage_sens_oracle}
ALL = ["scenarios", "conformal", "tune", "tune_select", "train", "evaluate", "evgnn", "oracle", "merge", "analysis"]
#: optional composite stage (DD28): run after 'train' (it re-uses the main checkpoints)
SENSITIVITY = ["sens_scenarios", "sens_train", "sens_evaluate", "sens_oracle", "merge"]


def run_jobs(jobs: list[Job], n_par: int, logdir: Path, list_only: bool) -> int:
    logdir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not j.done()]
    print(f"[pipeline] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)
    if list_only:
        for j in todo:
            print(" ".join(shlex.quote(x) for x in j.cmd))
        return 0
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHON", PY)

    def _run(j: Job) -> tuple[str, int, float]:
        t0 = time.time()
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in j.name)[:180]
        with open(logdir / f"{safe}.log", "w") as f:
            f.write(" ".join(shlex.quote(x) for x in j.cmd) + "\n")
            f.flush()
            rc = subprocess.call(j.cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
        return j.name, rc, time.time() - t0

    fails = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, n_par)) as ex:
        for name, rc, dt in ex.map(_run, todo):
            fails += int(rc != 0)
            print(f"[pipeline] {'OK  ' if rc == 0 else 'FAIL'} {name} ({dt:.0f}s)", flush=True)
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="all", help=f"one of {ALL} or 'all' (comma-separated list allowed)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tune", action="store_true", help="include the tuning stages in 'all'")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--with-evgnn", action="store_true", help="include the optional EV-GNN stage in 'all' (DD27)")
    ap.add_argument("--shard", default="0/1", help="i/M: run only jobs with index % M == i")
    ap.add_argument("--list", action="store_true", help="print the commands of the pending jobs and exit")
    a = ap.parse_args(argv)
    c = make_ctx(a)
    stages = ALL if a.stage == "all" else a.stage.split(",")
    if a.stage == "all" and not a.tune:
        stages = [s for s in stages if not s.startswith("tune")]
    if a.stage == "all" and not a.with_evgnn:
        stages = [s for s in stages if s != "evgnn"]
    stages = [x for s in stages for x in (SENSITIVITY if s == "sensitivity" else [s])]
    si, sm = (int(x) for x in a.shard.split("/"))
    fails = 0
    for st in stages:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st}")
        jobs = STAGES[st](c)
        if st not in ("merge", "analysis", "scenarios", "sens_scenarios", "evgnn"):
            jobs = [j for k, j in enumerate(jobs) if k % sm == si]
        print(f"[pipeline] stage {st}", flush=True)
        fails += run_jobs(jobs, 1 if st in ("scenarios", "merge", "analysis", "sens_scenarios", "evgnn") else a.jobs,
                          c.results / "logs" / st, a.list)
        if st == "tune_select" and not a.list:
            from v2gsafe import tuning
            for d in sorted((c.results / "tuning").glob("*")):
                if d.is_dir() and any(d.glob("*/validation.json")):
                    print(f"[pipeline] tuning {d.name}: {tuning.select(d)}")
    if fails:
        print(f"[pipeline] {fails} job(s) failed; see {c.results / 'logs'}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
