"""End-to-end pipeline driver used by ``reproduce.sh`` (job lists, resume, parallel jobs, machine sharding).

Stages: scenarios -> admission -> conformal -> tune -> tune_select -> train -> evaluate -> [evgnn] -> oracle ->
merge -> analysis. Optional composite stages: ``sensitivity`` (DD28/DD39) and ``reference`` (DD36, attn_safe trained
at N = 50; part of 'train'/'evaluate' since DD42, the composite stage re-runs only these jobs). Every job is a subprocess command; completed jobs are skipped (resume = re-run the same command). With
``--shard i/M`` only every M-th job (offset i) of a stage is run, so several machines can split a stage; copy
results/** back to one machine before 'merge' and 'analysis'.

Run sets (README "Compute"): the CORE set (default) = main methods incl. the DD33 baselines, attn_safe_trainN50
(3 seeds, evaluated at N in {50, 100, 200}; DD42), tuning, heuristics, HiGHS eMPC, noadm/safedeploy evaluation
passes and the oracle; ``--extended`` adds the ablations, the w-sensitivity,
eMPC-24h and the Gurobi eMPC on the literature configuration (DD34, if gurobipy is installed).

Final training runs need tuned hyperparameters (DD37): results of the tune/tune_select stages, or an explicit
``--hparams FILE`` (e.g. configs/hparams_defaults.yaml = the Alg. 6 defaults).

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

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ALPHAS = [0.01, 0.05, 0.1, 0.2]

#: core training configurations (5 seeds)
MAIN_CONFIGS = ["configs/train_attn_safe.yaml", "configs/train_attn_penalty.yaml", "configs/train_mlp_safe.yaml",
                "configs/train_mlp_penalty.yaml", "configs/train_attn_safe_conformal.yaml",
                "configs/train_attn_proj_nolb.yaml", "configs/train_attn_lagrangian.yaml",
                "configs/train_deepsets_safe.yaml"]
#: configurations tuned with their own grid (DD37); every other run re-uses the selection of its base method
TUNE_CONFIGS = ["configs/train_attn_safe.yaml", "configs/train_attn_penalty.yaml", "configs/train_mlp_safe.yaml",
                "configs/train_mlp_penalty.yaml", "configs/train_attn_proj_nolb.yaml",
                "configs/train_attn_lagrangian.yaml", "configs/train_deepsets_safe.yaml"]
#: extended set (5 seeds / 3 seeds)
ABLATION_CONFIGS = ["configs/ablations/train_attn_safe_beta0.yaml", "configs/ablations/train_attn_safe_curriculum.yaml",
                    "configs/ablations/train_attn_safe_small.yaml", "configs/ablations/train_attn_penalty_dense.yaml"]
SENS_CONFIGS = ["configs/ablations/train_attn_penalty_w1.yaml", "configs/ablations/train_attn_penalty_w100.yaml"]
REFERENCE_CONFIG = "configs/train_attn_safe_trainN50.yaml"
#: method -> (tuned base method, PPO keys only)
TUNED_METHODS = {"attn_safe": ("attn_safe", False), "attn_penalty": ("attn_penalty", False),
                 "mlp_safe": ("mlp_safe", False), "mlp_penalty": ("mlp_penalty", False),
                 "attn_proj_nolb": ("attn_proj_nolb", False), "attn_lagrangian": ("attn_lagrangian", False),
                 "deepsets_safe": ("deepsets_safe", False), "attn_safe_conformal": ("attn_safe", False),
                 "attn_safe_beta0": ("attn_safe", False), "attn_safe_curriculum": ("attn_safe", False),
                 "attn_safe_small": ("attn_safe", False), "attn_safe_dt5": ("attn_safe", False),
                 "attn_safe_trainN50": ("attn_safe", False), "attn_penalty_dense": ("attn_penalty", False),
                 "attn_penalty_w1": ("attn_penalty", True), "attn_penalty_w100": ("attn_penalty", True)}
HEURISTIC_METHODS = ["afap", "alap", "rr", "llf", "sllf", "price_llf"]
NOADM_METHODS = ("attn_safe", "attn_penalty")          # DD31: noadm pass only for these
SAFEDEPLOY_METHODS = ("attn_penalty",)                   # DD33(2)
REFERENCE_NS = (50, 100, 200)


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
    extended: bool
    hparams: Path | None


def make_ctx(a) -> Ctx:
    smoke = a.smoke
    res = ROOT / ("results/smoke" if smoke else "results")
    hp = Path(a.hparams).resolve() if a.hparams else None
    return Ctx(smoke=smoke, device=a.device, seeds=[0] if smoke else list(range(a.seeds)),
               sens_seeds=[0] if smoke else list(range(min(3, a.seeds))), results=res,
               grid="configs/test_grid_smoke.yaml" if smoke else "configs/test_grid.yaml",
               overlay=["configs/smoke.yaml"] if smoke else [], extended=a.extended, hparams=hp)


def _smoke_flags(c: Ctx) -> list[str]:
    return ["--smoke"] if c.smoke else []


# ------------------------------------------------------------------------------------------ stages
def stage_scenarios(c: Ctx) -> list[Job]:
    cmd = [PY, "-m", "v2gsafe.envs.scenarios", "build-test-sets", "--grid", c.grid, "--out",
           str(c.results / "test_sets")]
    return [Job("scenarios", cmd)]


def stage_admission(c: Ctx, grid: str | None = None) -> list[Job]:
    """DD31: controller-independent admission tables of every test replay (shared by all methods and the oracle)."""
    out = c.results / "admission"
    jobs = []
    for cid, _ in _grid_ids(c, grid):
        jobs.append(Job(f"admission_{cid}", [PY, "-m", "v2gsafe.safety.admission_prepass", "--grid", grid or c.grid,
                                             "--only", cid, "--test-sets", str(c.results / "test_sets"),
                                             "--out", str(out)],
                        done=lambda cid=cid: (out / f"{cid}.done").exists()))
    return jobs


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
    """Tuned hyperparameters of a final run (DD37): --hparams FILE (per-method overlays, key 'default' for all) or
    the tune_select output of the method's base; refuses to continue without either."""
    base, ppo_only = TUNED_METHODS.get(method, (method, False))
    if c.hparams is not None:
        hp = yaml.safe_load(c.hparams.read_text()) or {}
        ov = hp.get(method, hp.get(base, hp.get("default", {}))) or {}
        if ppo_only:
            ov = {"ppo": ov["ppo"]} if "ppo" in ov else {}
        p = c.results / "tuning" / "_hparams" / f"{method}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# from {c.hparams}\n" + yaml.safe_dump(ov))
        return [str(p)]
    p = c.results / "tuning" / base / ("selected_ppo.yaml" if ppo_only else "selected.yaml")
    if not p.exists():
        raise SystemExit(f"[pipeline] no tuned hyperparameters for {method} ({p} missing). Run the tune and "
                         f"tune_select stages first, or pass --hparams FILE (configs/hparams_defaults.yaml = Alg. 6 "
                         f"defaults) to train with explicit values (DD37).")
    return [str(p)]


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
    jname = f"train_{Path(out).name}" if "__" in Path(out).name else f"tune_{Path(out).parent.name}_{Path(out).name}"
    return Job(jname, cmd, done=lambda out=out: (Path(out) / "checkpoint_final.pt").exists())


def stage_tune(c: Ctx) -> list[Job]:
    from v2gsafe import tuning
    jobs = []
    for config in TUNE_CONFIGS:
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
    jobs += stage_reference_train(c)                    # DD42(5): attn_safe_trainN50 (3 seeds) is core
    if c.extended:
        jobs += [_train_job(c, cfg, s) for cfg in ABLATION_CONFIGS for s in c.seeds]
        jobs += [_train_job(c, cfg, s) for cfg in SENS_CONFIGS for s in c.sens_seeds]
    return jobs


def _eval_cmd(c: Ctx, extra: list[str], cids: list[str], grid: str | None = None) -> list[str]:
    return [PY, "evaluate.py", "--test-set", grid or c.grid, "--only", *cids, "--device", c.device,
            "--test-sets-dir", str(c.results / "test_sets"), "--conformal-cache", str(c.results / "conformal"),
            "--admission-cache", str(c.results / "admission"), "--out-root", str(c.results / "raw"),
            *_smoke_flags(c), *extra]


def _groups_by_N(ids, max_N: int | None = None) -> dict[int, list[str]]:
    """Test configurations grouped by station size (one EV2Gym env per job is reused across the group)."""
    out: dict[int, list[str]] = {}
    for cid, N in ids:
        if max_N is None or N <= max_N:
            out.setdefault(N, []).append(cid)
    return out


def _has_gurobi() -> bool:
    try:
        import gurobipy  # noqa: F401
        return True
    except ImportError:
        return False


def _baselines(c: Ctx) -> list[str]:
    out = [m for h in HEURISTIC_METHODS for m in (h, h + "_safe")] + ["empc_highs", "empc_highs_safe"]
    if c.extended:
        out += ["empc_highs_24h", "empc_highs_24h_safe"]
    return out


def _checkpoint_variants(method: str) -> list[list[str]]:
    if method.endswith("_conformal"):
        return [["--conformal-alpha", str(a)] for a in ALPHAS] + [["--conformal-uncalibrated"]]
    v = [[]]
    if method in NOADM_METHODS:
        v.append(["--admission", "off"])
    if method in SAFEDEPLOY_METHODS:
        v.append(["--deploy-safety"])
    return v


def stage_evaluate(c: Ctx) -> list[Job]:
    jobs = []
    ids = _grid_ids(c)
    groups = _groups_by_N(ids)
    for m in _baselines(c):
        for N, cids in groups.items():
            jobs.append(Job(f"eval_{m}_N{N}", _eval_cmd(c, ["--method", m], cids)))
    lit = [cid for cid, _ in ids if cid.startswith("literature")]
    if c.extended and lit and _has_gurobi():           # DD34: Gurobi eMPC only on the literature configuration
        for m in ("empc", "empc_safe"):
            jobs.append(Job(f"eval_{m}_literature", _eval_cmd(c, ["--method", m], lit)))
    for ck in sorted((c.results / "raw").glob("*/checkpoint_final.pt")):
        name = ck.parent.name
        method = name.split("__")[0]
        if method == "attn_safe_trainN50":             # evaluated at N in {50, 100, 200} only (below)
            continue
        g_m = groups
        if method.startswith("mlp_"):                    # DD43(3): fixed-size reference, N <= 20 only
            g_m = {N: cids for N, cids in groups.items() if N <= 20}
            skipped = sorted(N for N in groups if N > 20)
            if skipped:
                print(f"[pipeline] note: {name} not evaluated at N = {skipped} (MLP baseline only at N <= 20, DD43(3))")
        for v in _checkpoint_variants(method):
            for N, cids in g_m.items():
                jobs.append(Job(f"eval_{name}_{'_'.join(v)}_N{N}",
                                _eval_cmd(c, ["--checkpoint", str(ck), *v], cids)))
    return jobs + stage_reference_evaluate(c)


def stage_oracle(c: Ctx, grid: str | None = None) -> list[Job]:
    jobs = []
    for cid, _ in _grid_ids(c, grid):
        out = c.results / "raw" / f"oracle__{cid}" / "episodes.csv"
        jobs.append(Job(f"oracle_{cid}", [PY, "-m", "v2gsafe.baselines.oracle", "--grid", grid or c.grid, "--only", cid,
                                          "--test-sets", str(c.results / "test_sets"), "--out", str(c.results / "raw"),
                                          "--admission-cache", str(c.results / "admission"),
                                          *_smoke_flags(c)], done=lambda out=out: out.exists()))
    return jobs


# ------------------------------------------------------------------- optional stages (DD27, DD28)
SENS_TRAIN = "configs/ablations/train_attn_safe_dt5.yaml"
SENS_MAIN_METHODS = ("attn_safe", "attn_penalty", "mlp_safe", "mlp_penalty", "attn_proj_nolb", "attn_lagrangian",
                     "deepsets_safe")


def _sens_grid(c: Ctx) -> str:
    return "configs/test_grid_sensitivity_smoke.yaml" if c.smoke else "configs/test_grid_sensitivity.yaml"


def stage_sens_scenarios(c: Ctx) -> list[Job]:
    return [Job("sens_scenarios", [PY, "-m", "v2gsafe.envs.scenarios", "build-test-sets", "--grid", _sens_grid(c),
                                   "--out", str(c.results / "test_sets")])]


def stage_sens_train(c: Ctx) -> list[Job]:
    sets = ["scenario.T=48"] if c.smoke else []          # 5-min smoke horizon: 4 h (T = 16 fits no session)
    return [_train_job(c, SENS_TRAIN, s, sets=sets) for s in c.seeds]


def stage_sens_admission(c: Ctx) -> list[Job]:
    return stage_admission(c, _sens_grid(c))


def stage_sens_evaluate(c: Ctx) -> list[Job]:
    """15-min configurations (desired SoC 0.8, throughput cost, discharge price factor): main policies + baselines;
    5-min configurations: dt5 policy + baselines (DD28, DD39)."""
    g = _sens_grid(c)
    main = _groups_by_N(_grid_ids(c, g, dt=15))
    dt5 = _groups_by_N(_grid_ids(c, g, dt=5))
    jobs = []
    for groups, tag in ((main, "dt15"), (dt5, "dt5")):
        for m in _baselines(c):
            for N, cids in groups.items():
                jobs.append(Job(f"sens_eval_{m}_{tag}_N{N}", _eval_cmd(c, ["--method", m], cids, g)))
    for ck in sorted((c.results / "raw").glob("*/checkpoint_final.pt")):
        method = ck.parent.name.split("__")[0]
        groups = main if method in SENS_MAIN_METHODS else dt5 if method == "attn_safe_dt5" else {}
        for N, cids in groups.items():
            jobs.append(Job(f"sens_eval_{ck.parent.name}_N{N}", _eval_cmd(c, ["--checkpoint", str(ck)], cids, g)))
    return jobs


def stage_sens_oracle(c: Ctx) -> list[Job]:
    return stage_oracle(c, _sens_grid(c))


def stage_reference_train(c: Ctx) -> list[Job]:
    """DD36: SA-Attn trained at N = 50 (3 seeds)."""
    return [_train_job(c, REFERENCE_CONFIG, s) for s in c.sens_seeds]     # smoke: the smoke overlay sets N in {3, 5}


def stage_reference_evaluate(c: Ctx) -> list[Job]:
    """DD36: the N = 50 policy on the main-grid configurations with N in {50, 100, 200} (smoke: the largest N)."""
    ids = _grid_ids(c)
    Ns = REFERENCE_NS if not c.smoke else (max(N for _, N in ids),)
    groups = {N: cids for N, cids in _groups_by_N(ids).items() if N in Ns}
    jobs = []
    for ck in sorted((c.results / "raw").glob("attn_safe_trainN50__*/checkpoint_final.pt")):
        for N, cids in groups.items():
            jobs.append(Job(f"ref_eval_{ck.parent.name}_N{N}", _eval_cmd(c, ["--checkpoint", str(ck)], cids)))
    return jobs


def stage_gnndt(c: Ctx) -> list[Job]:
    """DD35: GNN-DT adapter stub (instructions only, v2gsafe/baselines/gnndt_adapter.md)."""
    return [Job("gnndt", ["bash", "scripts/run_gnndt.sh"] + (["--smoke"] if c.smoke else []))]


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


STAGES = {"scenarios": stage_scenarios, "admission": stage_admission, "conformal": stage_conformal,
          "tune": stage_tune, "tune_select": stage_tune_select, "train": stage_train, "evaluate": stage_evaluate,
          "oracle": stage_oracle, "merge": stage_merge, "analysis": stage_analysis, "evgnn": stage_evgnn,
          "gnndt": stage_gnndt, "sens_scenarios": stage_sens_scenarios, "sens_admission": stage_sens_admission,
          "sens_train": stage_sens_train, "sens_evaluate": stage_sens_evaluate, "sens_oracle": stage_sens_oracle,
          "reference_train": stage_reference_train, "reference_evaluate": stage_reference_evaluate}
ALL = ["scenarios", "admission", "conformal", "tune", "tune_select", "train", "evaluate", "evgnn", "gnndt", "oracle",
       "merge", "analysis"]
#: optional composite stages: run after 'train' (they re-use the main checkpoints / tuning results)
COMPOSITE = {"sensitivity": ["sens_scenarios", "sens_admission", "sens_train", "sens_evaluate", "sens_oracle", "merge"],
             "reference": ["reference_train", "reference_evaluate", "merge"]}
SERIAL = ("scenarios", "merge", "analysis", "sens_scenarios", "evgnn", "gnndt")


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
    ap.add_argument("--stage", default="all", help=f"one of {ALL}, {list(COMPOSITE)} or 'all' (comma-separated "
                    "list allowed)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=5, help="seeds per learned method; also caps the 3-seed runs "
                    "(attn_safe_trainN50, w-sensitivity)")
    ap.add_argument("--extended", action="store_true", help="include the extended run set (ablations, w-sensitivity, "
                    "eMPC-24h, Gurobi eMPC on the literature configuration)")
    ap.add_argument("--hparams", default=None, help="YAML of per-method hyperparameter overlays used instead of the "
                    "tuning results (DD37); configs/hparams_defaults.yaml = Alg. 6 defaults")
    ap.add_argument("--no-tune", action="store_true", help="skip tune/tune_select in 'all' (requires --hparams or "
                    "existing tuning results)")
    ap.add_argument("--with-evgnn", action="store_true", help="include the optional EV-GNN stage in 'all' (DD27)")
    ap.add_argument("--with-gnndt", action="store_true", help="include the GNN-DT stub stage in 'all' (DD35)")
    ap.add_argument("--shard", default="0/1", help="i/M: run only jobs with index % M == i")
    ap.add_argument("--list", action="store_true", help="print the commands of the pending jobs and exit")
    a = ap.parse_args(argv)
    c = make_ctx(a)
    stages = ALL if a.stage == "all" else a.stage.split(",")
    if a.stage == "all" and (a.no_tune or a.hparams):
        stages = [s for s in stages if not s.startswith("tune")]
    if a.stage == "all" and not a.with_evgnn:
        stages = [s for s in stages if s != "evgnn"]
    if a.stage == "all" and not a.with_gnndt:
        stages = [s for s in stages if s != "gnndt"]
    stages = [x for s in stages for x in COMPOSITE.get(s, [s])]
    si, sm = (int(x) for x in a.shard.split("/"))
    fails = 0
    for st in stages:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st}")
        jobs = STAGES[st](c)
        if st not in SERIAL:
            jobs = [j for k, j in enumerate(jobs) if k % sm == si]
        print(f"[pipeline] stage {st}", flush=True)
        fails += run_jobs(jobs, 1 if st in SERIAL else a.jobs, c.results / "logs" / st, a.list)
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
