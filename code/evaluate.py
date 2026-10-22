#!/usr/bin/env python
"""Evaluate a method on fixed test replays and write episodes.csv (code_spec.md Sec. 3.1).

Learned method (writes into the training run directory unless --out is given):
    python evaluate.py --checkpoint results/raw/<exp_id>/checkpoint_final.pt --test-set configs/test_grid.yaml
    python evaluate.py --checkpoint .../checkpoint_final.pt --test-set configs/test_grid.yaml --admission off
    python evaluate.py --checkpoint .../checkpoint_final.pt --test-set configs/test_grid.yaml \
                       --conformal-alpha 0.01 0.05 0.1 0.2          # SA-Attn-CP at several alpha
Baselines (afap, alap, rr, llf, empc, empc_highs, evgnn; '_safe' = + safety layer):
    python evaluate.py --method llf_safe --test-set configs/test_grid.yaml --out-root results/raw
Single configuration:
    python evaluate.py --method afap --test-set configs/test_grid.yaml --only workplace_N10_rho4

Options: --jobs N (parallel configurations), --log-steps (steps.parquet), --max-replays K, --smoke.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from v2gsafe.config import exp_id as make_exp_id  # noqa: E402
from v2gsafe.envs.scenarios import load_test_grid  # noqa: E402
from v2gsafe.eval.evaluate import MethodSpec, evaluate  # noqa: E402
from v2gsafe.utils import RunMeta  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", default=None, help="baseline name (ignored when --checkpoint is given)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--test-set", required=True, help="test grid YAML (e.g. configs/test_grid.yaml)")
    ap.add_argument("--test-sets-dir", default=None, help="manifest directory (default: results/test_sets, "
                    "results/smoke/test_sets with --smoke)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these test_cfg_ids")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-root", default="results/raw")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--admission", choices=["on", "off", "default"], default="default",
                    help="DD31: admission control is on for every method by default; 'off' -> <method>_noadm")
    ap.add_argument("--deploy-safety", action="store_true",
                    help="penalty-trained checkpoint evaluated with the safety layer (<method>_safedeploy, DD33)")
    ap.add_argument("--admission-targets", choices=["admitted", "lp_reachable"], default=None,
                    help="override safety.admission_existing_targets (IMPLEMENTATION_NOTES I-3)")
    ap.add_argument("--conformal-alpha", type=float, nargs="*", default=None)
    ap.add_argument("--conformal-uncalibrated", action="store_true", help="SA-Attn-point ablation (Q_hat = 0)")
    ap.add_argument("--conformal-cache", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--max-replays", type=int, default=None)
    ap.add_argument("--log-steps", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--evgnn-repo", default=None, help="clone of github.com/StavrosOrf/EV-GNN (method evgnn)")
    ap.add_argument("--evgnn-model-dir", default=None, help="EV-GNN saved_models/<run>/ directory")
    ap.add_argument("--seed", type=int, default=0, help="seed column/exp_id for non-checkpoint methods (e.g. EV-GNN)")
    args = ap.parse_args(argv)

    if args.checkpoint is None and args.method is None:
        ap.error("give --method or --checkpoint")
    test_sets = args.test_sets_dir or ("results/smoke/test_sets" if args.smoke else "results/test_sets")
    conf_cache = args.conformal_cache or ("results/smoke/conformal" if args.smoke else "results/conformal")
    adm = {"on": True, "off": False, "default": None}[args.admission]
    grid = load_test_grid(args.test_set)
    grid_name = Path(args.test_set).stem
    alphas = args.conformal_alpha if args.conformal_alpha else [None]
    for alpha in alphas:
        spec = MethodSpec(method=args.method or "learned", checkpoint=args.checkpoint, device=args.device,
                          admission=adm, deploy_safety=args.deploy_safety, conformal_alpha=alpha,
                          conformal_uncalibrated=args.conformal_uncalibrated,
                          conformal_cache=conf_cache, admission_existing_targets=args.admission_targets,
                          torch_threads=args.torch_threads, root=str(ROOT),
                          extra={"evgnn_repo": args.evgnn_repo, "evgnn_model_dir": args.evgnn_model_dir,
                                 "seed": args.seed})
        if args.checkpoint:
            out = Path(args.out) if args.out else Path(args.checkpoint).resolve().parent
            eid = out.name
        else:
            h = hashlib.sha256(json.dumps({"method": args.method, "admission": adm, "grid": Path(args.test_set).read_text(),
                                           "targets": args.admission_targets, "evgnn": args.evgnn_model_dir},
                                          sort_keys=True).encode()).hexdigest()[:8]
            name = args.method + ("_noadm" if adm is False else "")
            eid = make_exp_id(name, grid_name, args.seed, h)
            out = Path(args.out) if args.out else Path(args.out_root) / eid
        out.mkdir(parents=True, exist_ok=True)
        meta_name = "meta.json" if not args.checkpoint else None
        meta = None
        if meta_name:
            meta = RunMeta(out, args.smoke, args.device, {"exp_id": eid, "method": spec.method, "grid": args.test_set})
        else:
            (out / "eval_runs.jsonl").open("a").write(json.dumps({"argv": sys.argv, "alpha": alpha}) + "\n")
        path = evaluate(spec, grid, test_sets, out, eid, smoke=args.smoke, log_steps=args.log_steps, jobs=args.jobs,
                        only=args.only, max_replays=args.max_replays)
        if meta:
            meta.finish()
        print(f"[evaluate] wrote {path}")


if __name__ == "__main__":
    main()
