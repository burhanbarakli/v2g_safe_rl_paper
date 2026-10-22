#!/usr/bin/env python
"""Train a learned method with PPO (algorithm.md Alg. 2).

    python train.py --config configs/train_attn_safe.yaml --seed 0 --out results/raw/<exp_id>
    python train.py --config configs/train_attn_safe.yaml --seed 0 --out-root results/raw     # exp_id auto
    python train.py ... --resume                       # continue from <out>/checkpoint_last.pt
    python train.py ... --set ppo.lr=1e-4 --set ppo.total_env_steps=500000

Outputs in the run directory (code_spec.md Sec. 2): config.yaml, meta.json, train_log.csv, checkpoint_last.pt,
checkpoint_final.pt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from v2gsafe.algos.ppo import PPOTrainer  # noqa: E402
from v2gsafe.config import config_hash8, exp_id, load_config  # noqa: E402


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="run directory (default: <out-root>/<exp_id>)")
    ap.add_argument("--out-root", default="results/raw")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 ...")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tag outputs as smoke (never reported)")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. ppo.lr=1e-4")
    ap.add_argument("--overlay", action="append", default=[], help="YAML overlay merged after --config "
                    "(e.g. configs/smoke.yaml)")
    ap.add_argument("--backend", default=None, choices=[None, "sync", "subproc"])
    args = ap.parse_args(argv)
    overrides = list(args.set)
    if args.smoke:
        overrides.append("smoke=true")
    if args.backend:
        overrides.append(f"envs.backend={args.backend}")
    cfg = load_config(args.config, overrides, args.overlay)
    eid = exp_id(cfg["method"], cfg["_name"], args.seed, config_hash8(cfg))
    out = Path(args.out) if args.out else Path(args.out_root) / eid
    if (out / "checkpoint_final.pt").exists() and not args.resume:
        print(f"{out} already has checkpoint_final.pt; nothing to do (use --resume to continue a partial run)")
        return out / "checkpoint_final.pt"
    if not args.resume and (out / "checkpoint_last.pt").exists():
        print(f"{out} has checkpoint_last.pt from an interrupted run; resuming")
        args.resume = True
    trainer = PPOTrainer(cfg, out, args.seed, args.device, resume=args.resume, root=ROOT, exp_id=eid)
    print(f"[train] {eid}: {trainer.num_updates} updates x {trainer.E * trainer.S} transitions -> {out}", flush=True)
    path = trainer.train()
    print(f"[train] done: {path}")
    return path


if __name__ == "__main__":
    main()
