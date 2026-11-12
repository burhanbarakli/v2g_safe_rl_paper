#!/usr/bin/env bash
# Sec. 13, after the power outage of 11 Nov 2026: resume attn_gres2_safe_penrew s0 (update 700) and s1 (update 150)
# from checkpoint_last.pt in parallel (train logs repaired to the checkpoint update), evaluate attn_jres3_safe_penrew
# s0 alongside, then gres2.
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/joint
OVL=results/tuning/attn_safe/selected.yaml
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code"
note() { echo "$(date -Is) $*" | tee -a "$LOG/progress.log"; }
run() { note "START gres2 s$1 (resume after power outage)"
  python train.py --config configs/train_attn_gres2_safe_penrew.yaml --seed $1 --overlay "$OVL" --device cuda >> "$LOG/train_gres2_s$1.log" 2>&1
  note "END   gres2 s$1 rc=$?"; }
ev() { [[ -f $1/checkpoint_final.pt ]] || { note "missing $1"; return; }
  note "eval $(basename $1)"
  python evaluate.py --checkpoint "$1/checkpoint_final.pt" --test-set configs/test_grid.yaml --jobs ${2:-6} > "$LOG/eval_$(basename $1).log" 2>&1 || note "FAIL $1"; }
run 0 & A=$!
run 1 & B=$!
sleep 120
ev "$(ls -d results/raw/attn_jres3_safe_penrew__*__s0__*)" 6
wait $A; wait $B
for d in results/raw/attn_gres2_safe_penrew__*__s[01]__*; do ev "$d" 10; done
note "DONE gres2c"
