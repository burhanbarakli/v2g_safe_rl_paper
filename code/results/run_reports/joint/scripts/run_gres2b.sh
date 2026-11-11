#!/usr/bin/env bash
# Sec. 13, round 4: attn_gres2_safe_penrew s0 (resumed from checkpoint_last.pt after the OOM stop) and s1 in parallel,
# test-grid evaluation of attn_jres3_safe_penrew s0 alongside when memory allows, then of gres2.
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/joint
OVL=results/tuning/attn_safe/selected.yaml
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code"
note() { echo "$(date -Is) $*" | tee -a "$LOG/progress.log"; }
avail() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
run() { note "START gres2 s$1"
  python train.py --config configs/train_attn_gres2_safe_penrew.yaml --seed $1 --overlay "$OVL" --device cuda >> "$LOG/train_gres2_s$1.log" 2>&1
  note "END   gres2 s$1 rc=$?"; }
ev() { [[ -f $1/checkpoint_final.pt ]] || { note "missing $1"; return; }
  note "eval $(basename $1)"
  python evaluate.py --checkpoint "$1/checkpoint_final.pt" --test-set configs/test_grid.yaml --jobs ${2:-6} > "$LOG/eval_$(basename $1).log" 2>&1 || note "FAIL $1"; }
run 0 & A=$!
run 1 & B=$!
sleep 600
J3=$(ls -d results/raw/attn_jres3_safe_penrew__*__s0__*)
if (( $(avail) >= 7 )); then ev "$J3" 4; else note "jres3 eval deferred (MemAvailable $(avail) GB)"; fi
wait $A; wait $B
ev "$J3" 10
for d in results/raw/attn_gres2_safe_penrew__*__s[01]__*; do ev "$d" 10; done
note "DONE gres2b"
