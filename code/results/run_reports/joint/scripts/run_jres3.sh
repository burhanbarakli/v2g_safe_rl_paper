#!/usr/bin/env bash
# Mechanism round 3 (review/performance_diagnosis_TR.md Sec. 13): attn_jres3_safe_penrew (jres2 + rule prior).
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/joint
OVL=results/tuning/attn_safe/selected.yaml
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code"
note() { echo "$(date -Is) $*" | tee -a "$LOG/progress.log"; }
run() { note "START jres3 s$1"
  python train.py --config configs/train_attn_jres3_safe_penrew.yaml --seed $1 --overlay "$OVL" --device cuda > "$LOG/train_jres3_s$1.log" 2>&1
  note "END   jres3 s$1 rc=$?"; }
# memory: another project on this machine holds ~17 GB; start after mech2's evaluation, s1 in parallel only if
# MemAvailable stays above 9 GB with s0 running (else sequentially).
until grep -q "DONE mech2" "$LOG/progress.log"; do sleep 60; done
run 0 & A=$!
sleep 600
avail() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
if (( $(avail) >= 9 )); then run 1 & B=$!; wait $A; wait $B
else note "jres3 s1 deferred (MemAvailable $(avail) GB)"; wait $A; run 1; fi
for d in results/raw/attn_jres3_safe_penrew__*__s[01]__*; do
  [[ -f $d/checkpoint_final.pt ]] || { note "missing $d"; continue; }
  note "eval $(basename $d)"
  python evaluate.py --checkpoint "$d/checkpoint_final.pt" --test-set configs/test_grid.yaml --jobs 6 > "$LOG/eval_$(basename $d).log" 2>&1 || note "FAIL $d"
done
note "DONE jres3"
