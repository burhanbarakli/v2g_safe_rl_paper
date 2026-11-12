#!/usr/bin/env bash
# Mechanism round 3 (review/performance_diagnosis_TR.md Sec. 13): attn_gres2_safe_penrew (jres2 + global input).
# s0 starts when gjoint s1 frees its GPU slot, s1 when jres2 s1 does; then test-grid evaluation of both.
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/joint
OVL=results/tuning/attn_safe/selected.yaml
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code"
note() { echo "$(date -Is) $*" | tee -a "$LOG/progress.log"; }
run() { note "START gres2 s$1"
  python train.py --config configs/train_attn_gres2_safe_penrew.yaml --seed $1 --overlay "$OVL" --device cuda > "$LOG/train_gres2_s$1.log" 2>&1
  note "END   gres2 s$1 rc=$?"; }
( until grep -q "END   gjoint s1" "$LOG/progress.log"; do sleep 60; done; run 0 ) & A=$!
( until grep -q "END   jres2 s1" "$LOG/progress.log"; do sleep 60; done; run 1 ) & B=$!
wait $A; wait $B
for d in results/raw/attn_gres2_safe_penrew__*__s[01]__*; do
  [[ -f $d/checkpoint_final.pt ]] || { note "missing $d"; continue; }
  note "eval $(basename $d)"
  python evaluate.py --checkpoint "$d/checkpoint_final.pt" --test-set configs/test_grid.yaml --jobs 6 > "$LOG/eval_$(basename $d).log" 2>&1 || note "FAIL $d"
done
note "DONE gres2"
