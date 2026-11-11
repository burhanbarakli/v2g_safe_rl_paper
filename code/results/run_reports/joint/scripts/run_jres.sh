#!/usr/bin/env bash
# Residual policy around the price-LLF rule (attn_jres_safe_penrew), 2 seeds, then test-grid evaluation of all
# new SA-Attn checkpoints (joint s0/s1, jbox s0, jres s0/s1). Replaces the jbox seed-1 run (box action showed no gain).
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/joint
OVL=results/tuning/attn_safe/selected.yaml
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code"
tr() { python train.py --config configs/train_attn_jres_safe_penrew.yaml --seed "$1" --overlay "$OVL" --device cuda > "$LOG/train_jres_s$1.log" 2>&1; }
echo "$(date -Is) START jres s0" | tee -a "$LOG/progress.log"
tr 0 & P0=$!
while pgrep -f "train_attn_joint_safe_penrew.yaml --seed 1" >/dev/null; do sleep 60; done
echo "$(date -Is) START jres s1" | tee -a "$LOG/progress.log"
tr 1 & P1=$!
wait $P0; r0=$?; wait $P1; r1=$?
echo "$(date -Is) END   jres rc=$r0,$r1" | tee -a "$LOG/progress.log"
while pgrep -f "evaluate.py --checkpoint" >/dev/null; do sleep 60; done
for d in results/raw/attn_joint_safe_penrew__*__s[01]__* results/raw/attn_jbox_safe_penrew__*__s0__* results/raw/attn_jres_safe_penrew__*__s[01]__*; do
  [[ -f $d/checkpoint_final.pt ]] || { echo "missing $d"; continue; }
  echo "$(date -Is) eval $d" | tee -a "$LOG/progress.log"
  python evaluate.py --checkpoint "$d/checkpoint_final.pt" --test-set configs/test_grid.yaml --jobs 10 > "$LOG/eval_$(basename $d).log" 2>&1 || echo "FAIL $d" | tee -a "$LOG/progress.log"
done
echo "$(date -Is) DONE jres+eval" | tee -a "$LOG/progress.log"
