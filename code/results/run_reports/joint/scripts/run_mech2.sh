#!/usr/bin/env bash
# Mechanism round 2 (review/performance_diagnosis_TR.md Sec. 13). Lane A: attn_gjoint_safe_penrew s0, s1 (gin + joint
# layer, Deep Sets PPO hyperparameters, no tuning overlay). Lane B: after attn_joint s1 finishes, attn_jres2_safe_penrew
# s0, s1 (softened residual, attn_safe tuning overlay). Then test-grid evaluation of all new SA-Attn checkpoints.
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/joint
OVL=results/tuning/attn_safe/selected.yaml
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code"
note() { echo "$(date -Is) $*" | tee -a "$LOG/progress.log"; }
laneA() {
  for s in 0 1; do note "START gjoint s$s"
    python train.py --config configs/train_attn_gjoint_safe_penrew.yaml --seed $s --device cuda > "$LOG/train_gjoint_s$s.log" 2>&1
    note "END   gjoint s$s rc=$?"; done
}
laneB() {
  while pgrep -f "train_attn_joint_safe_penrew.yaml --seed 1" >/dev/null; do sleep 60; done
  for s in 0 1; do note "START jres2 s$s"
    python train.py --config configs/train_attn_jres2_safe_penrew.yaml --seed $s --overlay "$OVL" --device cuda > "$LOG/train_jres2_s$s.log" 2>&1
    note "END   jres2 s$s rc=$?"; done
}
laneA & A=$!
laneB & B=$!
wait $A; wait $B
while pgrep -f "evaluate.py --checkpoint" >/dev/null; do sleep 60; done
for d in results/raw/attn_joint_safe_penrew__*__s[01]__* results/raw/attn_jbox_safe_penrew__*__s0__* \
         results/raw/attn_gjoint_safe_penrew__*__s[01]__* results/raw/attn_jres2_safe_penrew__*__s[01]__*; do
  [[ -f $d/checkpoint_final.pt ]] || { note "missing $d"; continue; }
  note "eval $(basename $d)"
  python evaluate.py --checkpoint "$d/checkpoint_final.pt" --test-set configs/test_grid.yaml --jobs 10 > "$LOG/eval_$(basename $d).log" 2>&1 || note "FAIL $d"
done
note "DONE mech2"
