#!/usr/bin/env bash
# Early test-grid evaluation of finished Sec. 13 checkpoints (mech2 later skips completed parts).
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/joint
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code"
note() { echo "$(date -Is) $*" | tee -a "$LOG/progress.log"; }
for d in results/raw/attn_jres2_safe_penrew__*__s0__* results/raw/attn_joint_safe_penrew__*__s[01]__* \
         results/raw/attn_jbox_safe_penrew__*__s0__* results/raw/attn_gjoint_safe_penrew__*__s0__*; do
  [[ -f $d/checkpoint_final.pt ]] || { note "early: missing $d"; continue; }
  note "early eval $(basename $d)"
  python evaluate.py --checkpoint "$d/checkpoint_final.pt" --test-set configs/test_grid.yaml --jobs 6 > "$LOG/eval_$(basename $d).log" 2>&1 || note "early FAIL $d"
done
note "DONE early eval"
