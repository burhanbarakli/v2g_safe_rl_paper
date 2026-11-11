#!/usr/bin/env bash
# Sec. 13: resume attn_gres2_safe_penrew s0 from checkpoint_last.pt (OOM stop at 619/1464), then evaluate it.
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/joint
OVL=results/tuning/attn_safe/selected.yaml
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code"
note() { echo "$(date -Is) $*" | tee -a "$LOG/progress.log"; }
note "START gres2 s0 (resume)"
python train.py --config configs/train_attn_gres2_safe_penrew.yaml --seed 0 --overlay "$OVL" --device cuda >> "$LOG/train_gres2_s0.log" 2>&1
note "END   gres2 s0 rc=$?"
while pgrep -f "evaluate.py --checkpoint" >/dev/null; do sleep 60; done
d=$(ls -d results/raw/attn_gres2_safe_penrew__*__s0__*)
[[ -f $d/checkpoint_final.pt ]] && { note "eval $(basename $d)"; python evaluate.py --checkpoint "$d/checkpoint_final.pt" --test-set configs/test_grid.yaml --jobs 10 > "$LOG/eval_$(basename $d).log" 2>&1 || note "FAIL $d"; }
note "DONE gres2 s0"
