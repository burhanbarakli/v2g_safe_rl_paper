#!/usr/bin/env bash
# Sec. 13: re-run the attn_jres3_safe_penrew s0 full-grid evaluation (killed for memory on 11 Nov; completed parts are skipped).
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/joint
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code"
note() { echo "$(date -Is) $*" | tee -a "$LOG/progress.log"; }
d=$(ls -d results/raw/attn_jres3_safe_penrew__*__s0__*)
note "eval (retry, jobs 3) $(basename $d)"
python evaluate.py --checkpoint "$d/checkpoint_final.pt" --test-set configs/test_grid.yaml --jobs 3 > "$LOG/eval_$(basename $d)_retry.log" 2>&1 && note "OK   jres3 s0 eval" || note "FAIL jres3 s0 eval (retry)"
