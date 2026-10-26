#!/usr/bin/env bash
# Diagnostic run: SA-Attn with the penalty reward (2 seeds), then evaluation of the new checkpoints on the test grid.
# Resumable: train.py resumes from checkpoint_last.pt, the evaluate stage skips completed configurations.
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/penrew
mkdir -p "$LOG"
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
OVL=results/tuning/attn_safe/selected.yaml

echo "$(date -Is) START train" | tee -a "$LOG/progress.log"
pids=()
for s in 0 1; do
  python train.py --config configs/train_attn_safe_penrew.yaml --seed "$s" --overlay "$OVL" --device cuda \
    > "$LOG/train_s$s.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "$(date -Is) END   train rc=$rc" | tee -a "$LOG/progress.log"
[[ $rc -eq 0 ]] || exit 1

echo "$(date -Is) START evaluate" | tee -a "$LOG/progress.log"
./reproduce.sh --stage evaluate,merge --jobs 10 --seeds 2 > "$LOG/evaluate.log" 2>&1
rc=$?
echo "$(date -Is) END   evaluate rc=$rc" | tee -a "$LOG/progress.log"
[[ $rc -eq 0 ]] && echo "$(date -Is) DONE" | tee -a "$LOG/progress.log"
exit $rc
