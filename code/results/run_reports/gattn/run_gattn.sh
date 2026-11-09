#!/usr/bin/env bash
# Diagnostic runs (2 seeds each, penalty reward): SA-Attn with global features in every charger embedding
# (attn_gin_safe_penrew) and additionally the DeepSets-style head context (attn_gctx_safe_penrew). Two training jobs at
# a time on the GPU (both variants, one seed per round), then evaluation of the new checkpoints on the test grid.
# Resumable: train.py resumes from checkpoint_last.pt, the evaluate stage skips completed configurations.
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/gattn
OVL=results/tuning/attn_safe/selected.yaml
mkdir -p "$LOG"
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

train_round() {  # train_round <seed>
  local s=$1 pids=() rc=0
  echo "$(date -Is) START train seed $s" | tee -a "$LOG/progress.log"
  for v in gin gctx; do
    python train.py --config "configs/train_attn_${v}_safe_penrew.yaml" --seed "$s" --overlay "$OVL" --device cuda \
      > "$LOG/train_${v}_s$s.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "$(date -Is) END   train seed $s rc=$rc" | tee -a "$LOG/progress.log"
  return $rc
}

train_round 0 || exit 1
train_round 1 || exit 1

echo "$(date -Is) START evaluate" | tee -a "$LOG/progress.log"
./reproduce.sh --stage evaluate,merge --jobs 10 --seeds 2 > "$LOG/evaluate.log" 2>&1
rc=$?
echo "$(date -Is) END   evaluate rc=$rc" | tee -a "$LOG/progress.log"
[[ $rc -eq 0 ]] && echo "$(date -Is) DONE" | tee -a "$LOG/progress.log"
exit $rc
