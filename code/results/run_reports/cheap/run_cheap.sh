#!/usr/bin/env bash
# Diagnostic runs (2 seeds each): Deep Sets + layer with the penalty reward, and SA-Attn with the penalty reward plus an
# entropy penalty. Two training jobs at a time on the GPU, then evaluation of the new checkpoints on the test grid.
# Resumable: train.py resumes from checkpoint_last.pt, the evaluate stage skips completed configurations.
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/cheap
mkdir -p "$LOG"
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

train_pair() {  # train_pair <tag> <config> <overlay> [--set ...]
  local tag=$1 cfg=$2 ovl=$3; shift 3
  echo "$(date -Is) START train $tag" | tee -a "$LOG/progress.log"
  local pids=() rc=0
  for s in 0 1; do
    python train.py --config "$cfg" --seed "$s" --overlay "$ovl" --device cuda "$@" > "$LOG/train_${tag}_s$s.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "$(date -Is) END   train $tag rc=$rc" | tee -a "$LOG/progress.log"
  return $rc
}

train_pair deepsets_penrew configs/train_deepsets_safe_penrew.yaml results/tuning/deepsets_safe/selected.yaml || exit 1
train_pair attn_penrew_entpen configs/train_attn_safe_penrew_entpen.yaml results/tuning/attn_safe/selected.yaml \
  --set ppo.ent_coef=-0.01 || exit 1

echo "$(date -Is) START evaluate" | tee -a "$LOG/progress.log"
./reproduce.sh --stage evaluate,merge --jobs 10 --seeds 2 > "$LOG/evaluate.log" 2>&1
rc=$?
echo "$(date -Is) END   evaluate rc=$rc" | tee -a "$LOG/progress.log"
[[ $rc -eq 0 ]] && echo "$(date -Is) DONE" | tee -a "$LOG/progress.log"
exit $rc
