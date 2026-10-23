#!/usr/bin/env bash
# Core run set (tuning included), resumable: every phase skips jobs that are already complete.
# Phases use different parallelism: CPU-bound stages run many jobs, GPU training runs 2 jobs
# (each starts 16 EV2Gym processes). A failing phase stops the run; restart the service to resume.
set -uo pipefail
ROOT=/home/bob/v2g_safe_rl_paper
LOG=/home/bob/v2g_logs/core
DONE=$LOG/CORE_DONE
CPU_JOBS=${CPU_JOBS:-10}
GPU_JOBS=${GPU_JOBS:-2}
SEEDS=${SEEDS:-2}   # seeds per learned method (author decision 2026-10-22: at most 2)
mkdir -p "$LOG"
if [[ -f $DONE ]]; then echo "core set already complete ($DONE)"; exit 0; fi
source "$ROOT/venv/bin/activate"
cd "$ROOT/code"

phase() {  # phase <name> <reproduce.sh args...>
  local name=$1; shift
  echo "$(date -Is) START $name: $*" | tee -a "$LOG/progress.log"
  ./reproduce.sh "$@" >> "$LOG/$name.log" 2>&1
  local rc=$?
  echo "$(date -Is) END   $name rc=$rc" | tee -a "$LOG/progress.log"
  if [[ $rc -ne 0 ]]; then
    echo "$(date -Is) phase $name failed, see $LOG/$name.log" | tee -a "$LOG/progress.log"
    exit "$rc"
  fi
}

phase 1_prep     --stage scenarios,admission,conformal --jobs "$CPU_JOBS" --seeds "$SEEDS"
phase 2_tune     --stage tune,tune_select --jobs "$GPU_JOBS" --device cuda --seeds "$SEEDS"
phase 3_train    --stage train --jobs "$GPU_JOBS" --device cuda --seeds "$SEEDS"
phase 4_evaluate --stage evaluate,oracle,merge --jobs "$CPU_JOBS" --seeds "$SEEDS"
phase 5_analysis --stage analysis --seeds "$SEEDS"
touch "$DONE"
echo "$(date -Is) CORE SET COMPLETE" | tee -a "$LOG/progress.log"
