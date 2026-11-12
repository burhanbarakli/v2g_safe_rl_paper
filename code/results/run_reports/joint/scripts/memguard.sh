#!/usr/bin/env bash
# Stops this project's evaluate.py runs (never trainings, never other projects) when MemAvailable < 2 GB, so the OOM
# killer does not hit another project's process on this machine. Evaluations resume from completed parts.
LOG=/home/bob/v2g_logs/joint
while true; do
  a=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
  if (( a < 2048 )); then
    for p in $(pgrep -f '^python evaluate.py --checkpoint'); do
      kill "$p" && echo "$(date -Is) memguard: stopped evaluate.py pid $p (MemAvailable $a MB)" >> "$LOG/progress.log"
    done
  fi
  sleep 5
done
