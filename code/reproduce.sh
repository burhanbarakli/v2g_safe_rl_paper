#!/usr/bin/env bash
# End-to-end reproduction (code_spec.md Sec. 1):
#   (1) build scenarios / test replays, (2) conformal calibrations, [(2b) tuning, optional],
#   (3) train all learned methods x seeds, (4) evaluate all methods on the fixed test sets, (5) oracle LPs,
#   (6) merge episode files, (7) analysis scripts (../analysis/scripts/make_all.py, skipped if missing).
#
# Usage:
#   ./reproduce.sh [--smoke] [--stage <name>[,<name>...]] [--jobs N] [--device cpu|cuda] [--tune]
#                  [--seeds 5] [--no-ablations] [--with-evgnn] [--shard i/M] [--list]
# Stages: scenarios conformal tune tune_select train evaluate evgnn oracle merge analysis
#         (default 'all' = without tune and evgnn; --tune / --with-evgnn add them)
#         sensitivity = optional DD28 stage (sens_scenarios, sens_train, sens_evaluate, sens_oracle, merge);
#         run it after 'train': ./reproduce.sh --stage sensitivity
# Every stage is resumable: completed jobs are skipped when the script is run again.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
ARGS=()
STAGE="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) ARGS+=(--smoke); shift ;;
    --stage) STAGE="$2"; shift 2 ;;
    --jobs) ARGS+=(--jobs "$2"); shift 2 ;;
    --device) ARGS+=(--device "$2"); shift 2 ;;
    --seeds) ARGS+=(--seeds "$2"); shift 2 ;;
    --shard) ARGS+=(--shard "$2"); shift 2 ;;
    --tune) ARGS+=(--tune); shift ;;
    --no-ablations) ARGS+=(--no-ablations); shift ;;
    --with-evgnn) ARGS+=(--with-evgnn); shift ;;
    --list) ARGS+=(--list); shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m v2gsafe.pipeline --stage "$STAGE" "${ARGS[@]}"
