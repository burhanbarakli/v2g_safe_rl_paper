#!/usr/bin/env bash
# End-to-end reproduction (code_spec.md Sec. 1):
#   (1) build scenarios / test replays, (1b) admission tables (DD31), (2) conformal calibrations,
#   (2b) tuning (DD37, equal budget), (3) train all learned methods x seeds, (4) evaluate all methods on the fixed
#   test sets, (5) oracle LPs, (6) merge episode files, (7) analysis scripts (../analysis/scripts/make_all.py).
#
# Usage:
#   ./reproduce.sh [--smoke] [--stage <name>[,<name>...]] [--jobs N] [--device cpu|cuda] [--seeds 5]
#                  [--extended] [--hparams FILE] [--no-tune] [--with-evgnn] [--with-gnndt] [--shard i/M] [--list]
# Stages: scenarios admission conformal tune tune_select train evaluate evgnn gnndt oracle merge analysis
#         ('all' = core run set incl. tuning; --extended adds ablations/w-sensitivity/eMPC-24h/Gurobi eMPC;
#          --with-evgnn / --with-gnndt add the optional baselines). Final training refuses to start without tuning
#          results unless --hparams FILE is given (configs/hparams_defaults.yaml = Alg. 6 defaults, DD37).
#         Composite stages (run after 'train'): sensitivity (DD28/DD39), reference (DD36, attn_safe at N = 50).
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
    --extended) ARGS+=(--extended); shift ;;
    --hparams) ARGS+=(--hparams "$2"); shift 2 ;;
    --no-tune) ARGS+=(--no-tune); shift ;;
    --with-evgnn) ARGS+=(--with-evgnn); shift ;;
    --with-gnndt) ARGS+=(--with-gnndt); shift ;;
    --list) ARGS+=(--list); shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m v2gsafe.pipeline --stage "$STAGE" "${ARGS[@]}"
