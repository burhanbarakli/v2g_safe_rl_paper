#!/usr/bin/env bash
# GNN-DT baseline (DD35): STUB. No adapter is implemented (instructions only, see
# v2gsafe/baselines/gnndt_adapter.md). With --clone the repository is cloned/checked out at the pinned commit
# (third_party/GNNDT_COMMIT) for inspection; nothing is trained or evaluated and no result rows are written.
# Usage: bash scripts/run_gnndt.sh [--clone] [--smoke]      Env: GNNDT_DIR (default ../ext/DT4EVs)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT="$(tr -d '[:space:]' < "$HERE/third_party/GNNDT_COMMIT")"
GNNDT_DIR="${GNNDT_DIR:-$HERE/../ext/DT4EVs}"
CLONE=0
for a in "$@"; do [[ "$a" == --clone ]] && CLONE=1; done
if [[ $CLONE == 1 ]]; then
  [[ -d "$GNNDT_DIR/.git" ]] || git clone https://github.com/StavrosOrf/DT4EVs.git "$GNNDT_DIR"
  git -C "$GNNDT_DIR" checkout --quiet "$COMMIT"
  echo "[gnndt] $GNNDT_DIR at $(git -C "$GNNDT_DIR" rev-parse HEAD)"
fi
echo "[gnndt] GNN-DT adapter not implemented (DD35: instructions only). See v2gsafe/baselines/gnndt_adapter.md"
echo "[gnndt] pinned commit: $COMMIT (github.com/StavrosOrf/DT4EVs). No rows written."
exit 0
