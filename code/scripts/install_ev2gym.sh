#!/usr/bin/env bash
# Clone EV2Gym at the pinned commit (third_party/EV2GYM_COMMIT; no patches) and pip-install it (editable),
# plus the extra runtime dependencies EV2Gym imports but does not declare (ev2gym_notes Sec. 0).
#   bash scripts/install_ev2gym.sh [target_dir]      (default: ../ext/EV2Gym next to this code directory)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT="$(tr -d '[:space:]' < "$HERE/third_party/EV2GYM_COMMIT")"
TARGET="${1:-$HERE/../ext/EV2Gym}"
PIP="${PIP:-python -m pip}"
if [[ ! -d "$TARGET/.git" ]]; then
  git clone https://github.com/StavrosOrf/EV2Gym.git "$TARGET"
fi
git -C "$TARGET" fetch --quiet origin || true
git -C "$TARGET" checkout --quiet "$COMMIT"
HEAD="$(git -C "$TARGET" rev-parse HEAD)"
if [[ "$HEAD" != "$COMMIT" ]]; then echo "EV2Gym checkout failed ($HEAD != $COMMIT)" >&2; exit 1; fi
# EV2Gym's setup.py lists gurobipy; install without dependencies and add the real runtime deps explicitly.
$PIP install --no-deps -e "$TARGET"
$PIP install "pandapower>=2.13" "numba>=0.57" "multicopula>=0.0.2" "psutil>=5.9" "networkx>=3.0" \
             "setuptools<81" "gymnasium>=0.29" "matplotlib>=3.7" "pyyaml>=6.0" "pandas>=2.0"
python -c "import ev2gym, os; print('EV2Gym', ev2gym.__file__)"
echo "EV2Gym installed at commit $COMMIT"
