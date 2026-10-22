#!/usr/bin/env bash
# Optional EV-GNN baseline (DD27; algorithm.md Alg. 5 row 8; v2gsafe/baselines/evgnn_adapter.md).
#   One EV-GNN model (TD3_ActionGNN, its own script, reward and hyperparameters) per arrival profile, trained at
#   N = 20, rho_tr = 4 kW, 5 seeds; zero-shot evaluation on every (N, rho_tr) of that profile through our
#   evaluate.py adapter, raw ('evgnn') and with the safety layer ('evgnn_safe').
# Usage: bash scripts/run_evgnn.sh [--smoke] [--device cpu|cuda] [--seeds 5] [--steps 3000000]
#                                  [--stage train|evaluate|all] [--jobs N]
# Env: EVGNN_DIR (clone location, default ../ext/EV-GNN), EVGNN_SKIP_INSTALL=1 (skip pip installs), PYTHON.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
PYTHON="${PYTHON:-python}"
COMMIT="$(tr -d '[:space:]' < third_party/EVGNN_COMMIT)"
EVGNN_DIR="${EVGNN_DIR:-$HERE/../ext/EV-GNN}"
SMOKE=0; DEVICE=cpu; SEEDS=5; STEPS=3000000; STAGE=all; JOBS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;; --device) DEVICE="$2"; shift 2 ;; --seeds) SEEDS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;; --stage) STAGE="$2"; shift 2 ;; --jobs) JOBS="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline WANDB_SILENT=true
if [[ $SMOKE == 1 ]]; then
  RES="$HERE/results/smoke"; GRID=configs/test_grid_smoke.yaml; SEEDS=1; PROFILES="public workplace"
  TRAIN_ARGS=(--max_timesteps 200 --eval_freq 100 --eval_episodes 1 --start_timesteps 10 --batch_size 8)
  YAML_ARGS=(--N 5 --rho 4 --T 16 --min-stay 30 --hour 14); NTRAIN=5; EVAL_FLAGS=(--smoke)
else
  RES="$HERE/results"; GRID=configs/test_grid.yaml; PROFILES="workplace public private"
  TRAIN_ARGS=(--max_timesteps "$STEPS"); YAML_ARGS=(--N 20 --rho 4); NTRAIN=20; EVAL_FLAGS=()
fi

# ---- clone at the pinned commit (EV-GNN code is not vendored) and install its extra imports
if [[ ! -d "$EVGNN_DIR/.git" ]]; then git clone https://github.com/StavrosOrf/EV-GNN.git "$EVGNN_DIR"; fi
git -C "$EVGNN_DIR" fetch --quiet origin || true
git -C "$EVGNN_DIR" checkout --quiet "$COMMIT"
if [[ "${EVGNN_SKIP_INSTALL:-0}" != 1 ]]; then
  "$PYTHON" -m pip install "torch_geometric>=2.5" wandb tqdm icecream   # do NOT install its requirements.txt pins
fi

model_dir() { (ls -d "$EVGNN_DIR"/saved_models/"$1"-*/ 2>/dev/null || true) | head -1; }

if [[ $STAGE == train || $STAGE == all ]]; then
  for prof in $PROFILES; do
    Y="V2G_ProfixMaxWithLoads_ours_${prof}_N${NTRAIN}_rho4.yaml"   # prefix selects EV-GNN's V2G reward/state
    "$PYTHON" -m v2gsafe.envs.scenarios write-yaml --profile "$prof" "${YAML_ARGS[@]}" --out "$EVGNN_DIR/config_files/$Y"
    for ((s = 0; s < SEEDS; s++)); do
      name="ours_${prof}_N${NTRAIN}_s${s}$([[ $SMOKE == 1 ]] && echo _smoke)"
      d="$(model_dir "$name")"
      if [[ -n "$d" && -f "${d}model.last_actor" ]]; then echo "[evgnn] $name done"; continue; fi
      echo "[evgnn] training $name"
      "$PYTHON" scripts/evgnn_launch.py "$EVGNN_DIR" --policy TD3_ActionGNN --config "$Y" --seed "$s" \
          --device "$DEVICE" --save_model --name "$name" --log_to_wandb "" "${TRAIN_ARGS[@]}"
    done
  done
fi

if [[ $STAGE == evaluate || $STAGE == all ]]; then
  for prof in $PROFILES; do
    IDS="$("$PYTHON" -c "from v2gsafe.envs.scenarios import load_test_grid as g; print(' '.join(s.cfg_id for s, _ in g('$GRID') if s.profile == '$prof' and s.name is None))")"
    for ((s = 0; s < SEEDS; s++)); do
      name="ours_${prof}_N${NTRAIN}_s${s}$([[ $SMOKE == 1 ]] && echo _smoke)"
      d="$(model_dir "$name")"
      [[ -z "$d" ]] && { echo "[evgnn] no model for $name" >&2; exit 1; }
      mname=model.best; [[ -f "${d}model.best_actor" ]] || mname=model.last
      for m in evgnn evgnn_safe; do
        EVGNN_MODEL_NAME="$mname" "$PYTHON" evaluate.py --method "$m" --test-set "$GRID" --only $IDS --seed "$s" \
            --evgnn-repo "$EVGNN_DIR" --evgnn-model-dir "$d" --test-sets-dir "$RES/test_sets" \
            --out-root "$RES/raw" --jobs "$JOBS" "${EVAL_FLAGS[@]}"
      done
    done
  done
fi
