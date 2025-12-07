#!/usr/bin/env bash
set -euo pipefail

# Orchestrate experiment sweeps across models/datasets/modes via docker compose.
# By default this script prints the docker compose commands (dry-run).
# Pass --execute to actually run them sequentially.

DRY_RUN=1
COMPOSE_FILE="${COMPOSE_FILE:-docker/docker-compose.yml}"
LOG_ROOT="${LOG_ROOT:-experiments/artifacts/logs}"
CONFIG_FILE="${CONFIG_FILE:-experiments/config/run_all.conf}"
LIKELIHOOD_INTERVAL=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) DRY_RUN=0; shift ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --iter-speed) ITER_SPEED="$2"; shift 2 ;;
    --iter-conv) ITER_CONV="$2"; shift 2 ;;
    --likelihood-interval) LIKELIHOOD_INTERVAL="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: run_all.sh [--execute] [--log-root PATH] [--config PATH] [--iter-speed N] [--iter-conv N] [--likelihood-interval N]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

timestamp() {
  date -u +"%Y%m%dT%H%M%SZ"
}

run_cmd() {
  local cmd="$*"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $cmd"
  else
    echo "[exec] $cmd"
    eval "$cmd"
  fi
}

MODE_LIST=("speed" "convergence")
DATASETS=("synthetic" "dha" "adult")
MODELS=("original" "genjax_cpu" "genjax_gpu")

# Default sweeps; adjust as needed.
ROWS_SYN=("1000")
COLS_SYN=("10" "50")  # default synthetic cols available; adjust if generated
ROWS_DHA=("0")  # placeholder; will auto-fill from manifest if available
COLS_DHA=("0")  # placeholder; will auto-fill from manifest if available
ROWS_ADULT=("100" "1000" "10000" "50000")
COLS_ADULT=("15")
ITER_SPEED=200
ITER_CONV=200
N_GRID=100
# Optionally override defaults from config file if present.
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

# Auto-fill DHA rows/cols from manifest if still unset or left as placeholder.
if [[ "${ROWS_DHA[*]}" == "0" || "${COLS_DHA[*]}" == "0" ]]; then
  manifest_path="experiments/artifacts/processed/dha/manifest.json"
  if [[ -f "$manifest_path" ]]; then
    read -r DHA_ROWS_FROM_MANIFEST DHA_COLS_FROM_MANIFEST < <(python - <<'PY'
import json
path = "experiments/artifacts/processed/dha/manifest.json"
with open(path, "r", encoding="utf-8") as f:
    manifest = json.load(f)
artifacts = manifest.get("artifacts", [])
if not artifacts:
    raise SystemExit("0 0")
art = artifacts[0]
rows = int(art.get("rows", 0))
cols = int(art.get("cols", 0))
print(f"{rows} {cols}")
PY
    )
    if [[ -n "$DHA_ROWS_FROM_MANIFEST" && "$DHA_ROWS_FROM_MANIFEST" -gt 0 ]]; then
      ROWS_DHA=("$DHA_ROWS_FROM_MANIFEST")
    fi
    if [[ -n "$DHA_COLS_FROM_MANIFEST" && "$DHA_COLS_FROM_MANIFEST" -gt 0 ]]; then
      COLS_DHA=("$DHA_COLS_FROM_MANIFEST")
    fi
  fi
fi

for dataset in "${DATASETS[@]}"; do
  case "$dataset" in
    synthetic)
      rows_list=("${ROWS_SYN[@]}")
      cols_list=("${COLS_SYN[@]}")
      ;;
    dha)
      rows_list=("${ROWS_DHA[@]}")
      cols_list=("${COLS_DHA[@]}")
      ;;
    adult)
      rows_list=("${ROWS_ADULT[@]}")
      cols_list=("${COLS_ADULT[@]}")
      ;;
  esac

  for mode in "${MODE_LIST[@]}"; do
    iterations=$ITER_SPEED
    [[ "$mode" == "convergence" ]] && iterations=$ITER_CONV

    for model in "${MODELS[@]}"; do
      # Map compose service names to model flag expected by run_experiment.sh
      model_flag="$model"
      if [[ "$model" == "genjax_cpu" || "$model" == "genjax_gpu" ]]; then
        model_flag="genjax"
      fi

      for rows in "${rows_list[@]}"; do
        for cols in "${cols_list[@]}"; do
          log_file="${LOG_ROOT}/${model}/${dataset}/${mode}_r${rows}_c${cols}_$(timestamp).json"
          cmd="docker compose -f ${COMPOSE_FILE} run --rm ${model} bash experiments/scripts/run_experiment.sh \
            --model ${model_flag} \
            --mode ${mode} \
            --dataset ${dataset} \
            --rows ${rows} \
            --cols ${cols} \
            --iterations ${iterations} \
            --n-grid ${N_GRID} \
            --likelihood-interval ${LIKELIHOOD_INTERVAL} \
            --log-output ${log_file}"
          run_cmd "$cmd"
        done
      done
    done
  done
done
