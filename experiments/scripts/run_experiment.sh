#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_experiment.sh --model {original|genjax} --mode {speed|convergence} \
  --dataset {synthetic|dha|adult} --rows N --cols D --iterations K --n-grid G \
  --log-output /path/to/log.json [--seed S] [--likelihood-interval M] \
  [--num-views V] [--num-clusters C] [--num-categories K]

Options:
  --model               original | genjax
  --mode                speed (skip likelihood) | convergence (periodic likelihood)
  --dataset             synthetic | dha | adult
  --rows                Row count (after subsetting)
  --cols                Column count (synthetic control)
  --iterations          Number of MCMC steps
  --n-grid              Hyperparameter grid resolution (default 100)
  --likelihood-interval Interval (iters) to compute likelihood in convergence mode
  --log-output          JSON log path (schema in experiments/TODO.md)
  --seed                RNG seed (default 0)
  --num-views           (genjax only) override NUM_VIEWS
  --num-clusters        (genjax only) override NUM_CLUSTERS
  --num-categories      (genjax only) override NUM_CATEGORIES
USAGE
}

MODEL=""
MODE=""
DATASET=""
ROWS=""
COLS=""
ITERS=""
N_GRID=100
LL_INTERVAL=10
LOG_OUTPUT=""
SEED=0
NUM_VIEWS=""
NUM_CLUSTERS=""
NUM_CATEGORIES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --rows) ROWS="$2"; shift 2 ;;
    --cols) COLS="$2"; shift 2 ;;
    --iterations) ITERS="$2"; shift 2 ;;
    --n-grid) N_GRID="$2"; shift 2 ;;
    --likelihood-interval) LL_INTERVAL="$2"; shift 2 ;;
    --log-output) LOG_OUTPUT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --num-views) NUM_VIEWS="$2"; shift 2 ;;
    --num-clusters) NUM_CLUSTERS="$2"; shift 2 ;;
    --num-categories) NUM_CATEGORIES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$MODE" || -z "$DATASET" || -z "$ROWS" || -z "$COLS" || -z "$ITERS" || -z "$LOG_OUTPUT" ]]; then
  echo "Missing required arguments" >&2
  usage
  exit 1
fi

mkdir -p "$(dirname "$LOG_OUTPUT")"

if [[ "$MODEL" == "original" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python2.7}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

RUNNER_SCRIPT=""
if [[ "$MODEL" == "genjax" ]]; then
  RUNNER_SCRIPT="${RUNNER_SCRIPT:-experiments/scripts/runner_genjax.py}"
elif [[ "$MODEL" == "original" ]]; then
  RUNNER_SCRIPT="${RUNNER_SCRIPT:-experiments/scripts/runner_original.py}"
fi

if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "Runner script not found: $RUNNER_SCRIPT" >&2
  echo "TODO: implement the runner to produce JSON logs with the shared schema." >&2
  exit 2
fi

args=( \
  --mode "$MODE" \
  --dataset "$DATASET" \
  --rows "$ROWS" \
  --cols "$COLS" \
  --iterations "$ITERS" \
  --n-grid "$N_GRID" \
  --likelihood-interval "$LL_INTERVAL" \
  --log-output "$LOG_OUTPUT" \
  --seed "$SEED" \
)

if [[ "$MODEL" == "genjax" ]]; then
  [[ -n "$NUM_VIEWS" ]] && args+=(--num-views "$NUM_VIEWS")
  [[ -n "$NUM_CLUSTERS" ]] && args+=(--num-clusters "$NUM_CLUSTERS")
  [[ -n "$NUM_CATEGORIES" ]] && args+=(--num-categories "$NUM_CATEGORIES")
fi

exec "$PYTHON_BIN" "$RUNNER_SCRIPT" "${args[@]}"
