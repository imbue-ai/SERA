#!/usr/bin/env bash
set -euo pipefail

# Defaults
CONFIG="swebench_vllm.yaml"
REPOS="pallets__flask:flask"
SLICE="0:5"
WORKERS="1"
OUTPUT_DIR="results"
VLLM_URL=""
MODEL_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)      CONFIG="$2";     shift 2 ;;
    --repos)       REPOS="$2";      shift 2 ;;
    --slice)       SLICE="$2";      shift 2 ;;
    --workers)     WORKERS="$2";    shift 2 ;;
    --output)      OUTPUT_DIR="$2"; shift 2 ;;
    --vllm-url)    VLLM_URL="$2";  shift 2 ;;
    --model-name)  MODEL_NAME="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--config FILE] [--repos filter:name,...] [--slice SLICE] [--workers N] [--output DIR] [--vllm-url URL] [--model-name NAME]" >&2
      exit 1
      ;;
  esac
done

# Load .env
set -a
source "$(dirname "$0")/../.env"
set +a

# Map OpenRouter key from .env (stored as OPENAI_API_KEY) to what litellm expects
export OPENROUTER_API_KEY="${OPENAI_API_KEY:-}"

# Global safety limits
export MSWEA_GLOBAL_COST_LIMIT=50.00
export MSWEA_GLOBAL_CALL_LIMIT=5000
export MSWEA_COST_TRACKING=ignore_errors

echo "Running SWE-bench Verified (config=${CONFIG}, slice=${SLICE}, workers=${WORKERS})"

# Build optional config overrides
EXTRA_ARGS=()
if [[ -n "${VLLM_URL}" ]]; then
  EXTRA_ARGS+=("-c" "model.model_kwargs.api_base=${VLLM_URL}")
  echo "Using vLLM endpoint: ${VLLM_URL}"
fi
if [[ -n "${MODEL_NAME}" ]]; then
  EXTRA_ARGS+=("-c" "model.model_name=hosted_vllm/${MODEL_NAME}")
  echo "Using model: ${MODEL_NAME}"
fi

# Build slice args (empty slice = no --slice flag = all instances)
SLICE_ARGS=()
if [[ -n "${SLICE}" ]]; then
  SLICE_ARGS=("--slice" "${SLICE}")
fi

IFS=',' read -ra REPO_LIST <<< "${REPOS}"
for repo_filter in "${REPO_LIST[@]}"; do
  filter="${repo_filter%%:*}"
  name="${repo_filter##*:}"
  echo "=== Running ${name} ==="
  uv run mini-extra swebench \
    --config "${CONFIG}" \
    --subset verified \
    --split test \
    ${SLICE_ARGS[@]+"${SLICE_ARGS[@]}"} \
    --filter "^${filter}" \
    --workers "${WORKERS}" \
    -o "${OUTPUT_DIR}/${name}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
done
