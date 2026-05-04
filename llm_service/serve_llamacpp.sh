#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${1:-${LITAGENT_LLAMA_CONFIG:-${SCRIPT_DIR}/config.llamacpp.example.env}}"

if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
fi

: "${LITAGENT_LLAMA_HOST:=127.0.0.1}"
: "${LITAGENT_LLAMA_PORT:=8080}"
: "${LITAGENT_LLAMA_API_KEY:=llama-cpp}"
: "${LITAGENT_LLAMA_MODEL_ALIAS:=qwen3.5-9b-q4km}"
: "${LITAGENT_LLAMA_SERVER:=tools/llama.cpp/llama-server}"
: "${LITAGENT_LLAMA_MODEL:=models/Qwen3.5-9B-Q4_K_M.gguf}"
: "${LITAGENT_LLAMA_CTX_SIZE:=4096}"
: "${LITAGENT_LLAMA_GPU_LAYERS:=999}"
: "${LITAGENT_LLAMA_THREADS:=8}"
: "${LITAGENT_LLAMA_EXTRA_ARGS:=}"

[[ "$LITAGENT_LLAMA_SERVER" = /* ]] || LITAGENT_LLAMA_SERVER="${REPO_ROOT}/${LITAGENT_LLAMA_SERVER}"
[[ "$LITAGENT_LLAMA_MODEL" = /* ]] || LITAGENT_LLAMA_MODEL="${REPO_ROOT}/${LITAGENT_LLAMA_MODEL}"

if [[ ! -x "$LITAGENT_LLAMA_SERVER" ]]; then
  echo "llama-server executable not found or not executable: ${LITAGENT_LLAMA_SERVER}" >&2
  echo "Install a llama.cpp Linux CUDA build or use the Windows PowerShell launcher." >&2
  exit 127
fi
if [[ ! -f "$LITAGENT_LLAMA_MODEL" ]]; then
  echo "GGUF model not found: ${LITAGENT_LLAMA_MODEL}" >&2
  exit 2
fi

echo "Starting LitAgent llama.cpp microservice"
echo "  server:   ${LITAGENT_LLAMA_SERVER}"
echo "  model:    ${LITAGENT_LLAMA_MODEL}"
echo "  alias:    ${LITAGENT_LLAMA_MODEL_ALIAS}"
echo "  endpoint: http://${LITAGENT_LLAMA_HOST}:${LITAGENT_LLAMA_PORT}/v1"

# LITAGENT_LLAMA_EXTRA_ARGS is intentionally word-split so users can pass normal CLI flags.
# shellcheck disable=SC2086
exec "${LITAGENT_LLAMA_SERVER}" \
  --host "${LITAGENT_LLAMA_HOST}" \
  --port "${LITAGENT_LLAMA_PORT}" \
  --api-key "${LITAGENT_LLAMA_API_KEY}" \
  --model "${LITAGENT_LLAMA_MODEL}" \
  --alias "${LITAGENT_LLAMA_MODEL_ALIAS}" \
  --ctx-size "${LITAGENT_LLAMA_CTX_SIZE}" \
  --n-gpu-layers "${LITAGENT_LLAMA_GPU_LAYERS}" \
  --threads "${LITAGENT_LLAMA_THREADS}" \
  --jinja \
  ${LITAGENT_LLAMA_EXTRA_ARGS}
