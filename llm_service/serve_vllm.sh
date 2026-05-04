#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${1:-${LITAGENT_VLLM_CONFIG:-${SCRIPT_DIR}/config.example.env}}"

if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
fi

: "${LITAGENT_VLLM_HOST:=127.0.0.1}"
: "${LITAGENT_VLLM_PORT:=8000}"
: "${LITAGENT_VLLM_API_KEY:=token-abc123}"
: "${LITAGENT_VLLM_MODEL:=jc-builds/Qwen3.5-9B-Q4_K_M-GGUF:Q4_K_M}"
: "${LITAGENT_VLLM_TOKENIZER:=Qwen/Qwen3.5-9B}"
: "${LITAGENT_VLLM_HF_CONFIG_PATH:=Qwen/Qwen3.5-9B}"
: "${LITAGENT_VLLM_MAX_MODEL_LEN:=4096}"
: "${LITAGENT_VLLM_GPU_MEMORY_UTILIZATION:=0.85}"
: "${LITAGENT_VLLM_DTYPE:=auto}"
: "${LITAGENT_VLLM_EXTRA_ARGS:=}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vLLM is not installed in this environment." >&2
  echo "Create a Linux/WSL CUDA environment, then run:" >&2
  echo "  python -m pip install -r ${REPO_ROOT}/requirements-llm.txt" >&2
  exit 127
fi

echo "Starting LitAgent vLLM microservice"
echo "  model:     ${LITAGENT_VLLM_MODEL}"
echo "  tokenizer: ${LITAGENT_VLLM_TOKENIZER}"
echo "  hf config: ${LITAGENT_VLLM_HF_CONFIG_PATH}"
echo "  endpoint:  http://${LITAGENT_VLLM_HOST}:${LITAGENT_VLLM_PORT}/v1"

# LITAGENT_VLLM_EXTRA_ARGS is intentionally word-split so users can pass normal CLI flags.
# shellcheck disable=SC2086
exec vllm serve "${LITAGENT_VLLM_MODEL}" \
  --tokenizer "${LITAGENT_VLLM_TOKENIZER}" \
  --host "${LITAGENT_VLLM_HOST}" \
  --port "${LITAGENT_VLLM_PORT}" \
  --api-key "${LITAGENT_VLLM_API_KEY}" \
  --hf-config-path "${LITAGENT_VLLM_HF_CONFIG_PATH}" \
  --max-model-len "${LITAGENT_VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${LITAGENT_VLLM_GPU_MEMORY_UTILIZATION}" \
  --dtype "${LITAGENT_VLLM_DTYPE}" \
  ${LITAGENT_VLLM_EXTRA_ARGS}
