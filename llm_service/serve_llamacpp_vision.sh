#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${1:-${LITAGENT_VISION_CONFIG:-${SCRIPT_DIR}/config.llamacpp-vision.example.env}}"

if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
fi

: "${LITAGENT_VISION_HOST:=127.0.0.1}"
: "${LITAGENT_VISION_PORT:=8080}"
: "${LITAGENT_VISION_API_KEY:=llama-cpp}"
: "${LITAGENT_VISION_MODEL_ALIAS:=qwen3.5-9b-vlm}"
: "${LITAGENT_VISION_SERVER:=tools/llama.cpp/llama-server}"
: "${LITAGENT_VISION_HF_REPO:=}"
: "${LITAGENT_VISION_MODEL:=models/Qwen3.5-9B-Q4_K_M.gguf}"
: "${LITAGENT_VISION_MMPROJ:=models/mmproj-F16.gguf}"
: "${LITAGENT_VISION_CTX_SIZE:=4096}"
: "${LITAGENT_VISION_GPU_LAYERS:=999}"
: "${LITAGENT_VISION_THREADS:=8}"
: "${LITAGENT_VISION_IMAGE_MAX_TOKENS:=1024}"
: "${LITAGENT_VISION_EXTRA_ARGS:=}"

[[ "$LITAGENT_VISION_SERVER" = /* ]] || LITAGENT_VISION_SERVER="${REPO_ROOT}/${LITAGENT_VISION_SERVER}"

if [[ ! -x "$LITAGENT_VISION_SERVER" ]]; then
  echo "llama-server executable not found or not executable: ${LITAGENT_VISION_SERVER}" >&2
  echo "Install a llama.cpp Linux CUDA build or use the Windows PowerShell launcher." >&2
  exit 127
fi

args=(
  --host "${LITAGENT_VISION_HOST}"
  --port "${LITAGENT_VISION_PORT}"
  --api-key "${LITAGENT_VISION_API_KEY}"
  --alias "${LITAGENT_VISION_MODEL_ALIAS}"
  --ctx-size "${LITAGENT_VISION_CTX_SIZE}"
  --n-gpu-layers "${LITAGENT_VISION_GPU_LAYERS}"
  --threads "${LITAGENT_VISION_THREADS}"
  --image-max-tokens "${LITAGENT_VISION_IMAGE_MAX_TOKENS}"
  --jinja
)

if [[ -n "$LITAGENT_VISION_HF_REPO" ]]; then
  args+=(--hf-repo "$LITAGENT_VISION_HF_REPO")
else
  [[ "$LITAGENT_VISION_MODEL" = /* ]] || LITAGENT_VISION_MODEL="${REPO_ROOT}/${LITAGENT_VISION_MODEL}"
  if [[ ! -f "$LITAGENT_VISION_MODEL" ]]; then
    echo "Vision GGUF model not found: ${LITAGENT_VISION_MODEL}" >&2
    echo "Download Qwen3.5-9B-Q4_K_M.gguf into models/ or set LITAGENT_VISION_HF_REPO." >&2
    exit 2
  fi
  args+=(--model "$LITAGENT_VISION_MODEL")
  if [[ -n "$LITAGENT_VISION_MMPROJ" ]]; then
    [[ "$LITAGENT_VISION_MMPROJ" = /* ]] || LITAGENT_VISION_MMPROJ="${REPO_ROOT}/${LITAGENT_VISION_MMPROJ}"
    if [[ ! -f "$LITAGENT_VISION_MMPROJ" ]]; then
      echo "Vision mmproj file not found: ${LITAGENT_VISION_MMPROJ}" >&2
      echo "Download mmproj-F16.gguf from a Qwen3.5-9B VLM GGUF repo, or set LITAGENT_VISION_HF_REPO." >&2
      exit 2
    fi
    args+=(--mmproj "$LITAGENT_VISION_MMPROJ")
  fi
fi

echo "Starting LitAgent unified llama.cpp Qwen3.5 VLM microservice"
echo "  server:   ${LITAGENT_VISION_SERVER}"
echo "  hf_repo:  ${LITAGENT_VISION_HF_REPO}"
echo "  alias:    ${LITAGENT_VISION_MODEL_ALIAS}"
echo "  endpoint: http://${LITAGENT_VISION_HOST}:${LITAGENT_VISION_PORT}/v1"
echo "Tip: this server can serve both text synthesis and pixel vision; do not run a second text server on 8GB VRAM."

# shellcheck disable=SC2086
exec "${LITAGENT_VISION_SERVER}" "${args[@]}" ${LITAGENT_VISION_EXTRA_ARGS}
