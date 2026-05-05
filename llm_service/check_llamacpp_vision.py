from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import requests


DEFAULT_HOST = os.getenv("LITAGENT_VISION_HOST", "127.0.0.1")
DEFAULT_PORT = os.getenv("LITAGENT_VISION_PORT", "8080")
DEFAULT_BASE_URL = os.getenv("LITAGENT_VISION_BASE_URL", f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1")
DEFAULT_API_KEY = os.getenv("LITAGENT_VISION_API_KEY", "llama-cpp")
DEFAULT_EXPECTED_MODEL = os.getenv("LITAGENT_VISION_MODEL_ALIAS", "qwen3.5-9b-vlm")
DEFAULT_TIMEOUT = float(os.getenv("LITAGENT_VISION_TIMEOUT_SECONDS", "30"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the LitAgent llama.cpp vision server.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Bearer token expected by llama.cpp.")
    parser.add_argument("--expected-model", default=DEFAULT_EXPECTED_MODEL, help="Expected VLM model alias.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    return parser.parse_args()


def get_models(base_url: str, api_key: str, timeout: float) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.get(endpoint, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main() -> int:
    args = parse_args()
    try:
        payload = get_models(args.base_url, args.api_key, args.timeout)
    except requests.RequestException as exc:
        print(f"llama.cpp vision health check failed: {exc}", file=sys.stderr)
        print(f"Expected server at {args.base_url.rstrip('/')}/models", file=sys.stderr)
        return 2

    models = payload.get("data", [])
    if not models:
        print("llama.cpp vision health check failed: /models returned no model entries.", file=sys.stderr)
        print(payload, file=sys.stderr)
        return 3

    model_ids = {str(model.get("id") or model.get("root") or "") for model in models}
    if args.expected_model and args.expected_model not in model_ids:
        print("llama.cpp vision health check failed: server is reachable but expected VLM alias is missing.", file=sys.stderr)
        print(f"Expected alias: {args.expected_model}", file=sys.stderr)
        print(f"Returned models: {', '.join(sorted(model_ids))}", file=sys.stderr)
        return 4

    print("llama.cpp vision health check passed.")
    for model in models:
        print(f"- {model.get('id') or model.get('root') or '<unknown>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
