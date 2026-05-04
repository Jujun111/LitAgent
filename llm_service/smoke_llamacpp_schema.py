from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from litagent_backend import AI_DISCLAIMER, ResearchDossier, research_dossier_json_schema


DEFAULT_HOST = os.getenv("LITAGENT_LLAMA_HOST", "127.0.0.1")
DEFAULT_PORT = os.getenv("LITAGENT_LLAMA_PORT", "8080")
DEFAULT_BASE_URL = os.getenv("LITAGENT_LLAMA_BASE_URL", f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1")
DEFAULT_API_KEY = os.getenv("LITAGENT_LLAMA_API_KEY", "llama-cpp")
DEFAULT_MODEL = os.getenv("LITAGENT_LLAMA_MODEL_ALIAS", "qwen3.5-9b-q4km")
DEFAULT_TIMEOUT = float(os.getenv("LITAGENT_LLAMA_TIMEOUT_SECONDS", "120"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a schema-constrained LitAgent llama.cpp smoke test.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Bearer token expected by llama.cpp.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model alias served by llama.cpp.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    return parser.parse_args()


def build_payload(model: str) -> dict[str, Any]:
    prompt = f"""
Create a concise research dossier from the context.
Return JSON only. No markdown. No extra commentary.

User query:
graph neural networks for molecular property prediction

Context:
[Chunk 1]
paper_id: smoke-001
title: Graph Neural Networks for Molecular Property Prediction
year: 2024
venue: LitAgent Smoke Test
text: Graph neural networks encode atoms and bonds as graphs and use message passing to predict molecular properties. Common evaluations compare graph convolution, graph attention, and descriptor baselines.

Required JSON fields:
- query
- topic
- summary
- key_papers: paper_id, title, year, venue, reason, key_findings
- limitations
- disclaimer

Use this disclaimer exactly:
{AI_DISCLAIMER}
""".strip()

    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a research dossier extraction assistant. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_object",
            "schema": research_dossier_json_schema(),
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }


def request_completion(base_url: str, api_key: str, payload: dict[str, Any], timeout: float) -> tuple[str, float]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.perf_counter()
    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip(), elapsed


def main() -> int:
    args = parse_args()
    try:
        content, elapsed = request_completion(
            args.base_url,
            args.api_key,
            build_payload(args.model),
            args.timeout,
        )
    except requests.RequestException as exc:
        print(f"llama.cpp schema smoke test failed: {exc}", file=sys.stderr)
        return 2
    except (KeyError, IndexError, TypeError) as exc:
        print(f"llama.cpp schema smoke test failed: unexpected response shape: {exc}", file=sys.stderr)
        return 3

    try:
        dossier = ResearchDossier.model_validate_json(content)
    except ValidationError as exc:
        print("llama.cpp schema smoke test failed: response did not validate as ResearchDossier.", file=sys.stderr)
        print(content, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 4

    print("llama.cpp schema smoke test passed.")
    print(f"latency_seconds={elapsed:.2f}")
    print(json.dumps(dossier.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
