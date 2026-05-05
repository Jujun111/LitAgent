from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import requests
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vision_ingest import VisionObservation, call_vision_model


DEFAULT_HOST = os.getenv("LITAGENT_VISION_HOST", "127.0.0.1")
DEFAULT_PORT = os.getenv("LITAGENT_VISION_PORT", "8080")
DEFAULT_BASE_URL = os.getenv("LITAGENT_VISION_BASE_URL", f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/v1")
DEFAULT_API_KEY = os.getenv("LITAGENT_VISION_API_KEY", "llama-cpp")
DEFAULT_MODEL = os.getenv("LITAGENT_VISION_MODEL_ALIAS", "qwen3.5-9b-vlm")
DEFAULT_TIMEOUT = int(os.getenv("LITAGENT_VISION_TIMEOUT_SECONDS", "120"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a schema-constrained llama.cpp vision smoke test.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Bearer token expected by llama.cpp.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Vision model alias served by llama.cpp.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds.")
    return parser.parse_args()


def make_smoke_image(path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for the vision smoke image.") from exc

    image = Image.new("RGB", (320, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 70, 130, 180), fill="blue")
    draw.rectangle((180, 110, 240, 180), fill="orange")
    draw.text((62, 190), "blue", fill="black")
    draw.text((172, 190), "orange", fill="black")
    image.save(path)


def main() -> int:
    args = parse_args()
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "vision_smoke.png"
        make_smoke_image(image_path)
        asset = {
            "paper_id": "vision-smoke",
            "source_ref": "vision_smoke.png#figure-1",
            "page_number": 1,
            "visual_type": "figure",
            "image_path": str(image_path),
            "caption": "Two colored bars are shown.",
        }
        try:
            observation, _raw, elapsed = call_vision_model(
                query="Which bar is taller?",
                asset=asset,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                timeout=args.timeout,
            )
        except requests.RequestException as exc:
            print(f"llama.cpp vision schema smoke test failed: {exc}", file=sys.stderr)
            return 2
        except (KeyError, IndexError, TypeError, RuntimeError, ValueError, ValidationError) as exc:
            print(f"llama.cpp vision schema smoke test failed: {exc}", file=sys.stderr)
            return 3

    VisionObservation.model_validate(observation)
    print("llama.cpp vision schema smoke test passed.")
    print(f"latency_seconds={elapsed:.2f}")
    print(json.dumps(observation.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
