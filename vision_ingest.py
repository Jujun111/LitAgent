from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field, ValidationError


class VisionObservation(BaseModel):
    paper_id: str
    source_ref: str
    page_number: int | None = None
    visual_type: str
    observed_facts: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


def vision_observation_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["paper_id", "source_ref", "page_number", "visual_type", "observed_facts", "uncertainties"],
        "properties": {
            "paper_id": {"type": "string"},
            "source_ref": {"type": "string"},
            "page_number": {"type": ["integer", "null"]},
            "visual_type": {"type": "string"},
            "observed_facts": {
                "type": "array",
                "items": {"type": "string"},
            },
            "uncertainties": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def image_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def build_vision_prompt(query: str, asset: dict[str, Any]) -> str:
    caption = asset.get("caption") or ""
    return f"""
Extract only pixel-level facts that are visible in this scientific figure/table image.
Do not repeat caption-only facts unless they are needed to identify the image.
Prefer concrete comparisons, trends, maxima/minima, labels, table values, colors, and axis relationships.
If a value is unclear, put the uncertainty in uncertainties instead of inventing it.

Research query:
{query}

Image metadata:
paper_id: {asset.get("paper_id", "")}
source_ref: {asset.get("source_ref", "")}
page_number: {asset.get("page_number") or ""}
visual_type: {asset.get("visual_type", "image")}
caption: {caption}

Return JSON only using the requested schema.
""".strip()


def build_mock_observation(query: str, asset: dict[str, Any]) -> VisionObservation:
    seeded_facts = asset.get("expected_visual_facts") or asset.get("gold_visual_facts") or []
    observed = list(seeded_facts)
    if not observed:
        observed = [
            (
                f"Visual evidence from {asset.get('source_ref', 'the image')} was available "
                f"for the query {query}."
            )
        ]
    return VisionObservation(
        paper_id=str(asset.get("paper_id", "")),
        source_ref=str(asset.get("source_ref", "")),
        page_number=asset.get("page_number"),
        visual_type=str(asset.get("visual_type", "image")),
        observed_facts=observed,
        uncertainties=[],
    )


def call_vision_model(
    query: str,
    asset: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    timeout: int = 120,
) -> tuple[VisionObservation, str, float]:
    image_path = asset.get("image_path")
    if not image_path:
        raise ValueError(f"Visual asset is missing image_path: {asset.get('source_ref', '<unknown>')}")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You extract scientific visual evidence and return valid JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_vision_prompt(query, asset)},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            },
        ],
        "temperature": 0,
        "response_format": {
            "type": "json_object",
            "schema": vision_observation_json_schema(),
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    endpoint = base_url.rstrip("/") + "/chat/completions"
    start = time.perf_counter()
    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    raw = data["choices"][0]["message"]["content"].strip()

    try:
        observation = VisionObservation.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError(f"Vision response did not validate for {asset.get('source_ref', '')}: {exc}") from exc

    return observation, raw, elapsed


def extract_vision_observations(
    query: str,
    papers: list[dict[str, Any]],
    provider: str = "openai_compatible",
    base_url: str = "http://127.0.0.1:8080/v1",
    api_key: str = "llama-cpp",
    model: str = "qwen3.5-9b-vlm",
    timeout: int = 120,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []
    total_latency = 0.0

    assets = [
        asset
        for paper in papers
        for asset in paper.get("visual_assets", []) or []
        if asset.get("image_path")
    ]
    if not assets:
        return {
            "vision_observations": [],
            "vision_raw_outputs": [],
            "vision_latency_seconds": 0.0,
            "vision_asset_count": 0,
            "vision_mode": "none",
        }

    for asset in assets:
        if provider == "mock":
            observation = build_mock_observation(query, asset)
            raw = observation.model_dump_json()
            elapsed = 0.0
        else:
            observation, raw, elapsed = call_vision_model(
                query=query,
                asset=asset,
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
            )
        total_latency += elapsed
        observations.append(observation.model_dump())
        raw_outputs.append(
            {
                "source_ref": asset.get("source_ref", ""),
                "raw_output": raw,
                "latency_seconds": elapsed,
            }
        )

    return {
        "vision_observations": observations,
        "vision_raw_outputs": raw_outputs,
        "vision_latency_seconds": total_latency,
        "vision_asset_count": len(assets),
        "vision_mode": provider,
    }


def vision_observations_to_chunks(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for observation in observations:
        facts = [str(item).strip() for item in observation.get("observed_facts", []) if str(item).strip()]
        uncertainties = [
            str(item).strip() for item in observation.get("uncertainties", []) if str(item).strip()
        ]
        if not facts and not uncertainties:
            continue
        text = "\n".join(facts)
        if uncertainties:
            text = text + "\nUncertainties: " + "; ".join(uncertainties)
        chunks.append(
            {
                "paper_id": observation.get("paper_id", ""),
                "title": "",
                "year": None,
                "venue": "",
                "text": text,
                "chunk_type": "vision",
                "page_number": observation.get("page_number"),
                "source_ref": observation.get("source_ref", ""),
            }
        )
    return chunks


def pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)
