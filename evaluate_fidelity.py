from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from litagent_backend import ResearchDossier, run_pipeline


DEFAULT_BENCHMARK = Path("benchmarks/text_dossier/gold.jsonl")
DEFAULT_OUTPUT = Path("evidence") / f"fidelity_text_dossier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LitAgent text-only extraction fidelity.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK, help="Gold JSONL benchmark path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report path.")
    parser.add_argument(
        "--provider",
        choices=["llama.cpp", "openai-compatible", "mock"],
        default="llama.cpp",
        help="Model provider preset for evaluation.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="llama-cpp", help="OpenAI-compatible bearer token.")
    parser.add_argument("--model", default="qwen3.5-9b-q4km", help="Model name or llama.cpp alias.")
    parser.add_argument("--target-recall", type=float, default=0.95, help="Required fact recall threshold.")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N examples when set.")
    return parser.parse_args()


def normalize_text(value: str) -> str:
    lowered = value.lower()
    ascii_text = lowered.encode("ascii", errors="ignore").decode("ascii")
    no_punctuation = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", no_punctuation).strip()


def load_benchmark(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            required = {"id", "query", "source_papers", "gold_required_facts"}
            missing = sorted(required - set(item))
            if missing:
                raise ValueError(f"{path}:{line_number} missing required fields: {', '.join(missing)}")
            examples.append(item)
            if limit and len(examples) >= limit:
                break
    return examples


def field_coverage(dossier: dict[str, Any] | None) -> float:
    if not dossier:
        return 0.0

    checks: list[bool] = [
        bool(dossier.get("query")),
        bool(dossier.get("topic")),
        bool(dossier.get("summary")),
        bool(dossier.get("key_papers")),
        isinstance(dossier.get("limitations"), list),
        bool(dossier.get("disclaimer")),
    ]

    for paper in dossier.get("key_papers", []):
        checks.extend(
            [
                bool(paper.get("paper_id")),
                bool(paper.get("title")),
                paper.get("year") is not None,
                bool(paper.get("venue")),
                bool(paper.get("reason")),
                bool(paper.get("key_findings")),
            ]
        )

    return sum(1 for item in checks if item) / len(checks) if checks else 0.0


def evaluate_example(example: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    provider = "mock" if args.provider == "mock" else "openai_compatible"
    structured_mode = "json_schema" if provider == "openai_compatible" else "none"
    result = run_pipeline(
        example["query"],
        provider=provider,
        use_mock_fetch=False,
        fixed_papers=example["source_papers"],
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        structured_mode=structured_mode,
        use_langgraph=True,
    )

    dossier = result.get("dossier") if result.get("is_schema_valid") else None
    if dossier is not None:
        ResearchDossier.model_validate(dossier)

    generated_text = normalize_text(json.dumps(dossier or {}, ensure_ascii=False))
    matched_facts: list[str] = []
    missed_facts: list[str] = []
    for fact in example["gold_required_facts"]:
        if normalize_text(fact) in generated_text:
            matched_facts.append(fact)
        else:
            missed_facts.append(fact)

    total_facts = len(example["gold_required_facts"])
    return {
        "id": example["id"],
        "query": example["query"],
        "schema_valid": bool(result.get("is_schema_valid")),
        "fetch_mode": result.get("fetch_mode"),
        "llm_mode": result.get("llm_mode"),
        "orchestrator": result.get("orchestrator"),
        "latency_seconds": result.get("end_to_end_latency_seconds"),
        "llm_latency_seconds": result.get("llm_latency_seconds"),
        "validation_attempts": result.get("validation_attempts"),
        "field_coverage": field_coverage(dossier),
        "matched_facts": matched_facts,
        "missed_facts": missed_facts,
        "required_fact_recall": len(matched_facts) / total_facts if total_facts else 0.0,
        "dossier": dossier,
        "error": result.get("error", ""),
    }


def summarize(records: list[dict[str, Any]], target_recall: float) -> dict[str, Any]:
    total = len(records)
    total_facts = sum(len(item["matched_facts"]) + len(item["missed_facts"]) for item in records)
    matched_facts = sum(len(item["matched_facts"]) for item in records)
    latencies = [item["latency_seconds"] for item in records if item.get("latency_seconds") is not None]
    schema_valid_count = sum(1 for item in records if item["schema_valid"])
    avg_field_coverage = sum(item["field_coverage"] for item in records) / total if total else 0.0
    required_fact_recall = matched_facts / total_facts if total_facts else 0.0

    return {
        "examples": total,
        "total_required_facts": total_facts,
        "matched_required_facts": matched_facts,
        "schema_valid_rate": schema_valid_count / total if total else 0.0,
        "required_fact_recall": required_fact_recall,
        "average_field_coverage": avg_field_coverage,
        "average_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "max_latency_seconds": max(latencies) if latencies else None,
        "target_recall": target_recall,
        "passed": schema_valid_count == total and required_fact_recall >= target_recall,
    }


def main() -> int:
    args = parse_args()
    examples = load_benchmark(args.benchmark, limit=args.limit)
    records = [evaluate_example(example, args) for example in examples]
    summary = summarize(records, args.target_recall)

    report = {
        "benchmark": str(args.benchmark),
        "provider": args.provider,
        "model": args.model,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "records": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote_report={args.output}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
