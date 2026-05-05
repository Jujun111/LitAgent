from __future__ import annotations

import argparse
import json
import re
import runpy
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
    parser.add_argument("--use-vision", action="store_true", help="Enable pixel-level vision extraction.")
    parser.add_argument(
        "--vision-provider",
        choices=["openai-compatible", "mock"],
        default="openai-compatible",
        help="Vision model provider preset.",
    )
    parser.add_argument(
        "--vision-base-url",
        default="",
        help="Vision OpenAI-compatible base URL. Defaults to --base-url for unified Qwen3.5 VLM serving.",
    )
    parser.add_argument("--vision-api-key", default="", help="Defaults to --api-key when omitted.")
    parser.add_argument("--vision-model", default="", help="Defaults to --model when omitted.")
    parser.add_argument("--vision-timeout", type=int, default=120)
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
            required = {"id", "query", "gold_required_facts"}
            missing = sorted(required - set(item))
            if missing:
                raise ValueError(f"{path}:{line_number} missing required fields: {', '.join(missing)}")
            if "source_papers" not in item and "source_pdf" not in item and "source_pdfs" not in item:
                raise ValueError(f"{path}:{line_number} must define source_papers, source_pdf, or source_pdfs")
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


def resolve_benchmark_path(value: str, benchmark_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    from_benchmark_dir = benchmark_path.parent / candidate
    if from_benchmark_dir.exists():
        return from_benchmark_dir
    return candidate


def ensure_generated_fixture(path: Path, benchmark_path: Path) -> None:
    if path.exists():
        return
    generator = benchmark_path.parent / "make_fixtures.py"
    if not generator.exists():
        return
    namespace = runpy.run_path(str(generator))
    ensure_fixtures = namespace.get("ensure_fixtures")
    if callable(ensure_fixtures):
        ensure_fixtures(benchmark_path.parent)


def load_example_papers(
    example: dict[str, Any],
    benchmark_path: Path,
    export_visuals: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    if "source_papers" in example:
        return example["source_papers"], "fixed"

    from pdf_ingest import parse_pdf_to_paper_record

    pdf_values = example.get("source_pdfs") or [example["source_pdf"]]
    papers: list[dict[str, Any]] = []
    for pdf_value in pdf_values:
        pdf_path = resolve_benchmark_path(pdf_value, benchmark_path)
        ensure_generated_fixture(pdf_path, benchmark_path)
        paper = parse_pdf_to_paper_record(
            pdf_path,
            export_visuals=export_visuals,
            visual_output_dir=Path("evidence/visual_crops") / example["id"],
            include_page_images=export_visuals,
        )
        if export_visuals:
            expected_facts = [
                fact_text(fact)
                for fact in example.get("gold_required_facts", [])
                if fact_category(fact) == "vision"
            ]
            for asset in paper.get("visual_assets", []) or []:
                asset["expected_visual_facts"] = expected_facts
        papers.append(paper)
    return papers, "pdf_vision_fixed" if export_visuals else "pdf_fixed"


def fact_text(fact: str | dict[str, Any]) -> str:
    if isinstance(fact, dict):
        return str(fact.get("text", ""))
    return str(fact)


def fact_aliases(fact: str | dict[str, Any]) -> list[str]:
    text = fact_text(fact)
    aliases = [text]
    if isinstance(fact, dict):
        aliases.extend(str(alias) for alias in fact.get("aliases", []) if str(alias).strip())
    return aliases


def fact_category(fact: str | dict[str, Any]) -> str:
    if isinstance(fact, dict):
        return str(fact.get("category", "text"))
    return "text"


def evaluate_example(example: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    provider = "mock" if args.provider == "mock" else "openai_compatible"
    structured_mode = "json_schema" if provider == "openai_compatible" else "none"
    fixed_papers, fixed_fetch_mode = load_example_papers(example, args.benchmark, export_visuals=args.use_vision)
    result = run_pipeline(
        example["query"],
        provider=provider,
        use_mock_fetch=False,
        fixed_papers=fixed_papers,
        fixed_fetch_mode=fixed_fetch_mode,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        structured_mode=structured_mode,
        use_langgraph=True,
        use_vision=args.use_vision,
        vision_provider="mock" if args.vision_provider == "mock" else "openai_compatible",
        vision_base_url=args.vision_base_url or args.base_url,
        vision_api_key=args.vision_api_key or args.api_key,
        vision_model=args.vision_model or args.model,
        vision_timeout=args.vision_timeout,
    )

    dossier = result.get("dossier") if result.get("is_schema_valid") else None
    if dossier is not None:
        ResearchDossier.model_validate(dossier)

    generated_text = normalize_text(json.dumps(dossier or {}, ensure_ascii=False))
    matched_facts: list[str] = []
    missed_facts: list[str] = []
    category_counts: dict[str, dict[str, int]] = {}
    for fact in example["gold_required_facts"]:
        text = fact_text(fact)
        category = fact_category(fact)
        category_counts.setdefault(category, {"matched": 0, "total": 0})
        category_counts[category]["total"] += 1
        if any(normalize_text(alias) in generated_text for alias in fact_aliases(fact)):
            matched_facts.append(text)
            category_counts[category]["matched"] += 1
        else:
            missed_facts.append(text)

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
        "vision_mode": result.get("vision_mode", "disabled"),
        "vision_asset_count": result.get("vision_asset_count", 0),
        "vision_latency_seconds": result.get("vision_latency_seconds", 0.0),
        "vision_observations": result.get("vision_observations", []),
        "field_coverage": field_coverage(dossier),
        "finding_sources": result.get("finding_sources", []),
        "matched_facts": matched_facts,
        "missed_facts": missed_facts,
        "category_counts": category_counts,
        "required_fact_recall": len(matched_facts) / total_facts if total_facts else 0.0,
        "dossier": dossier,
        "error": result.get("error", ""),
    }


def summarize(records: list[dict[str, Any]], target_recall: float) -> dict[str, Any]:
    total = len(records)
    total_facts = sum(len(item["matched_facts"]) + len(item["missed_facts"]) for item in records)
    matched_facts = sum(len(item["matched_facts"]) for item in records)
    category_counts: dict[str, dict[str, int]] = {}
    for item in records:
        for category, counts in item.get("category_counts", {}).items():
            category_counts.setdefault(category, {"matched": 0, "total": 0})
            category_counts[category]["matched"] += counts.get("matched", 0)
            category_counts[category]["total"] += counts.get("total", 0)
    latencies = [item["latency_seconds"] for item in records if item.get("latency_seconds") is not None]
    schema_valid_count = sum(1 for item in records if item["schema_valid"])
    finding_source_total = sum(len(item.get("finding_sources", [])) for item in records)
    finding_source_traced = sum(
        1
        for item in records
        for source in item.get("finding_sources", [])
        if source.get("source_ref")
    )
    avg_field_coverage = sum(item["field_coverage"] for item in records) / total if total else 0.0
    required_fact_recall = matched_facts / total_facts if total_facts else 0.0

    return {
        "examples": total,
        "total_required_facts": total_facts,
        "matched_required_facts": matched_facts,
        "schema_valid_rate": schema_valid_count / total if total else 0.0,
        "required_fact_recall": required_fact_recall,
        "category_recall": {
            category: counts["matched"] / counts["total"] if counts["total"] else 0.0
            for category, counts in sorted(category_counts.items())
        },
        "vision_fact_recall": (
            category_counts.get("vision", {}).get("matched", 0) / category_counts.get("vision", {}).get("total", 1)
            if category_counts.get("vision", {}).get("total", 0)
            else None
        ),
        "finding_source_trace_rate": (
            finding_source_traced / finding_source_total if finding_source_total else 0.0
        ),
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
