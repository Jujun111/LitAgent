from __future__ import annotations

import argparse
import json
import sys

from litagent_backend import benchmark_pipeline, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LitAgent LangGraph FSM with mock data for a quick course demo."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="graph neural networks for molecular property prediction",
        help="Research topic to synthesize.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the 5x mock benchmark instead of a single dossier generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    common_kwargs = {
        "provider": "mock",
        "use_mock_fetch": True,
        "use_langgraph": True,
    }

    if args.benchmark:
        result = benchmark_pipeline(args.query, runs=5, **common_kwargs)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("valid_json_success_rate") == 1 else 1

    result = run_pipeline(args.query, **common_kwargs)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not result.get("is_schema_valid"):
        print("LitAgent mock pipeline did not produce schema-valid JSON.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
