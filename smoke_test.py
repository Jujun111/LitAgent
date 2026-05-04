from __future__ import annotations

from litagent_backend import benchmark_pipeline, run_pipeline


QUERY = "graph neural networks for molecular property prediction"


def main() -> int:
    result = run_pipeline(
        QUERY,
        provider="mock",
        use_mock_fetch=True,
        use_langgraph=True,
    )

    assert result["is_schema_valid"] is True
    assert result["orchestrator"] == "langgraph"
    assert result["fetch_mode"] == "mock"
    assert result["validation_attempts"] == 1

    benchmark = benchmark_pipeline(
        QUERY,
        runs=5,
        provider="mock",
        use_mock_fetch=True,
        use_langgraph=True,
    )
    assert benchmark["valid_json_success_rate"] == 1
    assert benchmark["acceptance_under_60s"] is True

    print("LitAgent smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
