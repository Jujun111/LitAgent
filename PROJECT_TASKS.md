# LitAgent Project Status

## Current Working Target

Course-ready prototype on `feat/langgraph-fsm`.

Default demo path:

```bash
python smoke_test.py
streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

Open:

```text
http://127.0.0.1:8501
```

## Task Status

| Task | Status | Implementation |
|---|---|---|
| Task 1.1: Semantic Scholar Integration | Done for prototype | `SemanticScholarClient`, `SemanticScholarRateLimiter`, `fetch_data_node` in `litagent_backend.py` retrieve up to 10 live papers |
| Task 1.2: FSM Orchestration | Done | `build_litagent_graph()` uses LangGraph nodes: `Fetch_Data`, `Chunk_Text`, `Build_Prompt`, `Call_LLM`, `Validate_JSON` |
| Task 1.3: Mock AI | Done | `mock_call_llm_node()` returns schema-compliant JSON for demos/tests |
| Task 1.4: Client UI | Done | `streamlit_app.py` generates dossiers, shows metrics, raw JSON, prompt, papers, and disclaimer |
| Task 2.1: Model Deployment | Optional / external | No model server is required for the mock prototype |
| Task 2.2: Prompt Engineering | Scaffolded | `build_prompt()` caps context chunks and formats paper metadata/text |
| Task 2.3: Constrained Decoding | Scaffolded | vLLM mode sends `structured_outputs` with `ResearchDossier.model_json_schema()` |
| Task 2.4: Benchmarking | Done for prototype | `benchmark_pipeline()` measures latency, JSON validity, mode, and retry attempts |

## Acceptance Checks

Run:

```bash
python smoke_test.py
python -m py_compile api_contracts.py litagent_backend.py litagent_fsm.py streamlit_app.py smoke_test.py
```

Expected smoke-test checks:

- `is_schema_valid == True`
- `orchestrator == "langgraph"`
- `fetch_mode == "mock"`
- `validation_attempts == 1`
- 5x benchmark valid JSON success rate is `100%`
- all mock runs finish under 60 seconds

## Live Mode Notes

- Turn off **Use mock papers** in Streamlit to call Semantic Scholar.
- Live API failures are surfaced by default. Turn on **Fallback to mock papers** only for demo fallback behavior.
- Set `SEMANTIC_SCHOLAR_API_KEY` or use the sidebar API key field for authenticated Semantic Scholar requests.

## Local Model Notes

vLLM is the intended course architecture for local Qwen inference, but it is not installed by `requirements.txt` because CUDA and OS compatibility vary.

Example compatible setup:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --api-key token-abc123
```

Streamlit preset:

```text
Model provider: vLLM local
Base URL: http://localhost:8000/v1
API key: token-abc123
JSON mode: vllm
```

## Repository Hygiene

- Assignment PDFs remain local and are ignored by `.gitignore`.
- Runtime caches, virtual environments, local env files, and logs are ignored.
- Commit code and project docs only.
