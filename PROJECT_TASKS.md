# LitAgent Project Status

## Current Working Target

Course-ready prototype plus optional vLLM AI microservice on `feat/llm-service`.

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
| Task 2.1: Model Deployment | Service scaffold done | `llm_service/serve_vllm.sh`, `config.example.env`, and `requirements-llm.txt` define the vLLM startup path |
| Task 2.2: Prompt Engineering | Scaffolded | `build_prompt()` caps context chunks and formats paper metadata/text |
| Task 2.3: Constrained Decoding | Done for service contract | vLLM mode sends `structured_outputs` with `ResearchDossier.model_json_schema()`; `smoke_vllm_schema.py` verifies the same contract |
| Task 2.4: Benchmarking | Done for prototype | `benchmark_pipeline()` measures latency, JSON validity, mode, and retry attempts; vLLM smoke test reports latency |

## Acceptance Checks

Run:

```bash
python smoke_test.py
python -m py_compile api_contracts.py litagent_backend.py litagent_fsm.py streamlit_app.py smoke_test.py llm_service/check_vllm.py llm_service/smoke_vllm_schema.py
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

vLLM is the intended course architecture for local Qwen inference, but it is not installed by `requirements.txt` because CUDA and OS compatibility vary. Use `requirements-llm.txt` inside a Linux/WSL CUDA environment.

Default model config:

```text
LITAGENT_VLLM_MODEL=jc-builds/Qwen3.5-9B-Q4_K_M-GGUF:Q4_K_M
LITAGENT_VLLM_TOKENIZER=Qwen/Qwen3.5-9B
LITAGENT_VLLM_MAX_MODEL_LEN=4096
LITAGENT_VLLM_GPU_MEMORY_UTILIZATION=0.85
```

Start and validate:

```bash
cp llm_service/config.example.env llm_service/.env
bash llm_service/serve_vllm.sh llm_service/.env
python llm_service/check_vllm.py
python llm_service/smoke_vllm_schema.py
```

Streamlit preset:

```text
Model provider: vLLM local
Base URL: http://localhost:8000/v1
API key: token-abc123
Model: jc-builds/Qwen3.5-9B-Q4_K_M-GGUF:Q4_K_M
JSON mode: vllm
```

GGUF support in vLLM is useful for memory reduction but still experimental/under-optimized. The detected local GPU is an RTX 4060 Laptop with 8GB VRAM, so the 9B Q4_K_M target may be tight; use a smaller GGUF model by changing the environment variables if needed.

## Repository Hygiene

- Assignment PDFs remain local and are ignored by `.gitignore`.
- Runtime caches, virtual environments, local env files, and logs are ignored.
- Commit code and project docs only.
