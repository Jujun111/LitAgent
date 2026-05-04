# LitAgent Project Status

## Current Working Target

Course-ready prototype plus validated llama.cpp AI microservice on `feat/llm-service`.

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
| Task 2.1: Model Deployment | Done with llama.cpp | `serve_llamacpp.ps1` starts `llama-server.exe` with Qwen3.5-9B Q4_K_M GGUF on CUDA |
| Task 2.2: Prompt Engineering | Scaffolded | `build_prompt()` caps context chunks and formats paper metadata/text |
| Task 2.3: Constrained Decoding | Done for llama.cpp contract | `json_schema` mode sends `response_format` with a llama.cpp-compatible `ResearchDossier` schema |
| Task 2.4: Benchmarking | Done with real local model | llama.cpp benchmark: 3/3 valid JSON, max 9.30s, under 60s |

## Acceptance Checks

Run:

```bash
python smoke_test.py
python -m py_compile api_contracts.py litagent_backend.py litagent_fsm.py streamlit_app.py smoke_test.py llm_service/check_llamacpp.py llm_service/smoke_llamacpp_schema.py
```

Expected smoke-test checks:

- `is_schema_valid == True`
- `orchestrator == "langgraph"`
- `fetch_mode == "mock"`
- `validation_attempts == 1`
- 5x benchmark valid JSON success rate is `100%`
- all mock runs finish under 60 seconds
- llama.cpp schema smoke test passes
- real llama.cpp benchmark has 100% valid JSON

## Live Mode Notes

- Turn off **Use mock papers** in Streamlit to call Semantic Scholar.
- Live API failures are surfaced by default. Turn on **Fallback to mock papers** only for demo fallback behavior.
- Set `SEMANTIC_SCHOLAR_API_KEY` or use the sidebar API key field for authenticated Semantic Scholar requests.
- Latest local live check passed with `fetch_mode=live`, `llm_mode=json_schema`, and 49.35s end-to-end latency.

## Local Model Notes

llama.cpp is the primary local inference server for the Qwen3.5 GGUF target. It loaded `Qwen3.5-9B-Q4_K_M.gguf` on RTX 4060 Laptop 8GB VRAM and served the OpenAI-compatible API at `http://127.0.0.1:8080/v1`.

Default model config:

```text
LITAGENT_LLAMA_MODEL=models/Qwen3.5-9B-Q4_K_M.gguf
LITAGENT_LLAMA_MODEL_ALIAS=qwen3.5-9b-q4km
LITAGENT_LLAMA_CTX_SIZE=4096
LITAGENT_LLAMA_GPU_LAYERS=999
```

Start and validate:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\llm_service\serve_llamacpp.ps1 llm_service\config.llamacpp.example.env
python llm_service/check_llamacpp.py
python llm_service/smoke_llamacpp_schema.py
```

Streamlit preset:

```text
Model provider: llama.cpp local
Base URL: http://127.0.0.1:8080/v1
API key: llama-cpp
Model: qwen3.5-9b-q4km
JSON mode: json_schema
```

vLLM was tested as a fallback but failed on this GGUF with `Unknown gguf model_type: qwen3_5`. Keep the vLLM scripts as experimental reference only.

## Repository Hygiene

- Assignment PDFs remain local and are ignored by `.gitignore`.
- Runtime caches, virtual environments, local env files, and logs are ignored.
- Commit code and project docs only.
