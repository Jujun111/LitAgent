# LitAgent Project Status

## Current Working Target

Course-ready prototype plus validated llama.cpp AI microservice, fidelity evaluation, Docling-first full-paper layout parsing, and optional pixel-level vision extraction.

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
| Task 2.5: Extraction Fidelity | Done for text-only scope | `evaluate_fidelity.py` checks fixed gold abstracts; latest run matched 59/60 required facts |
| Task 2.6: Full-Paper Layout | Done for Docling-first scope | `pdf_ingest.py` parses PDFs into text, table, caption, page, and source-ref chunks |
| Task 2.7: Pixel-Level Vision | Implemented as optional add-on | `vision_ingest.py` calls a llama.cpp multimodal endpoint and converts validated observations into `vision` chunks |

## Acceptance Checks

Run:

```bash
python smoke_test.py
python -m py_compile api_contracts.py litagent_backend.py litagent_fsm.py streamlit_app.py smoke_test.py evaluate_fidelity.py pdf_ingest.py vision_ingest.py smoke_pdf_ingest.py smoke_pixel_vision.py llm_service/check_llamacpp.py llm_service/smoke_llamacpp_schema.py llm_service/check_llamacpp_vision.py llm_service/smoke_llamacpp_vision_schema.py
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
- text-only fidelity eval has `schema_valid_rate == 100%` and `required_fact_recall >= 95%`
- full-paper layout eval has `schema_valid_rate == 100%`, `required_fact_recall >= 90%`, and `finding_source_trace_rate == 100%`
- pixel vision smoke exports PDF crops and produces mock `VisionObservation` chunks

## Fidelity Notes

- Run `python evaluate_fidelity.py --benchmark benchmarks/text_dossier/gold.jsonl --provider llama.cpp`.
- The current frozen text benchmark has 20 examples and 60 required facts.
- Latest local text run passed with `schema_valid_rate=1.0`, `required_fact_recall=0.9833`, and average latency 6.41s.
- Run `python smoke_pdf_ingest.py` to verify Docling PDF parsing on 3 fixture PDFs.
- Run `python evaluate_fidelity.py --benchmark benchmarks/full_paper_layout/gold.jsonl --provider llama.cpp --target-recall 0.90`.
- Latest full-paper layout run passed with `schema_valid_rate=1.0`, `required_fact_recall=1.0`, category recall of 1.0 for text/table/caption, and `finding_source_trace_rate=1.0`.
- This score covers layout/table/caption understanding.
- Run `python smoke_pixel_vision.py` to verify synthetic pixel fixture generation, PDF crop export, and mock vision chunk conversion.
- Run `python evaluate_fidelity.py --benchmark benchmarks/pixel_vision/gold.jsonl --provider llama.cpp --model qwen3.5-9b-vlm --use-vision --vision-provider mock --target-recall 0.80` for an offline pixel-pipeline check.
- When a unified Qwen3.5 VLM llama.cpp server is running on `http://127.0.0.1:8080/v1`, remove `--vision-provider mock` to test real pixel-level extraction.
- Latest real Qwen3.5 VLM pixel run passed with `schema_valid_rate=1.0`, `vision_fact_recall=0.8889`, `finding_source_trace_rate=1.0`, and average latency 25.44s.

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
Model: qwen3.5-9b-vlm
JSON mode: json_schema
```

vLLM was tested as a fallback but failed on this GGUF with `Unknown gguf model_type: qwen3_5`. Keep the vLLM scripts as experimental reference only.

## Pixel Vision Model Notes

The pixel-level route now defaults to one unified Qwen3.5-9B llama.cpp VLM server. Qwen3.5-9B is native multimodal, but llama.cpp still needs a matching `mmproj` GGUF projector for image input. On RTX 4060 Laptop 8GB VRAM, stop the older text-only server before launching the VLM server.

Default vision config:

```text
LITAGENT_VISION_PORT=8080
LITAGENT_VISION_MODEL_ALIAS=qwen3.5-9b-vlm
LITAGENT_VISION_MODEL=models/Qwen3.5-9B-Q4_K_M.gguf
LITAGENT_VISION_MMPROJ=models/mmproj-F16.gguf
```

Start and validate:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\llm_service\serve_llamacpp_vision.ps1 llm_service\config.llamacpp-vision.example.env
python llm_service/check_llamacpp_vision.py
python llm_service/smoke_llamacpp_vision_schema.py
```

## Repository Hygiene

- Assignment PDFs remain local and are ignored by `.gitignore`.
- Runtime caches, virtual environments, local env files, and logs are ignored.
- Commit code and project docs only.
