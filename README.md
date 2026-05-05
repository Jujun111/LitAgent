# LitAgent

LitAgent is a course prototype for automated academic literature retrieval and synthesis. It uses a deterministic LangGraph finite state machine (FSM) to control retrieval, prompt construction, local LLM invocation, JSON validation, retry behavior, and dashboard output.

The project is demo-ready with mock data by default and has been validated end-to-end with a local llama.cpp server running the quantized `Qwen3.5-9B-Q4_K_M.gguf` model. llama.cpp is the primary AI microservice because it natively supports GGUF and exposes an OpenAI-compatible API.

## Current Architecture

- `litagent_backend.py` is the canonical implementation. It contains the Semantic Scholar client, rate limiter, LangGraph graph builder, mock LLM, OpenAI-compatible LLM call, Pydantic schema validation, pipeline runner, and benchmark helper.
- `streamlit_app.py` is the dashboard UI for generating dossiers and running the 5x benchmark.
- `litagent_fsm.py` is a small CLI demo that calls the canonical backend in mock LangGraph mode.
- `api_contracts.py` preserves the original single-paper AI contract used during earlier parallel development.
- `smoke_test.py` verifies the mock LangGraph path and benchmark acceptance checks.
- `llm_service/` contains the llama.cpp AI microservice launch scripts, config example, health check, schema-constrained smoke test, and experimental vLLM fallback assets.
- `pdf_ingest.py` optionally parses full-paper PDFs with Docling into text, table, caption, page, and source-reference chunks.

## Requirements Mapping

| Requirement | Implementation |
|---|---|
| Req01: Retrieve up to 10 papers | `SemanticScholarClient.search_papers(... limit=10)` in live mode |
| Req02: FSM orchestration | `build_litagent_graph()` with LangGraph nodes |
| Req03: Prompt AI after retrieval/chunking | `Build_Prompt` then `Call_LLM` graph transition |
| Req04: Strict JSON schema | `ResearchDossier` Pydantic validation |
| Req05: Dashboard dossier | `streamlit_app.py` dossier tab and metrics |
| Req06: Stop after 3 validation retries | conditional edge after `Validate_JSON` |
| NfReq01: 100 requests / 5 minutes | `SemanticScholarRateLimiter` |
| NfReq02: Python + LangGraph | Python implementation using LangGraph |
| NfReq03: 60-second LLM latency | local model HTTP timeout, latency metrics, and llama.cpp benchmark evidence |
| NfReq04: Extraction fidelity target | source-grounded prompts plus schema-constrained JSON; full fidelity evaluation requires a labeled benchmark set |
| NfReq05: AI disclaimer | `AI_DISCLAIMER` rendered in every validated dossier |

## Setup

Python 3.10+ is recommended for the prototype. The local smoke checks in this workspace passed on Python 3.13; optional vLLM deployment may require a narrower Python/CUDA combination.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

Full-paper PDF layout parsing is optional because Docling installs a larger document AI stack:

```bash
python -m pip install -r requirements-pdf.txt
```

On Windows, the parser sets Hugging Face cache flags to avoid symlink permission failures during Docling model downloads.

## Run the Prototype

Run the smoke test:

```bash
python smoke_test.py
```

Run the CLI demo:

```bash
python litagent_fsm.py "graph neural networks for molecular property prediction"
python litagent_fsm.py --benchmark
```

Start the Streamlit dashboard:

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8501
```

## Live Retrieval and Local LLM Options

Mock papers are enabled by default so the course prototype works without external services.

For live Semantic Scholar retrieval, turn off **Use mock papers** in the sidebar. If the API fails, the app reports the error by default. Turn on **Fallback to mock papers** only when you explicitly want demo fallback behavior.

For local model inference, choose an OpenAI-compatible provider in the sidebar and provide the base URL, API key, model name, and JSON mode. llama.cpp is the primary local inference target for GGUF models. vLLM remains documented as an experimental fallback, but it is not the recommended path for the current Qwen3.5 GGUF target.

## llama.cpp AI Microservice

The `feat/llm-service` branch includes a llama.cpp service wrapper for the AI component. It serves an OpenAI-compatible `/v1/chat/completions` endpoint, and the controller sends schema-constrained JSON parameters in this shape:

```json
{
  "response_format": {
    "type": "json_object",
    "schema": "<ResearchDossier JSON schema>"
  }
}
```

Download a Windows CUDA llama.cpp release from ggml-org/llama.cpp and extract it into `tools/llama.cpp`. Place the GGUF model at:

```text
models/Qwen3.5-9B-Q4_K_M.gguf
```

Both `tools/` and `models/` are ignored by git.

Start llama.cpp on Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\llm_service\serve_llamacpp.ps1 llm_service\config.llamacpp.example.env
```

Or start a Linux/WSL build:

```bash
bash llm_service/serve_llamacpp.sh llm_service/config.llamacpp.example.env
```

Default llama.cpp settings:

```text
LITAGENT_LLAMA_MODEL=models/Qwen3.5-9B-Q4_K_M.gguf
LITAGENT_LLAMA_MODEL_ALIAS=qwen3.5-9b-q4km
LITAGENT_LLAMA_CTX_SIZE=4096
LITAGENT_LLAMA_GPU_LAYERS=999
```

Check the running service:

```bash
python llm_service/check_llamacpp.py
python llm_service/smoke_llamacpp_schema.py
```

## Text-Only Fidelity Evaluation

LitAgent evaluates the 95% extraction fidelity target with a frozen text-only benchmark, not live Semantic Scholar retrieval. This keeps the score reproducible without an API key and defines fidelity as required-fact recall over labeled source abstracts.

Run the evaluator with the local llama.cpp server:

```bash
python evaluate_fidelity.py --benchmark benchmarks/text_dossier/gold.jsonl --provider llama.cpp
```

The evaluator reports:

- `schema_valid_rate`: percentage of outputs that validate as `ResearchDossier`
- `required_fact_recall`: matched gold facts divided by total gold facts
- `average_field_coverage`: coverage of required dossier fields
- latency and validation attempts for every sample

The current acceptance target is `schema_valid_rate == 100%` and `required_fact_recall >= 95%`. Evaluation reports are written to ignored `evidence/` files so detailed outputs can be kept locally without committing generated model text.

## Full-Paper Layout Evaluation

The Docling-first full-paper path parses PDFs into text chunks, table markdown, figure/table captions, page numbers, and `source_ref` values before calling the existing LangGraph controller. It does not perform pixel-level chart or figure reasoning; that remains a future multimodal model stage.

Run the parser smoke test and full-paper layout benchmark:

```bash
python smoke_pdf_ingest.py
python evaluate_fidelity.py --benchmark benchmarks/full_paper_layout/gold.jsonl --provider llama.cpp --target-recall 0.90
```

The full-paper benchmark reports category recall for `text`, `table`, and `caption` facts plus `finding_source_trace_rate`, a deterministic sidecar that maps every generated finding back to the closest chunk source.

Validated local result on RTX 4060 Laptop 8GB VRAM:

```text
llama.cpp health check: passed
schema smoke test: passed, 7.61 seconds
LangGraph + mock fetch + real llama.cpp: schema-valid, 10.03 seconds
LangGraph + live Semantic Scholar + real llama.cpp: schema-valid, 49.35 seconds
3-run benchmark: 100% valid JSON, max 9.30 seconds
text-only fidelity eval: 20 examples, 59/60 required facts, 98.3% recall
full-paper layout eval: 3 PDFs, 15/15 required facts, 100% recall
full-paper finding source trace rate: 100%
```

Streamlit preset:

```text
Model provider: llama.cpp local
Base URL: http://127.0.0.1:8080/v1
API key: llama-cpp
Model: qwen3.5-9b-q4km
JSON mode: json_schema
```

## vLLM Fallback Notes

vLLM remains in `llm_service/` as an experimental fallback. It was installed successfully in WSL with CUDA, and CUDA visibility was confirmed. However, the current vLLM GGUF loader failed on the target Qwen3.5 GGUF with:

```text
RuntimeError: Unknown gguf model_type: qwen3_5
```

For this project, llama.cpp is the better primary inference server because it loaded the same GGUF model directly and completed schema-constrained end-to-end validation.

## Current Limitations

- The default demo uses mock abstracts unless live retrieval or PDF input is selected.
- Full-paper support is layout/table/caption understanding, not pixel-level chart or figure reasoning.
- Live Semantic Scholar retrieval may require an API key; unauthenticated requests can return HTTP 429 rate-limit errors.
- Assignment PDF files are intentionally ignored and should remain local unless a private course repository requires them.
