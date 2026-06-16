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
- `vision_ingest.py` optionally calls a llama.cpp multimodal endpoint, validates pixel-level visual observations, and converts them into `vision` evidence chunks.

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

There are two demo paths:

- Offline UI demo: no model server, no Semantic Scholar API key.
- Local LLM demo: requires a separate llama.cpp server on `http://127.0.0.1:8080/v1`.

Run the smoke test first:

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

For the offline UI demo, select:

```text
Model provider: Mock
Use mock papers: on
Use PDF paper: off
```

For the local LLM demo, start llama.cpp first in a second terminal, then select:

```text
Model provider: llama.cpp local
Base URL: http://127.0.0.1:8080/v1
API key: llama-cpp
Model: qwen3.5-9b-vlm
JSON mode: json_schema
```

If `Generate dossier` reports `Failed to establish a new connection` for port `8080`, llama.cpp is not running yet.

If `Generate dossier` reports `EOF while parsing a string`, the local model output was likely truncated before JSON finished. The app now sends a larger output budget by default and asks the model for a shorter dossier. Keep `Max output tokens` at `4096` or higher for live retrieval, and reduce `Context chunks` if you still see truncation on long topics.

Pixel-level figure/table understanding: optional llama.cpp vision service extracts visual facts from PDF crops before text synthesis

## Live Retrieval and Local LLM Options

Mock papers are enabled by default so the course prototype works without external services.

For live Semantic Scholar retrieval, turn off **Use mock papers** in the sidebar. If the API fails, the app reports the error by default. Turn on **Fallback to mock papers** only when you explicitly want demo fallback behavior.

For local model inference, choose an OpenAI-compatible provider in the sidebar and provide the base URL, API key, model name, and JSON mode. llama.cpp is the primary local inference target for GGUF models. vLLM remains documented as an experimental fallback, but it is not the recommended path for the current Qwen3.5 GGUF target.

## llama.cpp AI Microservice

The recommended local demo uses a unified Qwen3.5-9B VLM llama.cpp server. It serves an OpenAI-compatible `/v1/chat/completions` endpoint on `http://127.0.0.1:8080/v1`, and the controller sends schema-constrained JSON parameters in this shape:

```json
{
  "response_format": {
    "type": "json_object",
    "schema": "<ResearchDossier JSON schema>"
  }
}
```

Download a Windows CUDA llama.cpp release from [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and extract it into `tools/llama.cpp`. The repo should then contain:

```text
tools/llama.cpp/llama-server.exe
```

Create `models/` and place the Qwen3.5 model plus projector there:

```text
models/Qwen3.5-9B-Q4_K_M.gguf
models/mmproj-F16.gguf
```

Both `tools/` and `models/` are ignored by git. The model and projector can be downloaded manually from a Qwen3.5 VLM GGUF repo such as [jc-builds/Qwen3.5-9B-VLM-Q4_K_M-GGUF](https://huggingface.co/jc-builds/Qwen3.5-9B-VLM-Q4_K_M-GGUF). On PowerShell, this is the direct download shape:

```powershell
New-Item -ItemType Directory -Force models
$ProgressPreference = "SilentlyContinue"
Invoke-WebRequest `
  -Uri "https://huggingface.co/jc-builds/Qwen3.5-9B-VLM-Q4_K_M-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf" `
  -OutFile "models/Qwen3.5-9B-Q4_K_M.gguf"
Invoke-WebRequest `
  -Uri "https://huggingface.co/jc-builds/Qwen3.5-9B-VLM-Q4_K_M-GGUF/resolve/main/mmproj-F16.gguf" `
  -OutFile "models/mmproj-F16.gguf"
```

Start the unified Qwen3.5 VLM server on Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\llm_service\serve_llamacpp_vision.ps1 llm_service\config.llamacpp-vision.example.env
```

Or start a Linux/WSL build:

```bash
bash llm_service/serve_llamacpp_vision.sh llm_service/config.llamacpp-vision.example.env
```

Default llama.cpp VLM settings:

```text
LITAGENT_VISION_MODEL=models/Qwen3.5-9B-Q4_K_M.gguf
LITAGENT_VISION_MMPROJ=models/mmproj-F16.gguf
LITAGENT_VISION_MODEL_ALIAS=qwen3.5-9b-vlm
LITAGENT_VISION_CTX_SIZE=4096
LITAGENT_VISION_GPU_LAYERS=999
```

Check the running service:

```bash
python llm_service/check_llamacpp_vision.py
python llm_service/smoke_llamacpp_schema.py --model qwen3.5-9b-vlm
python llm_service/smoke_llamacpp_vision_schema.py
```

Keep this terminal open while Streamlit is running. Close it, or press `Ctrl+C`, when the demo is done.

For classroom demos with live Semantic Scholar retrieval, use these Streamlit defaults:

```text
Model provider: llama.cpp local
Use mock papers: off
Use PDF paper: off
Context chunks: 8
Max output tokens: 4096
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

The Docling-first full-paper path parses PDFs into text chunks, table markdown, figure/table captions, page numbers, and `source_ref` values before calling the existing LangGraph controller. This layout path is still useful without a vision model because many table and figure claims are present in extracted text/captions.

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
Model: qwen3.5-9b-vlm
JSON mode: json_schema
```

## Pixel-Level Vision Add-On

Pixel-level figure/table understanding now defaults to a unified Qwen3.5-9B llama.cpp VLM server on port `8080`. Qwen3.5-9B is a native multimodal model, but llama.cpp still expects the vision projector to be loaded as a separate `mmproj` GGUF file. With `mmproj-F16.gguf` loaded, the same server can handle text dossier synthesis and image crop extraction.

The official Qwen3.5-9B model card lists the model as image-text-to-text and documents image input through chat messages: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). llama.cpp documents multimodal support through `llama-server` and the OpenAI-compatible `/chat/completions` API, using either `--hf-repo` or a local `--model` plus `--mmproj` projector: [llama.cpp multimodal docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).

Default unified VLM config:

```text
LITAGENT_VISION_PORT=8080
LITAGENT_VISION_MODEL_ALIAS=qwen3.5-9b-vlm
LITAGENT_VISION_MODEL=models/Qwen3.5-9B-Q4_K_M.gguf
LITAGENT_VISION_MMPROJ=models/mmproj-F16.gguf
```

Download the projector from a Qwen3.5-9B VLM GGUF repo, for example `jc-builds/Qwen3.5-9B-VLM-Q4_K_M-GGUF` or `bartowski/Qwen_Qwen3.5-9B-GGUF`. If you prefer llama.cpp to auto-download model and projector together, set `LITAGENT_VISION_HF_REPO=jc-builds/Qwen3.5-9B-VLM-Q4_K_M-GGUF` and leave local model/mmproj paths empty.

Start the unified VLM server. On 8GB VRAM, stop any text-only llama.cpp server first:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\llm_service\serve_llamacpp_vision.ps1 llm_service\config.llamacpp-vision.example.env
```

Linux/WSL:

```bash
bash llm_service/serve_llamacpp_vision.sh llm_service/config.llamacpp-vision.example.env
```

Validate the vision service and crop pipeline:

```bash
python llm_service/check_llamacpp_vision.py
python llm_service/smoke_llamacpp_vision_schema.py
python smoke_pixel_vision.py
```

Run the synthetic pixel benchmark. Use real llama.cpp text synthesis with mock vision for offline pipeline checks, or remove `--vision-provider mock` when the unified VLM server is running:

```bash
python evaluate_fidelity.py --benchmark benchmarks/pixel_vision/gold.jsonl --provider llama.cpp --model qwen3.5-9b-vlm --use-vision --vision-provider mock --target-recall 0.80
```

The evaluator reports `vision_fact_recall`, category recall for `vision`, schema validity, latency, validation attempts, generated dossier text, and missed visual facts. This is a first-stage pixel benchmark, not a claim of 95% full-paper multimodal fidelity.

Validated local Qwen3.5 VLM result on RTX 4060 Laptop 8GB VRAM:

```text
Qwen3.5 VLM health check: passed
text schema smoke with qwen3.5-9b-vlm: passed, 8.83 seconds
vision schema smoke with qwen3.5-9b-vlm: passed, 6.62 seconds
pixel vision benchmark: 3 PDFs, 8/9 visual facts, 88.9% vision_fact_recall
pixel benchmark schema_valid_rate: 100%
pixel benchmark finding_source_trace_rate: 100%
```

The pixel benchmark uses deterministic matching with optional gold aliases for equivalent phrasing, while still reporting missed facts instead of lowering the target.

If Qwen3.5-9B + `mmproj` is unstable or too tight on 8GB VRAM, fall back to the officially listed smaller llama.cpp multimodal repos such as `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF` by overriding `LITAGENT_VISION_HF_REPO`, port, alias, and Streamlit/evaluator vision settings.

## vLLM Fallback Notes

vLLM remains in `llm_service/` as an experimental fallback. It was installed successfully in WSL with CUDA, and CUDA visibility was confirmed. However, the current vLLM GGUF loader failed on the target Qwen3.5 GGUF with:

```text
RuntimeError: Unknown gguf model_type: qwen3_5
```

For this project, llama.cpp is the better primary inference server because it loaded the same GGUF model directly and completed schema-constrained end-to-end validation.

## Current Limitations

- The default demo uses mock abstracts unless live retrieval or PDF input is selected.
- Full-paper layout support covers layout/table/caption facts; pixel-level chart or figure reasoning requires the optional vision server and is currently evaluated on a small synthetic benchmark.
- Live Semantic Scholar retrieval may require an API key; unauthenticated requests can return HTTP 429 rate-limit errors.
